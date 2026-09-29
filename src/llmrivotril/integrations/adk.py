"""Google ADK (Agent Development Kit) custom-model adapter.

Wraps a ``RivotrilAgent`` as an ADK ``BaseLlm``, so every ADK agent call
runs through llmrivotril's guardrails/PII redaction/RAG/observability.

ADK resends the full conversation on every ``generate_content_async()``
call via ``llm_request.contents``; this adapter flattens it into one prompt
via ``flatten_messages`` rather than relying on ``RivotrilAgent``'s own
``memory=`` store.

Streaming (``stream=True``) and multimodal parts (images/audio/function
responses) are not implemented -- only the first text part of each content
is read, and a single non-streaming ``LlmResponse`` is yielded.

Requires: pip install "llmrivotril[adk]"
"""

from collections.abc import AsyncGenerator
from typing import Any

try:
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        'google-adk is required for RivotrilLlm. Install with: pip install "llmrivotril[adk]"'
    ) from exc

from ..agent import RivotrilAgent
from ._common import flatten_messages


def _contents_to_dicts(contents: list[Any]) -> list[dict[str, str]]:
    return [
        {
            "role": content.role or "user",
            "content": "".join(part.text or "" for part in (content.parts or [])),
        }
        for content in contents
    ]


class RivotrilLlm(BaseLlm):
    """Adapts a ``RivotrilAgent`` to ADK's ``BaseLlm`` interface."""

    agent: RivotrilAgent
    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, agent: RivotrilAgent, **kwargs: Any) -> None:
        # pydantic validates against this class's full field set (model +
        # agent) even though the call is written as super().__init__() --
        # mypy only sees BaseLlm's own fields here (hence the call-arg
        # override for this module below), but omitting `agent` would fail
        # pydantic's "field required" validation at runtime.
        super().__init__(model=agent.model, agent=agent, **kwargs)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        prompt = flatten_messages(_contents_to_dicts(llm_request.contents))
        text = await self.agent.run_async(prompt)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
