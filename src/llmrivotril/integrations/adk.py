"""Google ADK (Agent Development Kit) custom-model adapter.

Wraps a ``RivotrilAgent`` as an ADK ``BaseLlm``, so every ADK agent call
runs through llmrivotril's guardrails/PII redaction/RAG/observability.

ADK resends the full conversation on every ``generate_content_async()``
call via ``llm_request.contents``; this adapter flattens it into one prompt
via ``flatten_messages`` rather than relying on ``RivotrilAgent``'s own
``memory=`` store.

Streaming uses ``RivotrilAgent.run_stream_async()`` when ``stream=True``.
Text, image, audio and file parts are translated to the normalized content
format accepted by ``RivotrilAgent``.

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
    result: list[dict[str, Any]] = []
    for content in contents:
        parts: list[dict[str, Any]] = []
        for part in content.parts or []:
            if getattr(part, "text", None):
                parts.append({"type": "text", "text": part.text})
                continue
            inline_data = getattr(part, "inline_data", None)
            if inline_data is not None and getattr(inline_data, "data", None) is not None:
                import base64

                encoded = base64.b64encode(inline_data.data).decode("ascii")
                mime = getattr(inline_data, "mime_type", "application/octet-stream")
                if mime.startswith("image/"):
                    parts.append(
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}
                    )
                elif mime.startswith("audio/"):
                    parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {"data": encoded, "format": mime.split("/", 1)[1]},
                        }
                    )
                continue
            file_data = getattr(part, "file_data", None)
            if file_data is not None and getattr(file_data, "file_uri", None):
                parts.append({"type": "file", "file": {"uri": file_data.file_uri}})
        result.append({"role": content.role or "user", "content": parts})
    return result


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
        if stream:
            async for text in self.agent.run_stream_async(prompt):
                yield LlmResponse(
                    content=types.Content(role="model", parts=[types.Part(text=text)])
                )
            return

        text = await self.agent.run_async(prompt)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=text)]))
