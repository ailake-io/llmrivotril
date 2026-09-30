"""AG2 1.x custom model adapter.

AG2 (formerly AutoGen) rewrote its entire API in the 1.0 line -- there is no
``AssistantAgent``/``register_model_client`` mechanism anymore (that only
exists in ``ag2<1.0``; see ``docs/integrations.md`` if you need that older
integration instead). This module targets AG2 1.x's ``Agent(config=...)``
extension point: :class:`RivotrilModelConfig` implements the (structural,
non-``runtime_checkable``) ``ag2.config.config.ModelConfig`` protocol, and
its ``create()`` returns a :class:`RivotrilLLMClient` implementing
``ag2.config.client.LLMClient``.

Unlike the CrewAI/LangChain/ADK adapters, this one needs ``ag2`` actually
importable at module load time -- AG2 1.x's ``ModelMessage``/``ModelResponse``
are real event classes this module constructs directly, not plain dicts, so
there's no way to stay import-free the way the pre-1.0 protocol-only version
of this adapter could.

AG2 resends the full conversation as a sequence of event objects on every
call; this adapter renders each one with AG2's own ``render_for_prompt()``
and flattens the result into one prompt via ``flatten_messages`` rather than
relying on ``RivotrilAgent``'s own ``memory=`` store.

Not implemented: tool-calling and structured output. AG2 passes `tools=`/
`response_schema=` into the ``LLMClient`` call, but this adapter ignores them
and always returns a plain-text ``ModelResponse`` -- AG2 treats that the same
as "the model chose not to call a tool", not as an error, so this degrades
safely rather than breaking. Streaming is not implemented either.

Requires: pip install "llmrivotril[autogen]"
"""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, cast

try:
    from ag2.config.config import ModelProvider
    from ag2.events import BaseEvent, ModelMessage, ModelResponse
    from ag2.events import render_for_prompt as _render_for_prompt
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        'ag2 is required for RivotrilModelConfig. Install with: pip install "llmrivotril[autogen]"'
    ) from exc

if TYPE_CHECKING:
    from ag2.config.client import LLMClient
    from ag2.context import ConversationContext
    from ag2.response import ResponseProto
    from ag2.tools.schemas import ToolSchema

from ..agent import RivotrilAgent
from ._common import flatten_messages

_PROVIDER_BY_NAME = {
    "openai": ModelProvider.OPENAI,
    "azure_openai": ModelProvider.OPENAI,
    "anthropic": ModelProvider.ANTHROPIC,
    "gemini": ModelProvider.GEMINI,
    "bedrock": ModelProvider.BEDROCK,
}


class RivotrilLLMClient:
    """AG2 1.x ``LLMClient`` backed by a ``RivotrilAgent``."""

    def __init__(self, agent: RivotrilAgent) -> None:
        self.agent = agent

    async def __call__(
        self,
        messages: Sequence[BaseEvent],
        context: "ConversationContext",
        *,
        tools: Iterable["ToolSchema"],
        response_schema: "ResponseProto | None",
        serializer: Any,
    ) -> ModelResponse:
        rendered = [_render_for_prompt(event) for event in messages]
        prompt = flatten_messages([{"role": "user", "content": text} for text in rendered if text])
        text = await self.agent.run_async(prompt)
        return ModelResponse(message=ModelMessage(text))


class RivotrilModelConfig:
    """AG2 1.x ``ModelConfig`` backed by a ``RivotrilAgent``.

    Usage::

        agent = RivotrilAgent(...)
        worker = ag2.Agent("worker", config=RivotrilModelConfig(agent))
        reply = worker.ask("Hello")
    """

    def __init__(self, agent: RivotrilAgent) -> None:
        self.agent = agent

    @property
    def provider(self) -> ModelProvider:
        # Informational/routing metadata only -- best-effort mapping from
        # the agent's actual provider; RivotrilAgent's own guardrails/PII/
        # cache layer is what actually runs regardless of this label.
        return _PROVIDER_BY_NAME.get(getattr(self.agent.provider, "name", ""), ModelProvider.OPENAI)

    @property
    def model(self) -> str:
        return cast(str, self.agent.model)

    def copy(self) -> "RivotrilModelConfig":
        return RivotrilModelConfig(self.agent)

    def create(self) -> "LLMClient":
        return RivotrilLLMClient(self.agent)
