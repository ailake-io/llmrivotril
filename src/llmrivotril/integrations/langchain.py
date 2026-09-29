"""LangChain custom BaseChatModel adapter.

Wraps a ``RivotrilAgent`` as a ``langchain_core`` chat model, so every
LangChain/LangGraph call into it runs through llmrivotril's guardrails/PII
redaction/RAG/observability.

LangChain resends the full message history on every ``_generate()`` call;
this adapter flattens it into one prompt via ``flatten_messages`` rather
than relying on ``RivotrilAgent``'s own ``memory=`` store. Async
(``_agenerate``) calls ``RivotrilAgent.run_async`` directly instead of
falling back to LangChain's default thread-pool wrapper around the sync
path.

Streaming and ``bind_tools`` are not implemented -- only plain
``invoke``/``ainvoke``.

Requires: pip install "llmrivotril[langchain]"
"""

from typing import Any

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "langchain-core is required for RivotrilChatModel. "
        'Install with: pip install "llmrivotril[langchain]"'
    ) from exc

from ..agent import RivotrilAgent
from ._common import flatten_messages


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, str]]:
    return [{"role": message.type, "content": str(message.content)} for message in messages]


class RivotrilChatModel(BaseChatModel):
    """Adapts a ``RivotrilAgent`` to LangChain's ``BaseChatModel`` interface."""

    agent: RivotrilAgent
    model_config = {"arbitrary_types_allowed": True}

    @property
    def _llm_type(self) -> str:
        return "rivotril"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = flatten_messages(_messages_to_dicts(messages))
        text = self.agent.run(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = flatten_messages(_messages_to_dicts(messages))
        text = await self.agent.run_async(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])
