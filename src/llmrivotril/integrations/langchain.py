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

Streaming delegates to ``RivotrilAgent.run_stream``/``run_stream_async``.
``bind_tools`` stores LangChain tools on a copied model and forwards them to
the agent, including tools that expose the standard ``name``, ``args_schema``
and ``invoke`` protocol.

Requires: pip install "llmrivotril[langchain]"
"""

from collections.abc import Callable, Sequence
from typing import Any

from pydantic import Field

try:
    from langchain_core.callbacks import (
        AsyncCallbackManagerForLLMRun,
        CallbackManagerForLLMRun,
    )
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
    from langchain_core.tools import BaseTool
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        "langchain-core is required for RivotrilChatModel. "
        'Install with: pip install "llmrivotril[langchain]"'
    ) from exc

from ..agent import RivotrilAgent
from ._common import flatten_messages


def _messages_to_dicts(messages: list[BaseMessage]) -> list[dict[str, Any]]:
    return [{"role": message.type, "content": message.content} for message in messages]


class RivotrilChatModel(BaseChatModel):
    """Adapts a ``RivotrilAgent`` to LangChain's ``BaseChatModel`` interface."""

    agent: RivotrilAgent
    bound_tools: list[Any] = Field(default_factory=list)
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
        if self.bound_tools:
            text = self.agent.run(prompt, tools=self.bound_tools)
        else:
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
        if self.bound_tools:
            text = await self.agent.run_async(prompt, tools=self.bound_tools)
        else:
            text = await self.agent.run_async(prompt)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        prompt = flatten_messages(_messages_to_dicts(messages))
        if self.bound_tools:
            stream = self.agent.run_stream(prompt, tools=self.bound_tools)
        else:
            stream = self.agent.run_stream(prompt)
        for text in stream:
            if run_manager is not None:
                run_manager.on_llm_new_token(text)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Any:
        prompt = flatten_messages(_messages_to_dicts(messages))
        if self.bound_tools:
            stream = self.agent.run_stream_async(prompt, tools=self.bound_tools)
        else:
            stream = self.agent.run_stream_async(prompt)
        async for text in stream:
            if run_manager is not None:
                await run_manager.on_llm_new_token(text)
            yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        if tool_choice not in (None, "auto"):
            raise ValueError("RivotrilChatModel currently supports tool_choice=None or 'auto'")
        return self.model_copy(update={"bound_tools": list(tools)})
