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

Function tool requests are returned as native AG2 events for the orchestrator
to execute. ResponseProto validates structured text. ``stream=True`` emits
native text chunk events; tool turns use completion mode. Tool schemas and
structured output cannot currently be combined in one turn. Validated with
AG2 1.3.0 using local providers and real orchestration.

Requires: pip install "llmrivotril[autogen]"
"""

import json
from collections.abc import Iterable, Sequence
from contextlib import aclosing
from typing import TYPE_CHECKING, Any, cast

try:
    from ag2.config.config import ModelProvider
    from ag2.events import (
        BaseEvent,
        ModelMessage,
        ModelMessageChunk,
        ModelResponse,
        ToolCallEvent,
        ToolCallsEvent,
    )
    from ag2.events import render_for_prompt as _render_for_prompt
    from ag2.response import ResponseSchema
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
from ..content import content_to_text
from ..tools import normalize_tool_calls
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

    def __init__(self, agent: RivotrilAgent, stream: bool = False) -> None:
        self.agent = agent
        self.stream = stream

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
        prompt = content_to_text(
            flatten_messages([{"role": "user", "content": text} for text in rendered if text])
        )
        context_prompt = getattr(context, "prompt", ()) if context is not None else ()
        if context_prompt:
            prompt = "\n\n".join([*context_prompt, prompt])
        schemas = []
        for tool in tools:
            if tool.type != "function" or getattr(tool, "defer_loading", False):
                raise ValueError("Rivotril AG2 adapter supports eager function tools only")
            function = getattr(tool, "function", None)
            if function is None:
                raise ValueError("Function tool requires a function definition")
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": function.name,
                        "description": function.description,
                        "parameters": function.parameters,
                    },
                }
            )
        if schemas and response_schema is not None:
            raise ValueError("Use tools and structured output in separate AG2 turns")
        if response_schema is not None:
            prompt += "\n" + (response_schema.system_prompt or "")
            if response_schema.json_schema is not None:
                prompt += "\nReturn JSON matching: " + json.dumps(response_schema.json_schema)
        if schemas:
            response = await self.agent.run_async(
                prompt,
                tools=schemas,
                execute_tools=False,
                return_provider_response=True,
            )
            calls = normalize_tool_calls(response)
            return ModelResponse(
                message=ModelMessage(response.text) if response.text else None,
                tool_calls=ToolCallsEvent(
                    [
                        ToolCallEvent(
                            id=call.id,
                            name=call.name,
                            arguments=json.dumps(call.arguments),
                        )
                        for call in calls
                    ]
                ),
            )
        if self.stream:
            if context is None:
                raise ValueError("AG2 streaming requires a context with send()")
            chunks = []
            async with aclosing(cast(Any, self.agent.run_stream_async(prompt))) as stream:
                async for chunk in stream:
                    chunks.append(chunk)
                    await context.send(ModelMessageChunk(chunk))
            text = "".join(chunks)
        else:
            text = await self.agent.run_async(prompt)
        if isinstance(response_schema, ResponseSchema):
            await response_schema.validate(text, context)
        message = ModelMessage(text)
        if context is not None:
            await context.send(message)
        return ModelResponse(message=message)


class RivotrilModelConfig:
    """AG2 1.x ``ModelConfig`` backed by a ``RivotrilAgent``.

    Usage::

        agent = RivotrilAgent(...)
        worker = ag2.Agent("worker", config=RivotrilModelConfig(agent))
        reply = worker.ask("Hello")
    """

    def __init__(self, agent: RivotrilAgent, stream: bool = False) -> None:
        self.agent = agent
        self.stream = stream

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
        return RivotrilModelConfig(self.agent, stream=self.stream)

    def create(self) -> "LLMClient":
        return RivotrilLLMClient(self.agent, stream=self.stream)
