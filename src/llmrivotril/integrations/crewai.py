"""CrewAI custom-LLM adapter.

Wraps a ``RivotrilAgent`` as a CrewAI-compatible LLM, so every CrewAI agent
call runs through llmrivotril's guardrails/PII redaction/RAG/observability.

CrewAI owns and resends the full task/conversation history on every call;
this adapter flattens it into one prompt via ``flatten_messages`` rather
than relying on ``RivotrilAgent``'s own ``memory=`` store, which would
double-count history the orchestrator already tracks.

Tool execution: if CrewAI supplies ``available_functions`` (name -> callable),
those callables are registered with llmrivotril's own ``ToolRegistry`` so
tool calls run under the same guardrails as the rest of the turn. Bare
``tools`` schemas without matching callables are passed through for the
provider to see, but can't be auto-executed by ``RivotrilAgent``.

CrewAI's ``stream_events()`` protocol is supported when the inherited
``stream`` flag is enabled: chunks from ``RivotrilAgent.run_stream()`` are
published through CrewAI's stream events and the final text is returned to the
executor.

Requires: pip install "llmrivotril[crewai]"
"""

from typing import Any, cast

from pydantic import BaseModel

try:
    from crewai.llms.base_llm import BaseLLM
    from crewai.utilities.types import LLMMessage
except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
    raise ImportError(
        'crewai is required for CrewAILLM. Install with: pip install "llmrivotril[crewai]"'
    ) from exc

from ..agent import RivotrilAgent
from ._common import flatten_messages


class CrewAILLM(BaseLLM):
    """Adapts a ``RivotrilAgent`` to CrewAI's ``BaseLLM`` interface."""

    def __init__(self, agent: RivotrilAgent, temperature: float | None = None) -> None:
        super().__init__(model=agent.model, temperature=temperature)
        self.agent = agent

    def call(
        self,
        messages: str | list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: type[BaseModel] | None = None,
    ) -> str | Any:
        prompt = messages if isinstance(messages, str) else flatten_messages(messages)
        agent_tools = list(available_functions.values()) if available_functions else tools
        if getattr(self, "_effective_stream", lambda: False)():
            result = self.agent.run_stream(
                prompt,
                tools=agent_tools,
                response_model=response_model,
            )
            if response_model is not None and hasattr(result, "result"):
                chunks = list(result)
                final = result.result
            else:
                chunks = list(result)
                final = "".join(chunks)
            for chunk in chunks:
                self._emit_stream_chunk_event(
                    chunk,
                    from_task=from_task,
                    from_agent=from_agent,
                )
            return final
        if response_model is not None:
            return self.agent.run(prompt, tools=agent_tools, response_model=response_model)
        # RivotrilAgent.run() is typed -> Any because it can also return a
        # response_model instance; without response_model= the result is
        # always the plain str CrewAI's call() promises in that case.
        return cast(str, self.agent.run(prompt, tools=agent_tools))

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        # RivotrilAgent.run() has no stop-sequence parameter; report this
        # honestly instead of letting CrewAI assume stop= is honored.
        return False
