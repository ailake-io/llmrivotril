"""Real AG2 event/schema classes with a local agent double."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel

pytest.importorskip("ag2")

from ag2.events import HumanMessage, ModelMessageChunk  # noqa: E402
from ag2.response import ResponseSchema  # noqa: E402
from ag2.tools.final.function_tool import FunctionDefinition, FunctionToolSchema  # noqa: E402

from llmrivotril.integrations.autogen import RivotrilLLMClient, RivotrilModelConfig  # noqa: E402
from llmrivotril.providers import ProviderResponse  # noqa: E402


@pytest.mark.asyncio
async def test_ag2_returns_tool_events_for_orchestrator_execution():
    agent = SimpleNamespace(
        run_async=AsyncMock(
            return_value=ProviderResponse(
                tool_calls=[
                    {"id": "local-call", "name": "add", "arguments": {"a": 1}},
                ]
            )
        )
    )
    context = SimpleNamespace(prompt=["system context"], send=AsyncMock())
    client = RivotrilLLMClient(agent)
    result = await client(
        [HumanMessage("hello")],
        context,
        tools=[
            FunctionToolSchema(
                function=FunctionDefinition(name="add", parameters={"type": "object"})
            )
        ],
        response_schema=None,
        serializer=None,
    )
    call = result.tool_calls.calls[0]
    assert call.name == "add" and call.id == "local-call"
    assert call.serialized_arguments == {"a": 1}
    kwargs = agent.run_async.call_args.kwargs
    assert kwargs["execute_tools"] is False
    assert kwargs["tools"][0]["function"]["name"] == "add"
    assert "system context" in agent.run_async.call_args.args[0]


class Answer(BaseModel):
    value: int


@pytest.mark.asyncio
async def test_custom_ag2_validator_is_not_executed_twice():
    import ag2
    from ag2.response.callable import CallableResponse

    calls = []

    async def validate(text, context, provider):
        calls.append(text)
        return text.upper()

    agent = SimpleNamespace(
        model="local",
        provider=SimpleNamespace(name="openai"),
        run_async=AsyncMock(return_value="ok"),
    )
    worker = ag2.Agent("local", config=RivotrilModelConfig(agent))
    schema = CallableResponse(validate, name="local", schema=None)
    reply = await worker.ask("hello", response_schema=schema)
    assert await reply.content() == "OK"
    assert calls == ["ok"]


@pytest.mark.asyncio
async def test_ag2_validates_structured_output_with_real_response_schema():
    agent = SimpleNamespace(run_async=AsyncMock(return_value='{"value": 42}'))
    context = SimpleNamespace(prompt=[], send=AsyncMock())
    client = RivotrilLLMClient(agent)
    schema = ResponseSchema(Answer, embed=False)
    result = await client(
        [HumanMessage("answer")], context, tools=[], response_schema=schema, serializer=None
    )
    assert result.content == '{"value": 42}'
    agent.run_async.return_value = '{"value": "invalid"}'
    with pytest.raises(ValueError):
        await client(
            [HumanMessage("answer")], context, tools=[], response_schema=schema, serializer=None
        )


@pytest.mark.asyncio
async def test_ag2_stream_sends_native_chunk_events():
    closed = []

    async def stream(prompt):
        try:
            yield "O"
            yield "K"
        finally:
            closed.append(True)

    agent = SimpleNamespace(run_stream_async=stream)
    context = SimpleNamespace(prompt=[], send=AsyncMock())
    config = RivotrilModelConfig(agent, stream=True)
    client = config.copy().create()
    result = await client(
        [HumanMessage("hello")], context, tools=[], response_schema=None, serializer=None
    )
    chunks = [
        c.args[0] for c in context.send.call_args_list if isinstance(c.args[0], ModelMessageChunk)
    ]
    assert [c.content for c in chunks] == ["O", "K"]
    assert result.content == "OK"
    assert closed == [True]


@pytest.mark.asyncio
async def test_real_ag2_agent_executes_tool_with_local_provider():
    import ag2

    from llmrivotril import RivotrilAgent
    from llmrivotril.metrics import MetricsCollector
    from llmrivotril.providers import BaseProvider

    executed = []

    @ag2.tool
    def add(a: int, b: int) -> int:
        """Add two integers locally."""
        executed.append((a, b))
        return a + b

    class LocalProvider(BaseProvider):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def complete(self, **kwargs):
            self.calls += 1
            if self.calls == 1:
                assert kwargs["tools"][0]["function"]["name"] == "add"
                return ProviderResponse(
                    tool_calls=[
                        {"id": "add-local", "name": "add", "arguments": {"a": 2, "b": 3}},
                    ]
                )
            return ProviderResponse(content="5")

        async def acomplete(self, **kwargs):
            return self.complete(**kwargs)

    provider = LocalProvider()
    agent = RivotrilAgent(provider=provider, metrics=MetricsCollector(include_content=False))
    worker = ag2.Agent("local", config=RivotrilModelConfig(agent))
    reply = await worker.ask("Add 2 and 3", tools=[add])
    assert await reply.content() == "5"
    assert executed == [(2, 3)]
    assert provider.calls == 2
