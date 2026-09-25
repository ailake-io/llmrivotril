from unittest.mock import AsyncMock, MagicMock

import pytest

from llmrivotril import RivotrilAgent, ToolCall, ToolRegistry
from llmrivotril.providers import BaseProvider, ProviderResponse


class _MockProvider(BaseProvider):
    name = "mock"

    def __init__(self):
        super().__init__()
        self._complete_mock = MagicMock()
        self._acomplete_mock = AsyncMock()

    def complete(self, *args, **kwargs):
        return self._complete_mock(*args, **kwargs)

    def acomplete(self, *args, **kwargs):
        return self._acomplete_mock(*args, **kwargs)


def _add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_tool_registry_builds_schema_from_callable():
    registry = ToolRegistry([_add])

    assert len(registry.schemas) == 1
    schema = registry.schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "_add"
    assert "a" in schema["function"]["parameters"]["properties"]


def test_tool_registry_executes_callable():
    registry = ToolRegistry([_add])
    call = ToolCall(id="call_1", name="_add", arguments={"a": 2, "b": 3})

    result = registry.execute(call)

    assert result == "5"


def test_tool_registry_accepts_openai_schema():
    schema = {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    registry = ToolRegistry([schema])
    assert len(registry.schemas) == 1


def test_agent_executes_tool_and_returns_final_answer():
    provider = _MockProvider()
    agent = RivotrilAgent(api_key="test", provider=provider)

    # First call asks for a tool; second call returns the final answer.
    provider._complete_mock.side_effect = [
        ProviderResponse(
            content=None,
            tool_calls=[{"id": "call_1", "name": "_add", "arguments": {"a": 1, "b": 2}}],
        ),
        ProviderResponse(content="The sum is 3."),
    ]

    result = agent.run("What is 1+2?", tools=[_add])

    assert result == "The sum is 3."
    assert provider._complete_mock.call_count == 2


@pytest.mark.asyncio
async def test_agent_executes_tool_async():
    provider = _MockProvider()
    agent = RivotrilAgent(api_key="test", provider=provider)

    provider._acomplete_mock.side_effect = [
        ProviderResponse(
            content=None,
            tool_calls=[{"id": "call_2", "name": "_add", "arguments": {"a": 5, "b": 7}}],
        ),
        ProviderResponse(content="12"),
    ]

    result = await agent.run_async("What is 5+7?", tools=[_add])

    assert result == "12"
    assert provider._acomplete_mock.call_count == 2


def test_agent_tool_error_is_returned_to_model():
    provider = _MockProvider()
    agent = RivotrilAgent(api_key="test", provider=provider)

    def _fail():
        raise RuntimeError("boom")

    provider._complete_mock.side_effect = [
        ProviderResponse(
            content=None,
            tool_calls=[{"id": "call_3", "name": "_fail", "arguments": {}}],
        ),
        ProviderResponse(content="Got an error."),
    ]

    result = agent.run("Call fail.", tools=[_fail])

    assert result == "Got an error."


def test_agent_skips_tool_calling_with_response_model():
    from pydantic import BaseModel

    class Answer(BaseModel):
        value: int

    provider = _MockProvider()
    agent = RivotrilAgent(api_key="test", provider=provider)
    answer = Answer(value=3)
    provider._complete_mock.return_value = ProviderResponse(structured=answer)

    result = agent.run("What is 1+2?", response_model=Answer, tools=[_add])

    assert result == answer
    assert provider._complete_mock.call_count == 1
