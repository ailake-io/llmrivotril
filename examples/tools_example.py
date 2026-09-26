"""Function calling example for LLM-Rivotril (mock mode, no API key).

Usage:
    python examples/tools_example.py
"""

from llmrivotril import RivotrilAgent
from llmrivotril.providers import BaseProvider, ProviderResponse


class _MockToolProvider(BaseProvider):
    """Mock provider that always asks to call get_weather."""

    def __init__(self) -> None:
        self._turn = 0

    def complete(self, messages, model, response_model=None, tools=None, **kwargs):
        self._turn += 1
        if self._turn == 1 and tools:
            return ProviderResponse(
                content=None,
                tool_calls=[
                    {"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}
                ],
            )
        return ProviderResponse(content="The weather in Paris is sunny.")

    async def acomplete(self, messages, model, response_model=None, tools=None, **kwargs):
        return self.complete(messages, model, response_model, tools, **kwargs)


def get_weather(city: str) -> str:
    """Return the current weather for a city."""
    return f"Sunny in {city}."


def main():
    agent = RivotrilAgent(
        api_key="mock-key",
        provider=_MockToolProvider(),
    )

    response = agent.run(
        "What is the weather in Paris?",
        tools=[get_weather],
    )
    print(response)


if __name__ == "__main__":
    main()
