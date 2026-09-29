# Async, Streaming & Function Calling

[← Back to README](../README.md)

## Async Usage

```python
import asyncio

async def main():
    agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"))
    response = await agent.run_async("Hello!")
    print(response)

asyncio.run(main())
```

## Streaming

```python
for chunk in agent.run_stream("Tell me a short story."):
    print(chunk, end="", flush=True)

# async version: agent.run_stream_async(...)
```

`tools=` is supported: a turn where the model answers directly still streams
token by token, and a turn where it calls a tool falls back to a single
blocking round-trip for that turn (tool-call argument deltas can't be
usefully streamed to the caller).

`response_model=` is also supported, but returns a `StreamedStructuredResult`
instead of a plain iterator, since partial JSON isn't a valid model -- there's
nothing to type-check until the stream ends:

```python
stream = agent.run_stream("Describe a planet.", response_model=Planet)
for chunk in stream:
    print(chunk, end="", flush=True)  # raw JSON text as it streams

planet = stream.result  # the validated Planet instance, set once the loop above ends
```

`response_model=` and `tools=` can't be combined in `run_stream()`/
`run_stream_async()` (raises `ValueError`) -- use `agent.run(...)` for that.

Output guardrails, grounding verification, and memory/metrics updates run
against the full response only after the stream ends, so a blocked or
ungrounded response still raises after you've already received its chunks
(or, for `response_model=`, when you next access `.result` / finish the
`for` loop).

## Function Calling

Give the agent tools as callables or OpenAI-style schemas:

```python
from llmrivotril import RivotrilAgent

def get_weather(city: str) -> str:
    """Return the weather for a city."""
    return f"Sunny in {city}."

agent = RivotrilAgent(api_key="sk-...")
response = agent.run("What is the weather in Paris?", tools=[get_weather])
```

The agent executes the requested tool call and returns the model's final answer.
