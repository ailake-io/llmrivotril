# Reliability: Resilience, Caching & Schema Repair

[← Back to README](../README.md)

## Resilience

Protect LLM calls from transient failures and overload:

```python
agent = RivotrilAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    rate_limit_max_calls=10,
    rate_limit_per_seconds=1,
    retry_max_attempts=3,
    retry_min_wait=1.0,
    retry_max_wait=10.0,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
)
```

## Response Caching

Avoid repeated LLM calls for identical prompts by enabling a cache:

```python
from llmrivotril import RivotrilAgent, InMemoryCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=InMemoryCache(),
)
```

For persistent caching across restarts, use `DiskCache` or set `RIVOTRIL_CACHE_PATH`:

```python
from llmrivotril import RivotrilAgent, DiskCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=DiskCache("llm_cache.sqlite3"),
)
```

For multi-process or distributed deployments, use `RedisCache` (requires
`pip install "llmrivotril[redis]"`):

```python
from llmrivotril import RivotrilAgent, RedisCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=RedisCache(url="redis://localhost:6379/0"),
)
```

## Schema-Repair Fallback

When structured outputs fail Pydantic validation, the agent can ask the model to
fix its response:

```python
from pydantic import BaseModel
from llmrivotril import RivotrilAgent

class Answer(BaseModel):
    answer: str

agent = RivotrilAgent(
    api_key="sk-...",
    schema_repair_attempts=2,
)

agent.run("Return a JSON answer.", response_model=Answer)
```
