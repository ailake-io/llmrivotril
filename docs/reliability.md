# Reliability: Resilience, Caching & Schema Repair

*[Português](reliability.pt-BR.md)*

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

Retries use transient exception types declared by the selected provider. This
prevents an Anthropic, Cohere, Gemini, or Bedrock adapter from being governed
only by OpenAI exception classes. Unknown custom providers retain the
OpenAI-compatible defaults unless they override
`BaseProvider.retryable_exceptions()`.

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

### Semantic Cache *(optional, off by default)*

`InMemoryCache`/`DiskCache`/`RedisCache` only hit on a byte-identical repeated
prompt. `SemanticCache` wraps any of them with an embedding-similarity lookup,
so a paraphrased repeat ("what's 2 plus 2?" vs. "what is 2+2?") hits too.
Requires `pip install "llmrivotril[semantic]"` (loaded lazily, same as the
semantic guardrails):

```python
from llmrivotril import RivotrilAgent
from llmrivotril.semantic_cache import SemanticCache

agent = RivotrilAgent(
    api_key="sk-...",
    cache=SemanticCache(
        similarity_threshold=0.95
    ),  # backend=DiskCache(...)/RedisCache(...) also accepted
)
```

**This trades precision for recall.** A prompt that's *similar but not
equivalent* (different numbers, a negated question, a changed constraint) can
embed close enough to return a wrong cached answer with full confidence. Keep
`similarity_threshold` conservative and only opt in where that trade-off is
acceptable -- it is a separate class specifically so it's never a silent
default.

## Memory Token Budget & Summarization

Two more, independent opt-ins for `RivotrilAgent`'s default `MemoryStore`
(both no-ops unless configured; neither applies if you pass your own
`memory=` instance):

```python
agent = RivotrilAgent(
    api_key="sk-...",
    memory_max_tokens=2000,  # trims oldest turns by actual token count
    memory_summarize=True,  # compacts trimmed turns into a summary instead of dropping them
    memory_summarize_trigger_turns=20,
)
```

- `memory_max_tokens` layers a token-budget cap on top of the existing
  `retention_window` turn-count cap -- turns vary a lot in length, so this
  bounds prompt growth in a long conversation far more directly than a fixed
  turn count. No behavior change unless set (`None` by default).
- `memory_summarize=True` replaces the turns that would otherwise be silently
  dropped with a short LLM-generated summary once history passes
  `memory_summarize_trigger_turns`, via a direct provider call (bypasses
  guardrails/cache/retry -- internal housekeeping, not a user-facing turn). A
  failed summarization call is logged and skipped rather than raised; the
  turns are still trimmed either way. When summarization fires, at most
  `memory_summarize_trigger_turns` raw turns are kept (capped by
  `retention_window * 2`), so a trigger below that cap takes effect as set.
  This costs one extra LLM call per
  summarization round to save tokens on every turn afterward -- only worth it
  for genuinely long conversations.

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
