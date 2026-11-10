# Observability: Metrics, Dashboard, Benchmark & Cost Tracking

*[Português](observability.pt-BR.md)*

[← Back to README](../README.md)

## Automatic Metrics Persistence

`get_summary()["provider_usage"]` reports every visible SDK invocation: `calls`,
`failures`, `input_tokens`, `output_tokens`, `estimated_calls`, `unpriced_calls`
and `cost_usd`. It includes repairs, tool continuations and memory summaries,
and excludes cache hits. Streaming uses reported OpenAI usage when available;
otherwise tokens are estimated. Prices require `track_costs=True` or explicit
session-budget rates; unknown prices are counted as unpriced, not free calls.
With a financial budget, unknown usage retains the conservative cost reserve.
These aggregates are persisted/reset with metrics. They are separate from the
legacy request-level `total_cost_usd`/`tokens`, which must not be added to them.
Opaque retries inside SDKs, external tools and provider fees are not an invoice.

Set `RIVOTRIL_METRICS_PATH` (or pass `metrics_path=`) to persist telemetry to JSON automatically:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    metrics_path="metrics.json",
)
```

Restore later with:

```python
from llmrivotril.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.load_metrics("metrics.json")
```

`MetricsCollector(max_logs=100)` bounds the in-memory audit trail. Set
`max_logs=0` when only aggregate counters are needed and request content must
not be retained in the metrics audit trail. This does not disable conversation
memory or the agent's token-count cache. The latest 1000 latency samples are
retained; the cumulative average uses constant-size counters. Error counts and
latency aggregates survive persistence, including loading legacy snapshots.

## Local Dashboard

For metadata-only telemetry, inject
`MetricsCollector(include_content=False)` into `RivotrilAgent(metrics=...)`.
Prompt, response and error details become `[omitted]` before retention or
persistence, including when loading old snapshots. Counters, timings, tokens
and error presence remain available. This setting applies to metrics only;
conversation memory, caches and application logging have separate policies.

Run `pytest -q tests/test_http_contracts.py` in the development venv for offline
HTTP contracts. The installed OpenAI SDK and Azure adapter use
`httpx.MockTransport`: no sockets, accounts or real credentials. Tests cover
sync/async completion, SSE, status errors, timeout mapping, SDK retry and tool
serialization. The retry case tests the SDK, not the agent retry layer.
Native providers and AG2 still need additional contracts; these tests do not
establish live service compatibility.

Start the telemetry dashboard:

```bash
llmrivotril dashboard --port 8000
```

Then open http://127.0.0.1:8000 in your browser. The dashboard works offline:
Tailwind CSS is bundled with the package instead of fetched from a CDN.

The dashboard exposes:

- `/` — HTML dashboard
- `/api/metrics` — JSON telemetry summary
- `/api/health` — Health check with version and UTC timestamp

Every endpoint is rate-limited per client IP (default 60 requests/minute,
tune with `RIVOTRIL_DASHBOARD_RATE_LIMIT_MAX_CALLS`/
`RIVOTRIL_DASHBOARD_RATE_LIMIT_PER_SECONDS`) to blunt brute-forcing the
dashboard token(s) and casual abuse.

Give each client its own revocable, labeled token instead of one shared
secret with `RIVOTRIL_DASHBOARD_TOKENS`:

```bash
export RIVOTRIL_DASHBOARD_TOKENS="alice:tok-for-alice,bob:tok-for-bob"
```

A matching token's label is logged on each successful auth, so dashboard
access can be attributed to whoever holds that token. `RIVOTRIL_DASHBOARD_TOKEN`
(singular) still works as a single unlabeled token. There's still no
per-user permissions -- every valid token gets full access -- so this
dashboard remains meant for local or trusted-network use, not multi-tenant
or public exposure; put a real auth/rate-limiting proxy in front of it for
that.

## Cache Hit Visibility

`get_summary()`'s `tokens` figure (and each per-request log's `tokens`) is a
local tiktoken estimate over the prompt+response text -- identical whether a
request actually hit the provider or was served from cache, since the text
is the same either way. To see whether caching is actually avoiding repeated
API calls, use `cache_hits`/`cache_hit_rate` in the summary, and the
per-request `cache_hit` boolean in `logs`:

```python
agent = RivotrilAgent(api_key="sk-...", cache=InMemoryCache())
agent.run("Explain what a guardrail is.")
agent.run("Explain what a guardrail is.")

summary = agent.metrics.get_summary()
print(summary["cache_hits"], summary["cache_hit_rate"])  # 1, 50.0
print(summary["logs"][0]["cache_hit"])  # True -- the most recent call, a hit
```

The dashboard's "Cache Hit Rate" tile and each row's "Cache Hit" badge in the
audit trail table reflect this same data. `/api/metrics/prometheus` exposes
it as `llmrivotril_cache_hits_total`/`llmrivotril_cache_hit_rate`.

## Benchmark

Run the built-in red-team benchmark with deterministic mocks (no API key, no cost):

```bash
llmrivotril benchmark --mock
```

It reports accuracy, false positives, and false negatives for guardrail and verifier behavior.

## Cost Tracking

Estimate spend per request and accumulate it in metrics:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    api_key="sk-...",
    track_costs=True,
)

agent.run("Hello")
print(agent.metrics.get_summary()["total_cost_usd"])
```

For providers or models not in the built-in table, register custom pricing:

```python
from llmrivotril import register_pricing

register_pricing("my-provider", "my-model", input_price=1.0, output_price=2.0)
```
