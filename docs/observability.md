# Observability: Metrics, Dashboard, Benchmark & Cost Tracking

*[Português](observability.pt-BR.md)*

[← Back to README](../README.md)

## Automatic Metrics Persistence

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

## Local Dashboard

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
