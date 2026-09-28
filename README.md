# LLM-Rivotril

A lightweight Python framework to reduce LLM hallucinations, enforce guardrails, manage stateful memory, and monitor performance through a local web dashboard.

## Features

- **Guardrails** — Block disallowed keywords, enforce allowed topics, limit output size, and validate JSON schemas on outputs.
- **Semantic Guardrails** *(optional)* — Match prompts against allowed topics using dense embeddings instead of exact keywords.
- **Moderation Guardrail** — Flag adversarial/unsafe content via OpenAI's moderation endpoint.
- **Stateful Memory** — Sliding-window conversation store to prevent context drift.
- **Anti-Hallucination Verifier** — Pluggable grounding checks, including keyword overlap, citation markers, and embedding-based faithfulness.
- **Resilience** — Built-in rate limiting, retry with backoff, and circuit breaker for LLM calls.
- **Telemetry & Dashboard** — Built-in FastAPI dashboard with live request logs, token usage, latency, and success rate; optional Bearer-token auth.
- **Benchmark / Red-Team Evaluator** — Labeled suite to measure guardrail and verifier accuracy without API costs.
- **CLI** — Launch the dashboard with a single command.
- **Type-Safe Responses** — Optional Pydantic response models via `instructor`.
- **Async API** — `run_async()` for non-blocking execution.
- **OpenAI-Compatible APIs** — Use `base_url` to point to Ollama, vLLM, or other compatible servers.
- **Vector-Store Retrievers** *(optional)* — pgvector, Qdrant, Weaviate, and Pinecone adapters for RAG beyond in-memory scale.

## Installation

```bash
pip install llmrivotril
```

For semantic (embedding-based) guardrails and verifiers:

```bash
pip install llmrivotril[semantic]
```

For local development:

```bash
git clone https://github.com/ailake-io/llmrivotril.git
cd llmrivotril
pip install -e ".[dev,semantic]"
```

This installs test, lint, type-check, and packaging tools (`pytest`, `ruff`, `mypy`, `build`, `twine`) plus `sentence-transformers`.

## Quick Start

```python
import os
from llmrivotril import RivotrilAgent, Guardrail
from llmrivotril.verifier import KeywordOverlapVerifier

agent = RivotrilAgent(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    guardrails=[
        Guardrail(
            name="safe-content",
            allowed_topics=["AI safety", "machine learning"],
            disallowed_keywords=["password", "secret"],
            max_tokens=500,
        )
    ],
    verifier=KeywordOverlapVerifier(threshold=0.1),
)

response = agent.run("Explain what a guardrail is in AI safety.")
print(response)
```

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

## Semantic Guardrails

Use dense embeddings to accept semantically related prompts even when they do not contain exact topic keywords (requires `llmrivotril[semantic]`):

```python
from llmrivotril import SemanticTopicGuardrail

guardrail = SemanticTopicGuardrail(
    name="semantic-topics",
    allowed_topics=["machine learning", "software engineering"],
    similarity_threshold=0.5,
)

agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

## Moderation Guardrail

The keyword/regex/topic guardrails above are cheap local heuristics, not a
moderation system. `ModerationGuardrail` calls OpenAI's moderation endpoint
for adversarial/unsafe content they aren't meant to catch (no extra install
needed, `openai` is already a core dependency):

```python
from llmrivotril import ModerationGuardrail

guardrail = ModerationGuardrail(api_key=os.getenv("OPENAI_API_KEY"))
agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

It fails closed by default: if the moderation call itself errors (network,
rate limit), the request is blocked rather than silently let through. Pass
`fail_open=True` to let requests through instead when the moderation call
fails, or `check_input=False`/`check_output=False` to only check one side.

## RAG (Retrieval-Augmented Generation)

Build a local RAG pipeline to feed retrieved context into the agent:

```python
from llmrivotril import RAGPipeline, RivotrilAgent

pipeline = RAGPipeline()
pipeline.ingest("docs/")

agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"))
response = agent.run(
    "What is a guardrail?",
    context_sources=pipeline.query("What is a guardrail?"),
)
```

The pipeline includes `TextLoader`, `MarkdownLoader`, `SimpleChunker` and `InMemoryKeywordRetriever`. For dense retrieval, install `llmrivotril[semantic]` and use `InMemoryEmbeddingRetriever(cache_path="embeddings.json")` -- `cache_path` avoids recomputing embeddings for content already seen in a prior run.

Inside `agent.run_async()` or any other async code, use `await pipeline.aingest(...)`/`await pipeline.aquery(...)` instead of `ingest`/`query` so the (synchronous, CPU/IO-bound) loading and embedding work doesn't block the event loop.

### Vector-Store Retrievers

`InMemoryKeywordRetriever`/`InMemoryEmbeddingRetriever` scan a Python list on
every query -- fine locally, not for a corpus that needs to survive restarts
or scale past a few thousand documents. Swap in a retriever backed by an
external vector database instead (all implement the same
`add_documents`/`retrieve` interface, so they drop into `RAGPipeline(retriever=...)`
unchanged):

```python
from llmrivotril import PgVectorRetriever, RAGPipeline

retriever = PgVectorRetriever(dsn="postgresql://localhost/mydb")
pipeline = RAGPipeline(retriever=retriever)
pipeline.ingest("docs/")
```

| Retriever | Install | Notes |
|---|---|---|
| `PgVectorRetriever` | `llmrivotril[pgvector]` | Postgres + the `vector` extension; creates its table on first use. |
| `QdrantRetriever` | `llmrivotril[qdrant]` | `url=` for a remote/Docker instance, or `location=":memory:"` for in-process. |
| `WeaviateRetriever` | `llmrivotril[weaviate]` | Takes an already-connected v4 client (`weaviate.connect_to_local()` etc.) -- connection setup varies too much by deployment to guess. |
| `PineconeRetriever` | `llmrivotril[pinecone]` | The Pinecone index must already exist with a matching dimension (384 for the default embedding model); this class doesn't create it (billing-relevant, left explicit). |

`llmrivotril[vector-stores]` installs all four plus `llmrivotril[semantic]`
(all four compute embeddings via `sentence-transformers`, same as
`InMemoryEmbeddingRetriever`).

> **Not verified against a live service.** These four are built from each
> platform's documented client API and covered by mocked unit tests, but this
> project has no running Postgres/Qdrant/Weaviate/Pinecone instance or API
> keys to test against a real deployment -- please report issues if
> something doesn't match your installed client version (Weaviate's v4 API
> in particular changed significantly from v3).

## Embedding-Based Grounding Verification

Compare response and context embeddings to detect ungrounded answers (requires `llmrivotril[semantic]`):

```python
from llmrivotril import EmbeddingFaithfulnessVerifier
from llmrivotril.verifier import Verifier

agent = RivotrilAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    verifier=Verifier(check_fn=EmbeddingFaithfulnessVerifier(similarity_threshold=0.6).as_callable()),
)
```

## Model-Based Grounding Verification

For critical use cases, use an LLM as a judge to decide whether a response is grounded in the provided context:

```python
from llmrivotril import ModelBasedFaithfulnessVerifier
from llmrivotril.verifier import Verifier

verifier = ModelBasedFaithfulnessVerifier(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
)

agent = RivotrilAgent(
    api_key=os.getenv("OPENAI_API_KEY"),
    verifier=Verifier(check_fn=verifier.as_callable()),
)
```

## Quick Project Scaffolding

Create a new ready-to-use project with a single command:

```bash
llmrivotril init my-project
```

It generates `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, a package directory with `agent.py` / `guardrails.py`, and a starter `tests/test_agent.py`.

## Benchmark

Run the built-in red-team benchmark with deterministic mocks (no API key, no cost):

```bash
llmrivotril benchmark --mock
```

It reports accuracy, false positives, and false negatives for guardrail and verifier behavior.

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

## OpenAI-Compatible Servers

Point to a local or custom endpoint:

```python
agent = RivotrilAgent(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="unused",
)
```

## Other Providers

Use Anthropic, Cohere, or Gemini natively:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    provider="anthropic",
    model="claude-3-opus",
    api_key="...",
)
```

Install the optional SDKs:

```bash
pip install llmrivotril[providers]
```

Supported providers: `openai` (default), `azure_openai`, `anthropic`, `cohere`, `gemini`, `bedrock`.

The plain OpenAI-compatible path (`base_url=`) only works for endpoints that
mirror the plain OpenAI REST API -- Ollama, vLLM, LM Studio, OpenRouter,
Together.ai, etc. Azure OpenAI and AWS Bedrock have different auth/request
shapes and need their own adapters, below.

> **Not verified against a live account.** The `azure_openai` and `bedrock`
> adapters are built against each platform's documented API shape and covered
> by mocked unit tests, but this project has no Azure/AWS credentials to test
> against a real deployment. Please report issues if something doesn't match
> your account's behavior.

### Azure OpenAI

`provider="azure_openai"` (a string) has no way to pass Azure-specific
constructor arguments through `RivotrilAgent`, so construct the provider
instance yourself and pass that instead:

```python
from llmrivotril import AzureOpenAIProvider, RivotrilAgent

provider = AzureOpenAIProvider(
    api_key="...",
    azure_endpoint="https://your-resource.openai.azure.com",
    api_version="2024-02-01",
)
agent = RivotrilAgent(
    provider=provider,
    model="my-deployment-name",  # your Azure *deployment* name, not the model name
)
```

No extra install needed -- `AzureOpenAI`/`AsyncAzureOpenAI` ship in the
`openai` package, already a core dependency.

### AWS Bedrock

Same reasoning as Azure -- construct `BedrockProvider` directly for its
`region_name`:

```python
from llmrivotril import BedrockProvider, RivotrilAgent

provider = BedrockProvider(region_name="us-east-1")
agent = RivotrilAgent(
    provider=provider,
    model="anthropic.claude-3-5-sonnet-20241022-v2:0",  # a Bedrock model ID
)
```

Requires `pip install "llmrivotril[bedrock]"` (`boto3`) and AWS credentials
resolved the normal boto3 way (environment variables,
`~/.aws/credentials`, an instance role, etc.) -- there's no `api_key=`
for Bedrock. Uses the Bedrock Runtime **Converse API**, which gives one
request/response shape across model families (Anthropic, Meta, Amazon,
Mistral, Cohere) on Bedrock. Two current limitations: `tools=` isn't
translated to Converse's tool-call format yet, and since boto3 has no
official async client, `run_async()`/streaming run the synchronous call in
a worker thread rather than being natively non-blocking.

## Token Budget

Cap prompt and per-session token usage to avoid runaway costs:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    max_prompt_tokens=2000,
    max_session_tokens=10000,
)
```

`max_prompt_tokens` rejects a single prompt that exceeds the limit.
`max_session_tokens` tracks cumulative token use across runs on the same agent instance.
Both raise `TokenBudgetExceededError` when exceeded.

## Plugins

Load guardrails and verifiers from installed packages via entry points:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    plugins="auto",  # load every llmrivotril.guardrails / llmrivotril.verifiers entry point
)
```

> **Security note:** `plugins="auto"` executes `entry.load()` for every
> matching entry point registered by **any** package installed in the current
> environment, with no sandboxing or confirmation prompt (the same trust
> model as pytest plugins or Flask extensions). Only use `"auto"` when you
> control what's installed in that environment. Prefer passing explicit
> plugin names or instances (below) in any environment where third-party
> packages might be installed.

Or pass specific names and instances:

```python
from llmrivotril import Guardrail

agent = RivotrilAgent(
    plugins=["safe-input", Guardrail(name="short", max_tokens=100)],
)
```

Package authors can register plugins in `pyproject.toml`:

```toml
[project.entry-points."llmrivotril.guardrails"]
safe-input = "my_package.guardrails:make_guardrail"
```

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

## PII Redaction

Redact sensitive information from inputs and outputs before they reach the LLM
or logs:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    api_key="sk-...",
    redact_pii=True,
)

agent.run("My email is alice@example.com")
```

Redaction runs before a response is written to `cache=` (`InMemoryCache`,
`DiskCache`, or `RedisCache`), so raw PII is never persisted at rest when
`redact_pii=True`.

You can also use `PIIRedactor` directly to scan or sanitize text:

```python
from llmrivotril import PIIRedactor

redactor = PIIRedactor()
text = redactor.redact("CPF: 123.456.789-09")
```

## Configuration Files

In addition to environment variables, `RivotrilAgent` reads configuration files.
Create a `llmrivotril.toml` in your working directory:

```toml
model = "gpt-4o-mini"
api_key = "sk-..."
request_timeout = 30.0
retry_max_attempts = 5
```

Or use the `[tool.llmrivotril]` section of your `pyproject.toml`:

```toml
[tool.llmrivotril]
model = "gpt-4o-mini"
provider = "anthropic"
```

Precedence: file defaults < environment variables < constructor arguments.

## Interactive Demo

Run a complete walkthrough with mock LLM responses (no API key, no cost):

```bash
python examples/demo.py --mock --dashboard
```

Then open http://127.0.0.1:8767 to watch the dashboard update live.

## Comparison: With vs. Without `llmrivotril`

See the same scenarios side-by-side:

```bash
python examples/comparison.py --mock
```

The comparison highlights how plain LLM calls return harmful or hallucinated
answers that are only caught manually afterwards, while `llmrivotril` blocks
them at runtime and records structured telemetry.

## Environment Variables

`RivotrilAgent` can be configured entirely through environment variables. Explicit constructor arguments always take precedence.

| Variable | Type | Description |
|----------|------|-------------|
| `OPENAI_API_KEY` | string | API key for OpenAI-compatible endpoints (inferred by the OpenAI client when not passed). |
| `RIVOTRIL_MODEL` | string | Default model name (e.g. `gpt-4o-mini`). |
| `RIVOTRIL_PROVIDER` | string | Provider adapter: `openai`, `anthropic`, `cohere`, or `gemini`. |
| `RIVOTRIL_API_KEY` | string | API key passed directly to the OpenAI client. |
| `RIVOTRIL_BASE_URL` | string | OpenAI-compatible base URL (e.g. `http://localhost:11434/v1`). |
| `RIVOTRIL_SYSTEM_PROMPT` | string | System prompt injected on every run. |
| `RIVOTRIL_REQUEST_TIMEOUT` | float | Timeout in seconds for LLM calls. |
| `RIVOTRIL_RATE_LIMIT_MAX_CALLS` | float | Rate-limit bucket capacity. |
| `RIVOTRIL_RATE_LIMIT_PER_SECONDS` | float | Rate-limit refill period. |
| `RIVOTRIL_RETRY_MAX_ATTEMPTS` | int | Maximum retry attempts. |
| `RIVOTRIL_RETRY_MIN_WAIT` | float | Minimum retry backoff in seconds. |
| `RIVOTRIL_RETRY_MAX_WAIT` | float | Maximum retry backoff in seconds. |
| `RIVOTRIL_CIRCUIT_FAILURE_THRESHOLD` | int | Failures before the circuit opens. |
| `RIVOTRIL_CIRCUIT_RECOVERY_TIMEOUT` | float | Seconds before the circuit tries again. |
| `RIVOTRIL_MAX_PROMPT_TOKENS` | int | Reject prompts above this token count. |
| `RIVOTRIL_MAX_SESSION_TOKENS` | int | Reject runs that would exceed this cumulative budget. |
| `RIVOTRIL_METRICS_PATH` | string | Persist metrics to this JSON file automatically. |
| `RIVOTRIL_MEMORY_PATH` | string | Persist conversation memory to this JSON file automatically. |
| `RIVOTRIL_CACHE_PATH` | string | Enable disk-based response caching at this path. |
| `RIVOTRIL_TRACK_COSTS` | bool | Enable estimated cost tracking in metrics. |
| `RIVOTRIL_SCHEMA_REPAIR_ATTEMPTS` | int | Retry structured-output validation failures this many times. |
| `RIVOTRIL_REDACT_PII` | bool | Redact detected PII from inputs and outputs. |
| `RIVOTRIL_DASHBOARD_TOKEN` | string | When set, dashboard routes require `Authorization: Bearer <token>`. |
| `RIVOTRIL_DASHBOARD_TOKENS` | string | Multiple named tokens as `"label1:token1,label2:token2"`; each valid token's label is logged on access. |
| `RIVOTRIL_DASHBOARD_RATE_LIMIT_MAX_CALLS` | float | Dashboard per-IP rate-limit bucket capacity (default 60). |
| `RIVOTRIL_DASHBOARD_RATE_LIMIT_PER_SECONDS` | float | Dashboard per-IP rate-limit refill period in seconds (default 60). |

Example:

```bash
export RIVOTRIL_MODEL="gpt-4o-mini"
export RIVOTRIL_RATE_LIMIT_MAX_CALLS="10"
export RIVOTRIL_RETRY_MAX_ATTEMPTS="5"
```

## Project Structure

```
llmrivotril/
├── src/llmrivotril/
│   ├── agent.py              # RivotrilAgent orchestrator (sync + async)
│   ├── config.py             # Environment-variable and file configuration loader
│   ├── guardrails.py         # Input/output guardrails
│   ├── memory.py             # Conversation memory store
│   ├── verifier.py           # Hallucination / grounding checks
│   ├── providers.py          # OpenAI / Anthropic / Cohere / Gemini adapters
│   ├── rag/                  # RAG loaders, chunkers, retrievers and pipeline
│   ├── resilience.py         # Rate limiter, retry, and circuit breaker
│   ├── semantic.py           # Optional embedding-based guardrails/verifiers
│   ├── metrics.py            # Telemetry collector
│   ├── server.py             # FastAPI dashboard
│   ├── cli.py                # Click CLI
│   ├── exceptions.py         # Custom exceptions
│   ├── templates/
│   │   └── dashboard.html    # Dashboard UI template
│   └── static/
│       └── tailwind.min.js   # Bundled Tailwind CSS for offline dashboard
├── tests/
├── examples/
└── docs/
```

## Development

Run lint, type checks, and tests:

```bash
pip install -e ".[dev,semantic]"
ruff check src tests examples scripts
ruff format --check src tests examples scripts
mypy src
pytest -v
```

Slow integration tests (e.g. loading `sentence-transformers` models) are skipped by
default. Run them with:

```bash
pytest -v -m slow
```

## CI/CD

![CI](https://github.com/ailake-io/llmrivotril/workflows/CI/badge.svg)

The GitHub Actions workflow runs linting, type checking, tests, and package builds on Python 3.10–3.13.

## License

MIT License — see [LICENSE](LICENSE).
