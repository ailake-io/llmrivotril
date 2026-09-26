# LLM-Rivotril

A lightweight Python framework to reduce LLM hallucinations, enforce guardrails, manage stateful memory, and monitor performance through a local web dashboard.

## Features

- **Guardrails** — Block disallowed keywords, enforce allowed topics, limit output size, and validate JSON schemas on outputs.
- **Semantic Guardrails** *(optional)* — Match prompts against allowed topics using dense embeddings instead of exact keywords.
- **Stateful Memory** — Sliding-window conversation store to prevent context drift.
- **Anti-Hallucination Verifier** — Pluggable grounding checks, including keyword overlap, citation markers, and embedding-based faithfulness.
- **Resilience** — Built-in rate limiting, retry with backoff, and circuit breaker for LLM calls.
- **Telemetry & Dashboard** — Built-in FastAPI dashboard with live request logs, token usage, latency, and success rate; optional Bearer-token auth.
- **Benchmark / Red-Team Evaluator** — Labeled suite to measure guardrail and verifier accuracy without API costs.
- **CLI** — Launch the dashboard with a single command.
- **Type-Safe Responses** — Optional Pydantic response models via `instructor`.
- **Async API** — `run_async()` for non-blocking execution.
- **OpenAI-Compatible APIs** — Use `base_url` to point to Ollama, vLLM, or other compatible servers.

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

The pipeline includes `TextLoader`, `MarkdownLoader`, `SimpleChunker` and `InMemoryKeywordRetriever`. For dense retrieval, install `llmrivotril[semantic]` and use `InMemoryEmbeddingRetriever`.

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

Supported providers: `openai` (default), `anthropic`, `cohere`, `gemini`.

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
| `RIVOTRIL_DASHBOARD_TOKEN` | string | When set, dashboard routes require `Authorization: Bearer <token>`. |

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
