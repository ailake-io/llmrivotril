# LLM-Rivotril

*[Português](https://github.com/ailake-io/llmrivotril/blob/main/README.pt-BR.md)*

A lightweight Python framework to reduce LLM hallucinations, enforce guardrails, manage stateful memory, and monitor performance through a local web dashboard.

## Features

- **Guardrails** — Block disallowed keywords, enforce allowed topics, limit output size, and validate JSON schemas on outputs.
- **Semantic Guardrails** *(optional)* — Match prompts against allowed topics using dense embeddings instead of exact keywords.
- **Moderation Guardrail** — Flag adversarial/unsafe content via OpenAI's moderation endpoint.
- **Stateful Memory** — Sliding-window conversation store to prevent context drift, with optional automatic disk persistence.
- **Anti-Hallucination Verifier** — Pluggable grounding checks, including keyword overlap, citation markers, and embedding-based faithfulness.
- **Resilience** — Built-in rate limiting, retry with backoff, and circuit breaker for LLM calls.
- **Token-Saving Controls** *(optional)* — Semantic (similarity-based) response cache, token-budget-aware memory trimming, and LLM-summarized history compaction.
- **Telemetry & Dashboard** — Built-in FastAPI dashboard with live request logs, token usage, latency, and success rate; optional per-token auth.
- **Benchmark / Red-Team Evaluator** — Labeled suite to measure guardrail and verifier accuracy without API costs.
- **CLI** — Launch the dashboard with a single command.
- **Type-Safe Responses** — Optional Pydantic response models via `instructor`, including streamed structured output.
- **Async API** — `run_async()` for non-blocking execution.
- **Multi-Provider** — OpenAI-compatible servers (Ollama, vLLM, ...), plus native Anthropic, Cohere, Gemini, Azure OpenAI, and AWS Bedrock adapters.
- **Vector-Store Retrievers** *(optional)* — pgvector, Qdrant, Weaviate, and Pinecone adapters for RAG beyond in-memory scale.
- **Framework Integrations** *(optional)* — Drop-in adapters for CrewAI, AG2/AutoGen, LangChain/LangGraph, and Google ADK, so guardrails/PII/RAG/observability apply inside those frameworks too.

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

This installs test, lint, type-check, and packaging tools (`pytest`, `ruff`, `mypy`, `build`, `twine`) plus every optional runtime dependency (all providers, RAG loaders, vector stores, Redis, framework integrations).

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

## Documentation

- [Providers](https://github.com/ailake-io/llmrivotril/blob/main/docs/providers.md) — OpenAI-compatible servers, Anthropic, Cohere, Gemini, Azure OpenAI, AWS Bedrock.
- [Guardrails & Safety](https://github.com/ailake-io/llmrivotril/blob/main/docs/guardrails-and-safety.md) — Semantic guardrails, moderation, PII redaction, token budget.
- [RAG](https://github.com/ailake-io/llmrivotril/blob/main/docs/rag.md) — Local pipeline, vector-store retrievers (pgvector/Qdrant/Weaviate/Pinecone), grounding verification.
- [Framework Integrations](https://github.com/ailake-io/llmrivotril/blob/main/docs/integrations.md) — CrewAI, AG2/AutoGen, LangChain/LangGraph, Google ADK adapters, and multi-agent setup notes.
- [Async, Streaming & Function Calling](https://github.com/ailake-io/llmrivotril/blob/main/docs/streaming-and-tools.md)
- [Reliability](https://github.com/ailake-io/llmrivotril/blob/main/docs/reliability.md) — Resilience, response caching (including semantic cache), memory token budget & summarization, schema-repair fallback.
- [Observability](https://github.com/ailake-io/llmrivotril/blob/main/docs/observability.md) — Metrics persistence, local dashboard, benchmark, cost tracking.
- [Configuration](https://github.com/ailake-io/llmrivotril/blob/main/docs/configuration.md) — Config files, environment variables, plugins, project scaffolding.
- [Releasing](https://github.com/ailake-io/llmrivotril/blob/main/docs/releasing.md) — Build validation, TestPyPI, and the PyPI release workflow.

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

## Project Structure

```
llmrivotril/
├── src/llmrivotril/
│   ├── agent.py              # RivotrilAgent orchestrator (sync + async)
│   ├── config.py             # Environment-variable and file configuration loader
│   ├── guardrails.py         # Input/output guardrails
│   ├── moderation.py         # OpenAI moderation-endpoint guardrail
│   ├── memory.py             # Conversation memory store
│   ├── verifier.py           # Hallucination / grounding checks
│   ├── providers.py          # OpenAI / Azure / Anthropic / Cohere / Gemini / Bedrock adapters
│   ├── rag/                  # RAG loaders, chunkers, retrievers, vector stores, and pipeline
│   ├── integrations/         # CrewAI / AG2 / LangChain / Google ADK adapters
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
pytest -v --run-slow
```

## CI/CD

![CI](https://github.com/ailake-io/llmrivotril/workflows/CI/badge.svg)

The GitHub Actions workflow runs linting, type checking, tests, and package builds on Python 3.10–3.13.

## License

MIT License — see [LICENSE](https://github.com/ailake-io/llmrivotril/blob/main/LICENSE).
