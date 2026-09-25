# LLM-Rivotril Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Quick Project Scaffolding](#quick-project-scaffolding)
3. [Environment Configuration](#environment-configuration)
4. [Creating an Agent](#creating-an-agent)
5. [System Prompt](#system-prompt)
6. [Configuring Guardrails](#configuring-guardrails)
7. [Semantic Guardrails](#semantic-guardrails)
8. [Using Memory](#using-memory)
9. [Structured Output](#structured-output)
10. [Grounding Verification](#grounding-verification)
11. [RAG](#rag)
12. [Resilience](#resilience)
13. [Async Execution](#async-execution)
14. [OpenAI-Compatible Servers](#openai-compatible-servers)
15. [Running the Dashboard](#running-the-dashboard)
16. [Benchmark](#benchmark)
17. [Telemetry](#telemetry)

## Installation

```bash
pip install llmrivotril
```

## Quick Project Scaffolding

```bash
llmrivotril init my-project
```

Creates a complete starter project:

- `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`
- `<my_project>/agent.py` and `<my_project>/guardrails.py`
- `tests/test_agent.py`

Then:

```bash
cd my-project
pip install -e .
pytest
python my_project/agent.py
```

## Environment Configuration

You can configure `RivotrilAgent` via environment variables instead of constructor arguments. Explicit arguments always win.

```bash
export RIVOTRIL_MODEL="gpt-4o-mini"
export RIVOTRIL_API_KEY="sk-..."
export RIVOTRIL_BASE_URL="http://localhost:11434/v1"
export RIVOTRIL_REQUEST_TIMEOUT="30"
export RIVOTRIL_RATE_LIMIT_MAX_CALLS="10"
export RIVOTRIL_RETRY_MAX_ATTEMPTS="5"
```

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent()  # reads every RIVOTRIL_* variable above
```

Supported variables: `RIVOTRIL_MODEL`, `RIVOTRIL_API_KEY`, `RIVOTRIL_BASE_URL`, `RIVOTRIL_SYSTEM_PROMPT`, `RIVOTRIL_REQUEST_TIMEOUT`, `RIVOTRIL_RATE_LIMIT_MAX_CALLS`, `RIVOTRIL_RATE_LIMIT_PER_SECONDS`, `RIVOTRIL_RETRY_MAX_ATTEMPTS`, `RIVOTRIL_RETRY_MIN_WAIT`, `RIVOTRIL_RETRY_MAX_WAIT`, `RIVOTRIL_CIRCUIT_FAILURE_THRESHOLD`, `RIVOTRIL_CIRCUIT_RECOVERY_TIMEOUT`.

## Creating an Agent

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    model="gpt-4o-mini",
    api_key="sk-..."
)

response = agent.run("What is retrieval-augmented generation?")
print(response)
```

## System Prompt

Set a persistent system prompt that is injected as the first message on every run:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    system_prompt="You are a concise, factual assistant.",
)
```

## Configuring Guardrails

```python
from pydantic import BaseModel
from llmrivotril import Guardrail

class Summary(BaseModel):
    title: str
    points: list[str]

guardrail = Guardrail(
    name="content-safety",
    allowed_topics=["machine learning", "software engineering"],
    disallowed_keywords=["hack", "exploit"],
    disallowed_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"],  # SSN-like numbers
    max_tokens=250,
    json_schema=Summary,
)
```

`allowed_topics` enforces that the prompt mentions at least one of the listed topics. `disallowed_keywords` and `disallowed_patterns` are checked on both input and output. `json_schema` validates that the model output parses and conforms to the Pydantic model.

## Using Memory

```python
from llmrivotril import MemoryStore

memory = MemoryStore(retention_window=10)
agent = RivotrilAgent(memory=memory)

agent.run("My name is Alice.")
agent.run("What is my name?")  # Uses conversation history
```

## Structured Output

```python
from pydantic import BaseModel

class Answer(BaseModel):
    concise: str
    details: str

result = agent.run("Explain overfitting.", response_model=Answer)
print(result.concise)
```

## Grounding Verification

Use a built-in verifier to reduce hallucinations when context sources are available:

```python
from llmrivotril.verifier import KeywordOverlapVerifier, CitationVerifier

agent = RivotrilAgent(
    verifier=KeywordOverlapVerifier(threshold=0.1),
)

context = "The speed of light is 299,792 km/s."
response = agent.run(f"Based on: {context}\n\nWhat is the speed of light?")
```

`CitationVerifier` requires quotes, brackets, or citation-style markers in the response.

The default `Verifier()` now uses `KeywordOverlapVerifier` automatically whenever
`context_sources` is provided, so ungrounded answers are rejected even without an
explicit verifier configuration.

### Multiple Context Sources

`context_sources` accepts a single string or a list of strings. Lists are joined with blank lines before grounding checks:

```python
agent.run(
    "Summarize the sources.",
    context_sources=[
        "France is in Europe.",
        "The capital of France is Paris.",
    ],
)
```

### Embedding-Based Faithfulness

For stronger semantic grounding checks, install `llmrivotril[semantic]` and use
`EmbeddingFaithfulnessVerifier`:

```python
from llmrivotril import EmbeddingFaithfulnessVerifier
from llmrivotril.verifier import Verifier

agent = RivotrilAgent(
    verifier=Verifier(
        check_fn=EmbeddingFaithfulnessVerifier(similarity_threshold=0.6).as_callable()
    ),
)
```

## Semantic Guardrails

`SemanticTopicGuardrail` matches prompts against allowed topics using dense
embeddings instead of exact keywords. This catches rephrased or related prompts
that keyword matching would miss.

```python
from llmrivotril import SemanticTopicGuardrail

guardrail = SemanticTopicGuardrail(
    name="semantic-topics",
    allowed_topics=["machine learning", "software engineering"],
    similarity_threshold=0.5,
)

agent = RivotrilAgent(guardrails=[guardrail])
```

## RAG

Build a local retrieval pipeline to feed external documents into the agent:

```python
from llmrivotril import RAGPipeline, RivotrilAgent

pipeline = RAGPipeline()
pipeline.ingest("docs/")  # loads .txt and .md files

agent = RivotrilAgent()
response = agent.run(
    "What is a guardrail?",
    context_sources=pipeline.query("What is a guardrail?"),
)
```

The default pipeline uses `TextLoader`, `SimpleChunker` and `InMemoryKeywordRetriever`. For dense retrieval, install `llmrivotril[semantic]` and pass an `InMemoryEmbeddingRetriever`:

```python
from llmrivotril.rag.retrievers import InMemoryEmbeddingRetriever

pipeline = RAGPipeline(retriever=InMemoryEmbeddingRetriever(model="all-MiniLM-L6-v2"))
```

## Resilience

Add rate limiting, retries with exponential backoff, and a circuit breaker to LLM
calls:

```python
agent = RivotrilAgent(
    rate_limit_max_calls=10,
    rate_limit_per_seconds=1,
    retry_max_attempts=3,
    retry_min_wait=1.0,
    retry_max_wait=10.0,
    circuit_failure_threshold=5,
    circuit_recovery_timeout=30.0,
)
```

## Async Execution

```python
import asyncio
from llmrivotril import RivotrilAgent

async def main():
    agent = RivotrilAgent(api_key="sk-...")
    response = await agent.run_async("Hello!")
    print(response)

asyncio.run(main())
```

## OpenAI-Compatible Servers

Use `base_url` to connect to Ollama, vLLM, or any OpenAI-compatible server:

```python
agent = RivotrilAgent(
    model="llama3",
    base_url="http://localhost:11434/v1",
    api_key="unused",
)
```

When the model name is not known to `tiktoken`, token counting automatically falls back to the `cl100k_base` encoder, so local or custom models still get accurate token estimates.

## Interactive Demo

Run the built-in demo to see guardrails, grounding checks, and dashboard telemetry
in action without spending API credits:

```bash
python examples/demo.py --mock --dashboard
```

Open http://127.0.0.1:8767 to watch metrics update live.

## Comparison: With vs. Without `llmrivotril`

Run the comparison script to see the same scenarios executed with a plain LLM
call and with `llmrivotril`:

```bash
python examples/comparison.py --mock
```

It demonstrates runtime blocking of policy violations and hallucinations, plus
structured telemetry, compared to manual post-hoc inspection.

## Running the Dashboard

```bash
llmrivotril dashboard --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 to view live metrics.

API endpoints:

- `/api/metrics` — JSON telemetry summary
- `/api/health` — Health check (`status`, `version`, `timestamp`)


### Dashboard Authentication

Set `RIVOTRIL_DASHBOARD_TOKEN` to require a Bearer token:

```bash
export RIVOTRIL_DASHBOARD_TOKEN="your-secret-token"
llmrivotril dashboard --port 8000
```

Clients must send:

```
Authorization: Bearer your-secret-token
```

## Benchmark

Run the built-in red-team benchmark with deterministic mock LLM responses (no API
cost):

```bash
python scripts/benchmark.py --mock
```

The benchmark reports accuracy, false positives, and false negatives for the
default guardrail and verifier configuration. Write a JSON report with:

```bash
python scripts/benchmark.py --mock --output report.json
```

## Telemetry

The global `MetricsCollector` records every request, including:

- Total requests
- Guardrail blocks
- Hallucination blocks
- Tokens consumed
- Average latency
- Success rate

Access it programmatically:

```python
from llmrivotril.metrics import global_metrics

print(global_metrics.get_summary())
```
