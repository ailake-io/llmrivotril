# Configuration, Plugins & Scaffolding

[← Back to README](../README.md)

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

## Quick Project Scaffolding

Create a new ready-to-use project with a single command:

```bash
llmrivotril init my-project
```

It generates `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`, a package directory with `agent.py` / `guardrails.py`, and a starter `tests/test_agent.py`.
