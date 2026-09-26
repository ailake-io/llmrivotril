# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Token budget controls on `RivotrilAgent`:
  - `max_prompt_tokens` and `max_session_tokens` constructor arguments.
  - `RIVOTRIL_MAX_PROMPT_TOKENS` and `RIVOTRIL_MAX_SESSION_TOKENS` environment variables.
  - `TokenBudgetExceededError` raised when a run would exceed the configured budget.
- Plugin system via Python entry points:
  - Groups `llmrivotril.guardrails` and `llmrivotril.verifiers`.
  - `plugins="auto"` or `plugins=[...]` on `RivotrilAgent`.
  - `discover_plugins()` and `load_plugins()` helpers exported from the top-level package.
- Automatic metrics persistence:
  - `RIVOTRIL_METRICS_PATH` environment variable and `metrics_path=` argument on `RivotrilAgent`.
  - `MetricsCollector.load_metrics()` alias for `load_from_json()`.
- Function calling / tools support:
  - `tools=` argument on `agent.run()` and `agent.run_async()`.
  - `ToolRegistry` and `ToolCall` exported from the top-level package.
  - Accepts OpenAI-style tool dicts or plain Python callables (auto-introspected into schemas).
  - Built-in tool-call loop with follow-up completion.
- Offline dashboard: Tailwind CSS is now bundled and served locally instead of loaded from CDN.
- File-based configuration via `llmrivotril.toml`, `llmrivotril.yaml`, or `[tool.llmrivotril]` in `pyproject.toml`.
  - Precedence: file defaults < environment variables < constructor arguments.
- `RIVOTRIL_PROVIDER` environment variable and `provider=` argument on `RivotrilAgent`.
- Multi-provider support with adapters for OpenAI, Anthropic, Cohere, and Gemini in `llmrivotril.providers`.
- `ModelBasedFaithfulnessVerifier` for LLM-as-a-judge grounding checks.

## [0.5.0] - 2026-09-25

### Added

- Local RAG module (`llmrivotril.rag`) with:
  - `Document` model
  - `TextLoader` and `MarkdownLoader`
  - `SimpleChunker` with configurable size and overlap
  - `InMemoryKeywordRetriever` (no external dependencies)
  - `InMemoryEmbeddingRetriever` (optional `sentence-transformers`)
  - `RAGPipeline` to ingest sources and feed context into `RivotrilAgent`
- Re-exported RAG primitives from the top-level `llmrivotril` package.

## [0.4.0] - 2026-09-25

### Added

- `llmrivotril init [PATH]` CLI command to scaffold a new project with:
  - `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`
  - package directory with `agent.py` and `guardrails.py`
  - starter `tests/test_agent.py`

## [0.3.0] - 2026-09-25

### Added

- Environment-variable configuration via `llmrivotril.config.load_env_config()`.
  - `RIVOTRIL_MODEL`, `RIVOTRIL_API_KEY`, `RIVOTRIL_BASE_URL`, `RIVOTRIL_SYSTEM_PROMPT`
  - `RIVOTRIL_REQUEST_TIMEOUT`
  - `RIVOTRIL_RATE_LIMIT_MAX_CALLS`, `RIVOTRIL_RATE_LIMIT_PER_SECONDS`
  - `RIVOTRIL_RETRY_MAX_ATTEMPTS`, `RIVOTRIL_RETRY_MIN_WAIT`, `RIVOTRIL_RETRY_MAX_WAIT`
  - `RIVOTRIL_CIRCUIT_FAILURE_THRESHOLD`, `RIVOTRIL_CIRCUIT_RECOVERY_TIMEOUT`
  - Explicit constructor arguments always override environment values.
- `/api/health` endpoint on the dashboard returning status, version, and UTC timestamp.
- Async-safe `AsyncRateLimiter` used by `RivotrilAgent.run_async()`.
- `system_prompt` support injected as a `system` message.
- `context_sources` accepts `str | list[str] | None` and joins multiple sources.
- Structured logging through the `llmrivotril` logger.
- CLI commands: `benchmark`, `doctor`, `version`.
- `request_timeout` forwarded to OpenAI/instructor calls.
- Per-agent `metrics` injection and `MetricsCollector.reset()`.
- `disallowed_patterns` regex guardrails on input and output.

### Changed

- `tiktoken` encoding now falls back to `cl100k_base` for unknown model names, improving compatibility with local/Ollama models.
- `Guardrail` output validation also checks `disallowed_keywords`.

## [0.2.0] - 2026-09-24

### Added

- First packaged release of LLM-Rivotril.

## [0.1.0] - 2026-09-20

### Added

- Proof-of-concept agent with basic guardrails.
