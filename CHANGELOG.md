# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `MetricsCollector` now tracks whether each request was a cache hit:
  `cache_hits`/`cache_hit_rate` in `get_summary()`, and a `cache_hit` boolean
  on each per-request log entry. Previously the only way to tell a cache hit
  from a miss was inferring it from latency -- the `tokens` figure is a
  local tiktoken estimate over prompt+response text, identical either way.
  The dashboard shows a "Cache Hit Rate" tile and a "Cache Hit" badge per
  row; `/api/metrics/prometheus` exposes
  `llmrivotril_cache_hits_total`/`llmrivotril_cache_hit_rate`. See
  `docs/observability.md`.

## [0.1.2] - 2026-09-30

### Changed

- Rewrote the AG2 integration adapter to target **current AG2 1.x**
  (`llmrivotril[autogen]`, `pip install "llmrivotril[autogen]"`) instead of
  the pre-1.0 `AssistantAgent`/`register_model_client` API, which no longer
  exists in any currently-installable package (`ag2>=1.0` replaced its
  entire API; `pyautogen` is now a proxy for Microsoft's separate
  `autogen-agentchat`/`autogen-core` rewrite). `RivotrilModelConfig`/
  `RivotrilLLMClient` implement AG2 1.x's `ModelConfig`/`LLMClient`
  protocols via its new `Agent(config=...)` extension point. Unlike the
  pre-1.0 adapter, this needs `ag2` actually importable (AG2 1.x's
  `ModelMessage`/`ModelResponse` are real event classes this module
  constructs, not plain dicts). Verified end-to-end against a real provider
  with `ag2==1.1.1`. Tool-calling and structured output are not implemented
  for this version; see `docs/integrations.md`.

### Fixed

- CI now runs on pull requests targeting `develop`, not just `main` --
  PRs into `develop` previously got zero CI coverage.

## [0.1.1] - 2026-09-30

### Fixed

- Tool-calling (`run`/`run_async`/`run_stream`/`run_stream_async` with
  `tools=`) sent a follow-up request missing the assistant's own
  `tool_calls` message before the `role: "tool"` result messages -- real
  OpenAI-compatible APIs reject this ("messages with role 'tool' must be a
  response to a preceding message with 'tool_calls'"), but every existing
  test mocked `complete()` directly and never validated message ordering, so
  it went uncaught until a real end-to-end smoke test against a live
  provider (OpenRouter) right after the 0.1.0 PyPI release.
- `MetricsCollector`/`global_metrics` were importable from
  `llmrivotril.metrics` but not from the top-level `llmrivotril` package,
  even though `RivotrilAgent(metrics=...)` documents them as the expected
  argument type. Found during the same smoke test.

## [0.1.0] - 2026-09-29

### Added

- First release candidate for PyPI distribution.
- Framework-integration adapters for CrewAI (`BaseLLM`), AG2/pyautogen
  (`ModelClient` protocol, no extra dependency), LangChain/LangGraph
  (`BaseChatModel`), and Google ADK (`BaseLlm`) under
  `llmrivotril.integrations`. Each is optional (`llmrivotril[crewai]`,
  `llmrivotril[langchain]`, `llmrivotril[adk]`, or `llmrivotril[integrations]`
  for all three) and documented in `docs/integrations.md`, including
  multi-agent/crew setup notes.
- Three independent, opt-in token-saving controls, all off by default and
  documented in `docs/reliability.md`:
  - `SemanticCache` (`llmrivotril.semantic_cache`) -- wraps any `BaseCache`
    backend with an embedding-similarity lookup, so paraphrased repeat
    prompts hit the cache too, not just byte-identical ones. Requires
    `llmrivotril[semantic]`.
  - `MemoryStore(max_tokens=..., count_tokens=...)` -- trims the oldest
    memory turns by actual token count, layered on top of the existing
    turn-count `retention_window` cap. `RivotrilAgent(memory_max_tokens=...)`
    wires this in automatically for its default memory store.
  - `MemoryStore(summarize=..., summarize_trigger_turns=...)` -- compacts
    turns that would otherwise be dropped into a short summary instead.
    `RivotrilAgent(memory_summarize=True, memory_summarize_trigger_turns=...)`
    wires in a summarizer backed by the agent's own provider.
  - `BaseCache.get`/`set` gained an optional `prompt=` parameter (ignored by
    the existing exact-match backends) so a fuzzy-matching cache can see the
    raw prompt text; this is a backward-compatible addition, not a breaking
    change to the interface.
- Brazilian Portuguese translations (`*.pt-BR.md`) of the README and every
  public doc, each cross-linked with its English original.
- RAG context injection, deterministic vector-store identifiers, protected
  memory persistence, cache namespacing, and expanded telemetry.
- Explicit grounding failure policies and provider-specific retry handling.
- Gemini adapter migration to the `google-genai` SDK.
- `RAGPipeline.retrieve()`/`aretrieve()` for accessing ranked documents with
  IDs and metadata before formatting context for an LLM.
- Public top-level exports for `BaseLoader`, `BaseChunker` and `BaseRetriever`
  so custom RAG components do not depend on internal module paths.
- Async RAG methods now execute synchronous loading, chunking and retrieval in
  a short-lived worker without blocking the event loop.
- RAG queries support exact metadata filters, source labels and maximum context
  size limits.
- PostgreSQL, Qdrant and Pinecone use native metadata filters when available;
  Weaviate now supports native filters for explicitly configured flattened
  metadata fields and keeps a client-side fallback for other fields.

### Changed

- CI and release workflows now use `.github/constraints-ci.txt` to keep lint,
  type-check, test, and packaging tool versions reproducible without pinning
  runtime/provider dependencies for downstream applications.
- LangChain, CrewAI and Google ADK integrations now support streaming through
  their native adapter protocols; LangChain also supports `bind_tools()`.
- Prompt content can contain text, image, audio and document parts. OpenAI,
  Azure, Anthropic, Gemini and Bedrock preserve supported rich content while
  guardrails, PII redaction, token accounting and memory use a safe text view.
- Bedrock Converse streaming now accumulates `toolUse` deltas and executes the
  generic agent tool loop.

### Fixed

- RAG retrievers now reject negative `top_k` values instead of relying on
  Python slicing semantics.
- Three `WeaviateRetriever` tests only passed when `weaviate-client` happened
  to be installed locally, because `_collection_properties()`/
  `_native_metadata_filter()` do a real (unmocked) `from weaviate.classes...
  import ...` that a mocked `client=` object can't stand in for. This wasn't
  caught by the "326 tests passed" gate recorded earlier in
  `docs/technical-review.md`, because that run had `weaviate-client`
  installed -- CI's `lint-and-test` job (`pip install -e ".[ci]"`, no
  optional extras) would have failed on these 3. Fixed by mocking
  `weaviate.classes.config`/`.query` the same way the other optional
  vector-store dependencies already are; verified clean with zero optional
  packages installed, matching real CI exactly.

### Security

- Dashboard log rendering no longer interpolates untrusted content as HTML.
- SQL table identifiers are validated before being used by the pgvector
  adapter.

## [0.0.9] - 2026-09-28

### Added

- `AnthropicProvider`/`CohereProvider`/`GeminiProvider` now implement
  `stream()`/`astream()` (previously unimplemented, raised
  `NotImplementedError`), each using that provider's native streaming API
  and yielding plain text chunks. Not verified against a live account --
  same caveat as the Azure/Bedrock adapters.
- `run_stream()`/`run_stream_async()` now accept `response_model=`. Returns
  a `StreamedStructuredResult`/`AsyncStreamedStructuredResult` instead of a
  plain iterator: iterate it for the raw JSON text as it streams, and read
  `.result` after the loop ends for the validated `response_model`
  instance (partial JSON isn't a valid model, so there's nothing to
  validate until the stream is done). Can't be combined with `tools=`
  (raises `ValueError`).

## [0.0.8] - 2026-09-28

### Added

- `ModerationGuardrail`: flags input/output via OpenAI's moderation
  endpoint, for adversarial/unsafe content the local keyword/regex/topic
  guardrails aren't meant to catch. Fails closed by default (`fail_open=`
  to opt out).
- `RIVOTRIL_DASHBOARD_TOKENS`: multiple named bearer tokens for the
  dashboard (`"label1:token1,label2:token2"`) instead of one shared secret;
  each valid token's label is logged on access. `RIVOTRIL_DASHBOARD_TOKEN`
  (singular) still works as a single unlabeled token.
- `AzureOpenAIProvider` and `BedrockProvider`: new provider adapters.
  Azure OpenAI uses the `openai` SDK's dedicated `AzureOpenAI`/
  `AsyncAzureOpenAI` clients (no extra install). Bedrock uses the Bedrock
  Runtime Converse API for one request shape across model families
  (requires `llmrivotril[bedrock]`); `tools=` isn't translated to Converse's
  tool-call format yet, and `acomplete`/`astream` run the synchronous boto3
  call in a worker thread since boto3 has no official async client. Neither
  adapter has been verified against a live Azure/AWS account -- built from
  documented API shapes and covered by mocked unit tests only.
- `PgVectorRetriever`, `QdrantRetriever`, `WeaviateRetriever`,
  `PineconeRetriever`: external vector-store retrievers for RAG beyond
  in-memory scale, behind `llmrivotril[pgvector|qdrant|weaviate|pinecone]`
  (or `llmrivotril[vector-stores]` for all four). Also not verified against
  a live service -- same caveat as the two new providers.

### Fixed

- The non-streaming multi-round tool-calling loop
  (`_handle_tool_calls`/`_handle_tool_calls_async`) passed `tools=None` on
  every follow-up call after the first, so a conversation needing a second
  tool call in the same turn could never request one -- the model was never
  offered any tool schemas past round one.

## [0.0.7] - 2026-09-28

### Added

- `InMemoryEmbeddingRetriever(cache_path=...)`: content-hash-keyed embedding
  cache persisted to JSON, so re-ingesting unchanged document content across
  process restarts skips recomputing its embedding.
- `run_stream()`/`run_stream_async()` accept `tools=`; a turn where the model
  answers directly still streams token by token, and a turn where it calls a
  tool falls back to a single blocking round-trip for that turn.
- `MemoryStore(auto_save_path=...)` / `RIVOTRIL_MEMORY_PATH`: persist
  conversation history to a JSON file automatically, mirroring
  `RIVOTRIL_METRICS_PATH`/`RIVOTRIL_CACHE_PATH`.
- `RAGPipeline.aingest()`/`RAGPipeline.aquery()`: async wrappers (via a
  worker thread) so ingestion/retrieval don't block the event loop inside
  `run_async()`.
- `py.typed` marker (PEP 561) so downstream type checkers pick up this
  package's types.
- Dashboard endpoints are now rate-limited per client IP
  (`RIVOTRIL_DASHBOARD_RATE_LIMIT_MAX_CALLS`/`_PER_SECONDS`).

### Fixed

- `InMemoryEmbeddingRetriever.retrieve()` scored documents with a pure-Python
  cosine-similarity loop; switched to a cached numpy matrix (~5x faster at a
  few thousand documents), with the pure-Python path kept as a fallback when
  numpy isn't installed.
- `InMemoryEmbeddingRetriever._load_model()` had the same unlocked
  lazy-model-load race already fixed in `semantic.py`; added the same
  double-checked lock.
- Deleted `InMemoryKeywordRetriever`'s duplicated `[a-z0-9]+` tokenizer/
  stopword list (same accent-splitting bug as `verifier.py`); it now reuses
  the shared, fixed `_tokenize`.
- Test isolation: `tests/test_config.py` set `os.environ[...]` directly and
  only had a before-test cleanup fixture, so a var left set by its last test
  leaked into whatever test module pytest ran next. Also fixed a subtler
  variant of the same bug in the fix itself: calling `monkeypatch.delenv()`
  from *inside* another fixture's post-`yield` teardown gets undone by
  monkeypatch's own finalizer (which runs afterwards, in reverse dependency
  order) -- switched to plain `os.environ.pop()` for this cleanup.

### Documentation

- Documented that `plugins="auto"` executes code from any installed
  package's entry points, with no sandboxing.
- Documented that Azure OpenAI and AWS Bedrock are not supported via
  `base_url` (they need a different client/auth/request shape, not just a
  different endpoint).
- Documented the local dashboard's trust model (single shared token, no
  per-user accounts -- not for multi-tenant or public exposure).

## [0.0.6] - 2026-09-27

### Added

- Response caching:
  - `cache=` argument on `RivotrilAgent` (`InMemoryCache`, `DiskCache`, `RedisCache`, or custom `BaseCache`).
  - `RIVOTRIL_CACHE_PATH` environment variable to activate disk-based caching.
  - Deterministic cache keys based on model, messages, tools, and response model.
  - `RedisCache` for multi-process/distributed deployments (optional `llmrivotril[redis]` dependency).
- Schema-repair fallback for structured outputs:
  - `schema_repair_attempts=` argument on `RivotrilAgent`.
  - `RIVOTRIL_SCHEMA_REPAIR_ATTEMPTS` environment variable.
  - Re-prompts the model with validation errors when Pydantic schema validation fails.
- Cost tracking:
  - `track_costs=` argument on `RivotrilAgent`.
  - `RIVOTRIL_TRACK_COSTS` environment variable.
  - Built-in pricing table for OpenAI, Anthropic, Cohere, and Gemini models.
  - `estimate_cost()`, `register_pricing()`, and `list_supported_models()` helpers.
- PII detection and redaction:
  - `redact_pii=` argument on `RivotrilAgent`.
  - `RIVOTRIL_REDACT_PII` environment variable.
  - `PIIRedactor` with regex scanners for email, CPF, CNPJ, phone, and credit card.
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

### Fixed

- PII redaction now applies to structured (`response_model=`) results and runs
  before a response is written to `cache=`; previously `redact_pii=True` left
  structured outputs unredacted and could persist raw PII at rest in the cache.
- `DiskCache` no longer uses `pickle` (arbitrary code execution risk if the
  cache file is ever tampered with); it now stores JSON in a local SQLite
  file, with indexed per-key reads/writes instead of rewriting the whole
  store on every `get`/`set`.
- `PIIRedactor.redact()` no longer corrupts text when two pattern matches
  overlap; overlapping spans are merged before replacement.
- The local dashboard's token check now uses a constant-time comparison
  (`secrets.compare_digest`) instead of `!=`, closing a timing side-channel.
- `_load_dotenv()` no longer silently swallows a missing `python-dotenv`
  install; it's a core dependency and is now imported unconditionally.
- Fixed a duplicated `_ENV_INTS` definition in `config.py` that silently
  discarded the first set of keys.
- Corrected the configuration-precedence docstring in `config.py` to match
  the actual lookup order.
- `CircuitBreaker` no longer lets multiple threads dispatch a HALF_OPEN probe
  concurrently; checking and transitioning state is now a single atomic
  operation instead of two separately-locked steps.
- `KeywordOverlapVerifier`'s tokenizer used `[a-z0-9]+`, which split accented
  words (e.g. "informação" -> "informa" + "o"), corrupting grounding checks
  for Portuguese and other accented-language text. Switched to a
  Unicode-aware pattern and added Portuguese stopwords alongside the
  existing English list.

## [0.0.5] - 2026-09-25

### Added

- Local RAG module (`llmrivotril.rag`) with:
  - `Document` model
  - `TextLoader` and `MarkdownLoader`
  - `SimpleChunker` with configurable size and overlap
  - `InMemoryKeywordRetriever` (no external dependencies)
  - `InMemoryEmbeddingRetriever` (optional `sentence-transformers`)
  - `RAGPipeline` to ingest sources and feed context into `RivotrilAgent`
- Re-exported RAG primitives from the top-level `llmrivotril` package.

## [0.0.4] - 2026-09-25

### Added

- `llmrivotril init [PATH]` CLI command to scaffold a new project with:
  - `pyproject.toml`, `.gitignore`, `.env.example`, `README.md`
  - package directory with `agent.py` and `guardrails.py`
  - starter `tests/test_agent.py`

## [0.0.3] - 2026-09-25

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

## [0.0.2] - 2026-09-24

### Added

- First packaged release of LLM-Rivotril.

## [0.0.1] - 2026-09-20

### Added

- Proof-of-concept agent with basic guardrails.
