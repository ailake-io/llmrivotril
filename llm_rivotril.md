# LLM-Rivotril: Complete Python Package Specification, Implementation Guide & Local Dashboard

## 1. Project Directory Structure

To publish `llmrivotril` as a production-grade Python package with built-in CLI and Local Dashboard capabilities, use the following structure:

```
llmrivotril/
├── .github/
│   └── workflows/
│       └── ci.yml
├── pyproject.toml
├── README.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
├── .gitignore
├── src/
│   └── llmrivotril/
│       ├── __init__.py
│       ├── agent.py
│       ├── guardrails.py
│       ├── memory.py
│       ├── verifier.py
│       ├── metrics.py
│       ├── server.py
│       ├── cli.py
│       ├── exceptions.py
│       └── templates/
│           └── dashboard.html
├── tests/
│   ├── __init__.py
│   ├── test_guardrails.py
│   ├── test_memory.py
│   ├── test_verifier.py
│   ├── test_metrics.py
│   └── test_agent.py
├── examples/
│   └── basic_usage.py
└── docs/
    └── usage.md
```

## 2. Packaging Configuration (`pyproject.toml`)

```toml
[build-system]
requires = ["hatchling>=1.18.0"]
build-backend = "hatchling.build"

[project]
name = "llmrivotril"
version = "0.2.0"
description = "A lightweight framework to reduce LLM hallucinations, enforce guardrails, manage stateful memory, and monitor performance via local dashboard."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [
    { name = "Developer", email = "developer@example.com" }
]
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "pydantic>=2.0.0",
    "instructor>=1.0.0",
    "openai>=1.0.0",
    "tiktoken>=0.5.0",
    "fastapi>=0.100.0",
    "uvicorn>=0.22.0",
    "click>=8.0.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "httpx>=0.24.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
    "build>=1.0.0",
    "twine>=5.0.0"
]

[project.scripts]
llmrivotril = "llmrivotril.cli:cli"

[project.urls]
Homepage = "https://github.com/ailake-io/llmrivotril"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "W", "N", "UP", "B", "C4"]

[tool.mypy]
python_version = "3.10"
strict = true
warn_return_any = true
warn_unused_ignores = true
```

## 3. Core Implementation Code (`src/llmrivotril/`)

### 3.1 Exceptions (`exceptions.py`)
```python
class LLMRivotrilError(Exception):
    """Base exception for all llmrivotril errors."""


class GuardrailViolationError(LLMRivotrilError):
    """Raised when input or output violates configured guardrails."""


class HallucinationDetectedError(LLMRivotrilError):
    """Raised when verification layers catch ungrounded output."""
```

### 3.2 Metrics & Telemetry Collector (`metrics.py`)
```python
import time
from typing import Any
from threading import Lock


class MetricsCollector:
    """Thread-safe telemetry store.

    Tracks requests, tokens, guardrail blocks, and hallucinations.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.guardrail_blocks = 0
        self.hallucinations_detected = 0
        self.total_tokens_consumed = 0
        self.latencies: list[float] = []
        self.logs: list[dict[str, Any]] = []

    def log_execution(
        self,
        prompt: str,
        response: str,
        tokens: int,
        latency: float,
        guardrail_blocked: bool = False,
        hallucination_blocked: bool = False,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.requests_total += 1
            self.total_tokens_consumed += tokens
            self.latencies.append(latency)

            if guardrail_blocked:
                self.guardrail_blocks += 1
            if hallucination_blocked:
                self.hallucinations_detected += 1

            self.logs.insert(
                0,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "tokens": tokens,
                    "latency_sec": round(latency, 3),
                    "guardrail_blocked": guardrail_blocked,
                    "hallucination_blocked": hallucination_blocked,
                    "error": error,
                },
            )
            # Keep history capped at 100 items
            if len(self.logs) > 100:
                self.logs.pop()

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            blocked = self.guardrail_blocks + self.hallucinations_detected
            if self.requests_total > 0:
                success_rate = max(0.0, (self.requests_total - blocked) / self.requests_total * 100)
            else:
                success_rate = 100.0
            return {
                "requests_total": self.requests_total,
                "guardrail_blocks": self.guardrail_blocks,
                "hallucinations_detected": self.hallucinations_detected,
                "total_tokens_consumed": self.total_tokens_consumed,
                "avg_latency": round(avg_latency, 3),
                "success_rate": round(success_rate, 2),
                "logs": self.logs,
            }


# Global singleton metrics instance
global_metrics = MetricsCollector()
```

### 3.3 Guardrails (`guardrails.py`)
```python
import json

import tiktoken
from pydantic import BaseModel, Field, ValidationError

from .exceptions import GuardrailViolationError


class Guardrail(BaseModel):
    name: str
    allowed_topics: list[str] = Field(default_factory=list)
    disallowed_keywords: list[str] = Field(default_factory=list)
    max_tokens: int = 1000
    json_schema: type[BaseModel] | None = None

    def validate_input(self, prompt: str) -> None:
        prompt_lower = prompt.lower()
        for kw in self.disallowed_keywords:
            if kw.lower() in prompt_lower:
                raise GuardrailViolationError(
                    f"Input blocked by guardrail '{self.name}': Disallowed keyword -> '{kw}'"
                )

        if self.allowed_topics:
            self._validate_allowed_topics(prompt_lower)

    def _validate_allowed_topics(self, prompt_lower: str) -> None:
        for topic in self.allowed_topics:
            if topic.lower() in prompt_lower:
                return
        topics_list = ", ".join(repr(t) for t in self.allowed_topics)
        raise GuardrailViolationError(
            f"Input blocked by guardrail '{self.name}': "
            f"prompt does not match any allowed topic ({topics_list})"
        )

    def validate_output(self, response_text: str, model: str = "gpt-4o-mini") -> None:
        encoding = tiktoken.encoding_for_model(model)
        token_count = len(encoding.encode(response_text))
        if token_count > self.max_tokens:
            raise GuardrailViolationError(
                f"Output blocked: {token_count} tokens exceeds limit of {self.max_tokens}"
            )
        if self.json_schema is not None:
            self._validate_json_schema(response_text)

    def _validate_json_schema(self, response_text: str) -> None:
        """Validate that response_text parses against the configured Pydantic schema."""
        if self.json_schema is None:
            return
        try:
            data = json.loads(response_text)
            self.json_schema.model_validate(data)
        except json.JSONDecodeError as exc:
            raise GuardrailViolationError(
                f"Output blocked: response is not valid JSON ({exc})"
            ) from exc
        except ValidationError as exc:
            raise GuardrailViolationError(
                f"Output blocked: JSON schema validation failed ({exc})"
            ) from exc
```

### 3.4 Memory Store (`memory.py`)
```python
class MemoryStore:
    """Manages working memory and sliding windows to prevent context drift."""

    def __init__(self, retention_window: int = 10) -> None:
        self.retention_window = retention_window
        self.history: list[dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > (self.retention_window * 2):
            self.history = self.history[-(self.retention_window * 2) :]

    def get_context(self) -> list[dict[str, str]]:
        return self.history

    def clear(self) -> None:
        self.history.clear()
```

### 3.5 Anti-Hallucination Verifier (`verifier.py`)
```python
import re
from collections.abc import Callable
from typing import Protocol

GroundingCheck = Callable[[str, str | None], bool]


class Verifier:
    """Pluggable grounding verifier.

    The default implementation is intentionally conservative:
    - If no context sources are provided, the output is accepted.
    - If context sources are provided, the output is accepted as long as it is non-empty.

    Replace `check_fn` with a custom function or use one of the built-in verifiers
    for real semantic grounding checks.
    """

    def __init__(self, check_fn: GroundingCheck | None = None) -> None:
        self.check_fn = check_fn or self._default_check

    @staticmethod
    def _default_check(response: str, context_sources: str | None = None) -> bool:
        if not context_sources:
            return True
        return len(response.strip()) > 0

    def verify_grounding(self, response: str, context_sources: str | None = None) -> bool:
        return self.check_fn(response, context_sources)


class GroundingVerifier(Protocol):
    """Protocol for custom grounding verifiers."""

    def verify(self, response: str, context_sources: str) -> bool: ...


_BASIC_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
    "from", "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "under", "and", "but", "or", "yet", "so", "if",
    "because", "although", "though", "while", "where", "when", "that",
    "which", "who", "whom", "whose", "what", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "yourselves", "themselves", "one", "ones", "all",
    "any", "both", "each", "few", "more", "most", "other", "some", "such",
    "no", "nor", "not", "only", "own", "same", "than", "too", "very", "just",
    "now", "then", "here", "there", "once", "again", "also", "back", "still",
    "even", "about", "up", "out", "down", "off", "over", "away", "on", "how",
    "why", "where", "what", "who", "whom", "whose", "which", "whatever",
    "whoever", "whomever", "whichever", "s", "t", "don", "doesn", "didn",
    "wasn", "weren", "haven", "hasn", "hadn", "won", "wouldn", "couldn",
    "shouldn", "mightn", "mustn", "needn", "daren", "oughtn", "shan",
}


def _tokenize(text: str) -> set[str]:
    """Extract lowercase alphanumeric tokens, excluding basic stopwords."""
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    return tokens - _BASIC_STOPWORDS


class KeywordOverlapVerifier:
    """Grounding verifier based on keyword overlap with context sources.

    Accepts the response if at least `threshold` fraction of non-trivial context
    words appear in the response. Defaults to 0.1 (10%).
    """

    def __init__(self, threshold: float = 0.1, min_common: int = 1) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0.0 and 1.0")
        self.threshold = threshold
        self.min_common = min_common

    def verify(self, response: str, context_sources: str) -> bool:
        response_tokens = _tokenize(response)
        context_tokens = _tokenize(context_sources)
        if not context_tokens:
            return True
        common = response_tokens & context_tokens
        overlap = len(common) / len(context_tokens)
        return overlap >= self.threshold and len(common) >= self.min_common

    def as_callable(self) -> GroundingCheck:
        return self.verify


class CitationVerifier:
    """Grounding verifier that requires explicit citation-style markers.

    Accepts the response only if it contains markers such as quotes, brackets,
    or parenthetical references that suggest it is grounded in provided sources.
    """

    _CITATION_PATTERNS = [
        r'"[^"]{4,}"',  # quoted text
        r"\[[^\]]+\]",  # [1], [source]
        r"\([^)]*\d{4}[^)]*\)",  # (Author, 2023)
        r"according to",
        r"stated in",
        r"mentioned in",
        r"source",
    ]

    def verify(self, response: str, context_sources: str) -> bool:
        if not context_sources:
            return True
        response_lower = response.lower()
        return any(re.search(pattern, response_lower) for pattern in self._CITATION_PATTERNS)

    def as_callable(self) -> GroundingCheck:
        return self.verify
```

### 3.6 Agent Orchestrator (`agent.py`)

The orchestrator delegates LLM calls to a pluggable `BaseProvider`. The default provider is `OpenAIProvider`; adapters for Anthropic, Cohere, and Gemini live in `providers.py`.

```python
import logging
import time
from typing import Any, cast

import tiktoken
from pydantic import BaseModel, ValidationError

from .config import load_config
from .exceptions import GuardrailViolationError, HallucinationDetectedError
from .guardrails import Guardrail
from .memory import MemoryStore
from .metrics import MetricsCollector, global_metrics
from .providers import BaseProvider, OpenAIProvider, get_provider
from .resilience import (
    AsyncRateLimiter,
    CircuitBreaker,
    RateLimiter,
    default_retryable_exceptions,
    make_retry,
)
from .verifier import Verifier

logger = logging.getLogger("llmrivotril")
ContextSources = str | list[str] | None
_UNSET = object()


def _get_tokenizer(model: str) -> Any:
    """Return a tiktoken encoder, falling back to cl100k_base for unknown models."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning("Unknown model %r for tiktoken; falling back to cl100k_base", model)
        return tiktoken.get_encoding("cl100k_base")


def _normalize_context_sources(context_sources: ContextSources) -> str | None:
    if context_sources is None:
        return None
    if isinstance(context_sources, str):
        return context_sources
    return "\n\n".join(context_sources)


class RivotrilAgent:
    def __init__(
        self,
        model: Any = _UNSET,
        guardrails: list[Guardrail] | None = None,
        memory: MemoryStore | None = None,
        api_key: Any = _UNSET,
        base_url: Any = _UNSET,
        verifier: Verifier | None = None,
        system_prompt: Any = _UNSET,
        request_timeout: Any = _UNSET,
        metrics: MetricsCollector | None = None,
        rate_limit_max_calls: Any = _UNSET,
        rate_limit_per_seconds: Any = _UNSET,
        retry_max_attempts: Any = _UNSET,
        retry_min_wait: Any = _UNSET,
        retry_max_wait: Any = _UNSET,
        circuit_failure_threshold: Any = _UNSET,
        circuit_recovery_timeout: Any = _UNSET,
        provider: str | BaseProvider | None = None,
    ) -> None:
        config = load_config()

        def _resolve(value: Any, key: str, default: Any) -> Any:
            if value is not _UNSET:
                return value
            return config.get(key, default)

        self.model = _resolve(model, "model", "gpt-4o-mini")
        self.api_key = _resolve(api_key, "api_key", None)
        self.base_url = _resolve(base_url, "base_url", None)
        self.system_prompt = _resolve(system_prompt, "system_prompt", None)
        self.request_timeout = _resolve(request_timeout, "request_timeout", None)
        rate_limit_max_calls = _resolve(rate_limit_max_calls, "rate_limit_max_calls", 0.0)
        rate_limit_per_seconds = _resolve(rate_limit_per_seconds, "rate_limit_per_seconds", 1.0)
        retry_max_attempts = _resolve(retry_max_attempts, "retry_max_attempts", 3)
        retry_min_wait = _resolve(retry_min_wait, "retry_min_wait", 1.0)
        retry_max_wait = _resolve(retry_max_wait, "retry_max_wait", 10.0)
        circuit_failure_threshold = _resolve(
            circuit_failure_threshold, "circuit_failure_threshold", 5
        )
        circuit_recovery_timeout = _resolve(
            circuit_recovery_timeout, "circuit_recovery_timeout", 30.0
        )

        self.guardrails = guardrails or []
        self.memory = memory or MemoryStore()
        self.metrics = metrics or global_metrics
        self.verifier = verifier or Verifier()
        self.tokenizer = _get_tokenizer(self.model)

        # Provider selection. Explicit argument wins, then env/config default.
        if provider is None:
            provider = cast("str | None", config.get("provider", "openai"))
        if isinstance(provider, str) and provider.lower() == "openai":
            self.provider: BaseProvider = OpenAIProvider(
                api_key=self.api_key, base_url=self.base_url
            )
        elif isinstance(provider, str):
            self.provider = get_provider(provider, api_key=self.api_key, base_url=self.base_url)
        elif isinstance(provider, BaseProvider):
            self.provider = provider
        else:
            self.provider = OpenAIProvider(api_key=self.api_key, base_url=self.base_url)

        self.rate_limiter: RateLimiter | None = None
        self.async_rate_limiter: AsyncRateLimiter | None = None
        if rate_limit_max_calls > 0:
            self.rate_limiter = RateLimiter(rate_limit_max_calls, rate_limit_per_seconds)
            self.async_rate_limiter = AsyncRateLimiter(rate_limit_max_calls, rate_limit_per_seconds)

        retry_exceptions = default_retryable_exceptions()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=circuit_recovery_timeout,
            expected_exception=retry_exceptions,
        )

        self._retry_decorator = make_retry(
            max_attempts=retry_max_attempts,
            min_wait=retry_min_wait,
            max_wait=retry_max_wait,
            exceptions=retry_exceptions,
        )

    def _build_messages(self, prompt: str) -> list[Any]:
        messages: list[Any] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self.memory.get_context())
        messages.append({"role": "user", "content": prompt})
        return messages

    def _run_preflight(self, prompt: str) -> None:
        for guardrail in self.guardrails:
            try:
                guardrail.validate_input(prompt)
            except GuardrailViolationError as exc:
                logger.warning("Guardrail %r blocked input: %s", guardrail.name, exc)
                raise

    def _run_post_generation(
        self,
        prompt: str,
        response_text: str,
        response: Any,
        context_sources: ContextSources,
    ) -> Any:
        normalized_context = _normalize_context_sources(context_sources)
        for guardrail in self.guardrails:
            try:
                guardrail.validate_output(response_text, model=self.model)
            except GuardrailViolationError as exc:
                logger.warning("Guardrail %r blocked output: %s", guardrail.name, exc)
                raise

        if not self.verifier.verify_grounding(response_text, normalized_context):
            logger.warning("Grounding verification failed for prompt")
            raise HallucinationDetectedError("Output failed grounding verification.")

        self.memory.add_turn("user", prompt)
        self.memory.add_turn("assistant", response_text)
        logger.debug("Memory updated after successful generation")
        return response

    def _llm_call_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.request_timeout is not None:
            kwargs["timeout"] = self.request_timeout
        return kwargs

    def _execute_structured(self, messages: list[Any], response_model: type[BaseModel]) -> Any:
        response = self.provider.complete(
            model=self.model,
            response_model=response_model,
            messages=messages,
            **self._llm_call_kwargs(),
        )
        return response.structured

    def _execute_unstructured(self, messages: list[Any]) -> Any:
        response = self.provider.complete(
            model=self.model,
            messages=messages,
            **self._llm_call_kwargs(),
        )
        return response.text

    async def _execute_structured_async(
        self, messages: list[Any], response_model: type[BaseModel]
    ) -> Any:
        response = await self.provider.acomplete(
            model=self.model,
            response_model=response_model,
            messages=messages,
            **self._llm_call_kwargs(),
        )
        return response.structured

    async def _execute_unstructured_async(self, messages: list[Any]) -> Any:
        response = await self.provider.acomplete(
            model=self.model,
            messages=messages,
            **self._llm_call_kwargs(),
        )
        return response.text

    def _call_llm(self, messages: list[Any], response_model: type[BaseModel] | None) -> Any:
        if self.rate_limiter is not None:
            self.rate_limiter.acquire()

        @self._retry_decorator
        def _call() -> Any:
            if response_model is not None:
                return self._execute_structured(messages, response_model)
            return self._execute_unstructured(messages)

        return self.circuit_breaker.call(_call)

    async def _call_llm_async(
        self, messages: list[Any], response_model: type[BaseModel] | None
    ) -> Any:
        if self.async_rate_limiter is not None:
            await self.async_rate_limiter.acquire()

        @self._retry_decorator
        async def _call() -> Any:
            if response_model is not None:
                return await self._execute_structured_async(messages, response_model)
            return await self._execute_unstructured_async(messages)

        return await self.circuit_breaker.call_async(_call)

    def run(
        self,
        prompt: str,
        response_model: type[BaseModel] | None = None,
        context_sources: ContextSources = None,
    ) -> Any:
        start_time = time.time()
        guardrail_blocked = False
        hallucination_blocked = False
        error_msg: str | None = None
        response_text = ""
        response: Any = None
        tokens = len(self.tokenizer.encode(prompt))
        if self.system_prompt:
            tokens += len(self.tokenizer.encode(self.system_prompt))

        logger.debug("Starting agent.run")
        try:
            self._run_preflight(prompt)
            messages = self._build_messages(prompt)
            response = self._call_llm(messages, response_model)

            if response_model is not None:
                response_text = response.model_dump_json()
            else:
                response_text = response

            tokens += len(self.tokenizer.encode(response_text))
            result = self._run_post_generation(prompt, response_text, response, context_sources)
            logger.info("Agent run completed successfully")
            return result

        except ValidationError as exc:
            guardrail_blocked = True
            error_msg = str(exc)
            logger.warning("Response model validation failed: %s", exc)
            raise GuardrailViolationError(
                f"Output blocked: response model validation failed ({exc})"
            ) from exc
        except GuardrailViolationError as exc:
            guardrail_blocked = True
            error_msg = str(exc)
            logger.warning("Guardrail violation: %s", exc)
            raise
        except HallucinationDetectedError as exc:
            hallucination_blocked = True
            error_msg = str(exc)
            logger.warning("Hallucination detected: %s", exc)
            raise
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("Agent run failed with unexpected error")
            raise
        finally:
            latency = time.time() - start_time
            self.metrics.log_execution(
                prompt=prompt,
                response=response_text if response_text else "N/A",
                tokens=tokens,
                latency=latency,
                guardrail_blocked=guardrail_blocked,
                hallucination_blocked=hallucination_blocked,
                error=error_msg,
            )

    async def run_async(
        self,
        prompt: str,
        response_model: type[BaseModel] | None = None,
        context_sources: ContextSources = None,
    ) -> Any:
        start_time = time.time()
        guardrail_blocked = False
        hallucination_blocked = False
        error_msg: str | None = None
        response_text = ""
        response: Any = None
        tokens = len(self.tokenizer.encode(prompt))
        if self.system_prompt:
            tokens += len(self.tokenizer.encode(self.system_prompt))

        logger.debug("Starting agent.run_async")
        try:
            self._run_preflight(prompt)
            messages = self._build_messages(prompt)
            response = await self._call_llm_async(messages, response_model)

            if response_model is not None:
                response_text = response.model_dump_json()
            else:
                response_text = response

            tokens += len(self.tokenizer.encode(response_text))
            result = self._run_post_generation(prompt, response_text, response, context_sources)
            logger.info("Agent async run completed successfully")
            return result

        except ValidationError as exc:
            guardrail_blocked = True
            error_msg = str(exc)
            logger.warning("Response model validation failed: %s", exc)
            raise GuardrailViolationError(
                f"Output blocked: response model validation failed ({exc})"
            ) from exc
        except GuardrailViolationError as exc:
            guardrail_blocked = True
            error_msg = str(exc)
            logger.warning("Guardrail violation: %s", exc)
            raise
        except HallucinationDetectedError as exc:
            hallucination_blocked = True
            error_msg = str(exc)
            logger.warning("Hallucination detected: %s", exc)
            raise
        except Exception as exc:
            error_msg = str(exc)
            logger.exception("Agent async run failed with unexpected error")
            raise
        finally:
            latency = time.time() - start_time
            self.metrics.log_execution(
                prompt=prompt,
                response=response_text if response_text else "N/A",
                tokens=tokens,
                latency=latency,
                guardrail_blocked=guardrail_blocked,
                hallucination_blocked=hallucination_blocked,
                error=error_msg,
            )
```

Key design points:

- Configuration is merged in the order **file < environment < constructor arguments** via `load_config()`.
- `provider` can be a string (`"openai"`, `"anthropic"`, `"cohere"`, `"gemini"`) or an instance of `BaseProvider`.
- Sync and async paths both use the provider abstraction; `run()` calls `provider.complete()` and `run_async()` calls `provider.acomplete()`.
- Resilience is applied uniformly: rate limiter, retry decorator, and circuit breaker wrap `_call_llm` / `_call_llm_async`.
- `context_sources` accepts a single string or a list of strings; lists are joined with blank lines before verification.

### 3.7 Local Dashboard Web Server (`server.py`)
```python
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from .metrics import global_metrics

app = FastAPI(title="LLM-Rivotril Local Dashboard")

_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
HTML_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def get_dashboard() -> str:
    return HTML_TEMPLATE


@app.get("/api/metrics")
def get_metrics() -> dict[str, Any]:
    return global_metrics.get_summary()


def run_dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
```

The HTML template lives in `src/llmrivotril/templates/dashboard.html` and is bundled with the package.

### 3.8 Command-Line Interface (`cli.py`)
```python
import click
from .server import run_dashboard


@click.group()
def cli() -> None:
    """LLM-Rivotril CLI tools for agent safety and metrics."""


@cli.command()
@click.option("--host", default="127.0.0.1", help="Host for the local web dashboard.")
@click.option("--port", default=8000, help="Port for the local web dashboard.")
def dashboard(host: str, port: int) -> None:
    """Launch the local web dashboard to monitor token consumption and guardrails."""
    click.echo(f"Starting LLM-Rivotril dashboard at http://{host}:{port} ...")
    run_dashboard(host=host, port=port)


if __name__ == "__main__":
    cli()
```

### 3.9 Package Initialization (`__init__.py`)
```python
from .agent import RivotrilAgent
from .exceptions import GuardrailViolationError, HallucinationDetectedError, LLMRivotrilError
from .guardrails import Guardrail
from .memory import MemoryStore
from .providers import (
    AnthropicProvider,
    BaseProvider,
    CohereProvider,
    GeminiProvider,
    OpenAIProvider,
    get_provider,
)
from .rag import Document, RAGPipeline
from .rag.chunkers import SimpleChunker
from .rag.loaders import MarkdownLoader, TextLoader
from .rag.retrievers import InMemoryEmbeddingRetriever, InMemoryKeywordRetriever
from .semantic import EmbeddingFaithfulnessVerifier, SemanticTopicGuardrail
from .verifier import ModelBasedFaithfulnessVerifier

__version__ = "0.5.0"
__all__ = [
    "RivotrilAgent",
    "Guardrail",
    "MemoryStore",
    "LLMRivotrilError",
    "GuardrailViolationError",
    "HallucinationDetectedError",
    "SemanticTopicGuardrail",
    "EmbeddingFaithfulnessVerifier",
    "ModelBasedFaithfulnessVerifier",
    "Document",
    "RAGPipeline",
    "TextLoader",
    "MarkdownLoader",
    "SimpleChunker",
    "InMemoryKeywordRetriever",
    "InMemoryEmbeddingRetriever",
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "CohereProvider",
    "GeminiProvider",
    "get_provider",
]
```

### 3.10 Multi-Provider Support (`providers.py`)

```python
from llmrivotril.providers import OpenAIProvider, AnthropicProvider

# OpenAI-compatible (default)
provider = OpenAIProvider(api_key="sk-...")

# Anthropic
provider = AnthropicProvider(api_key="sk-ant-...")

agent = RivotrilAgent(provider=provider)
```

Supported providers: `openai`, `anthropic`, `cohere`, `gemini`.

### 3.11 File-Based Configuration (`config.py`)

`RivotrilAgent` merges configuration from files, environment variables, and constructor arguments in the following precedence:

1. `pyproject.toml` section `[tool.llmrivotril]`
2. `llmrivotril.toml` in the working directory
3. `RIVOTRIL_CONFIG_FILE` pointing to a custom file
4. `RIVOTRIL_*` environment variables
5. Explicit constructor arguments

Example `llmrivotril.toml`:

```toml
model = "gpt-4o-mini"
provider = "anthropic"
api_key = "sk-..."
request_timeout = 30.0
```

### 3.12 Known Limitations & TODOs

The following gaps have been addressed in the current implementation:

1. **Dashboard offline mode**: Tailwind CSS is bundled in `src/llmrivotril/static/` and served locally.
2. **Provider support**: `llmrivotril.providers` includes adapters for OpenAI, Anthropic, Cohere, and Gemini.
3. **Configuration files**: `RivotrilAgent` reads `llmrivotril.toml`, `llmrivotril.yaml`, or `[tool.llmrivotril]` in `pyproject.toml`.
4. **Grounding verifier**: `ModelBasedFaithfulnessVerifier` provides an LLM-as-a-judge scorer for critical use cases.

## 4. Usage & Running the Dashboard

After installing your local package (`pip install -e ".[dev]"`), you can launch the live telemetry dashboard via terminal:

```bash
llmrivotril dashboard --port 8000
```

### Interactive examples

Run the mock demo (no API key required) to see guardrails, grounding checks,
JSON schema validation, and live dashboard telemetry:

```bash
python examples/demo.py --mock --dashboard
```

Run the side-by-side comparison to see the same scenarios with and without
`llmrivotril`:

```bash
python examples/comparison.py --mock
```