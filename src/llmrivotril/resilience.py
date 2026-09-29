"""Resilience primitives for LLM calls: rate limiting, retry, and circuit breaker."""

import asyncio
import importlib
import logging
import time
from collections.abc import Callable
from enum import Enum
from threading import Lock
from typing import Any, TypeVar

from tenacity import (
    retry as tenacity_retry,
)
from tenacity import (
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger("llmrivotril")

F = TypeVar("F", bound=Callable[..., Any])


def default_retryable_exceptions() -> tuple[type[BaseException], ...]:
    """Return the set of OpenAI infrastructure exceptions worth retrying.

    Falls back to ``(Exception,)`` when ``openai`` is not installed, so the
    primitives remain usable in tests and non-OpenAI environments.
    """
    try:
        import openai
    except Exception:  # pragma: no cover
        return (Exception,)

    return (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )


def _load_exception_types(
    module_name: str, names: tuple[str, ...]
) -> tuple[type[BaseException], ...]:
    """Load exception classes without importing optional provider SDKs eagerly."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return ()

    exceptions: list[type[BaseException]] = []
    for name in names:
        exception = getattr(module, name, None)
        if isinstance(exception, type) and issubclass(exception, BaseException):
            exceptions.append(exception)
    return tuple(exceptions)


def retryable_exceptions_for_provider(provider: str) -> tuple[type[BaseException], ...]:
    """Return transient exception types for a named provider.

    Provider SDKs are optional, so their exception modules are imported only
    when the corresponding provider is selected. Unknown providers retain the
    OpenAI-compatible default used by earlier releases.
    """
    provider = provider.lower()
    if provider in {"openai", "azure_openai"}:
        return default_retryable_exceptions()

    specs: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
        "anthropic": (
            (
                "anthropic",
                ("RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"),
            ),
        ),
        "cohere": (
            (
                "cohere.errors",
                (
                    "TooManyRequestsError",
                    "ServiceUnavailableError",
                    "GatewayTimeoutError",
                    "InternalServerError",
                ),
            ),
        ),
        "gemini": (
            (
                "google.api_core.exceptions",
                (
                    "ResourceExhausted",
                    "ServiceUnavailable",
                    "DeadlineExceeded",
                    "InternalServerError",
                ),
            ),
        ),
        "bedrock": (
            (
                "botocore.exceptions",
                (
                    "ConnectTimeoutError",
                    "ConnectionClosedError",
                    "EndpointConnectionError",
                    "ReadTimeoutError",
                ),
            ),
        ),
    }
    loaded: list[type[BaseException]] = []
    for module_name, names in specs.get(provider, ()):
        loaded.extend(_load_exception_types(module_name, names))
    return tuple(dict.fromkeys(loaded)) or default_retryable_exceptions()


class RateLimiter:
    """Token-bucket rate limiter.

    Limits calls to `max_calls` per `per_seconds` window. Useful to avoid
    hitting provider rate limits or runaway costs.
    """

    def __init__(self, max_calls: float = 10.0, per_seconds: float = 1.0) -> None:
        if max_calls <= 0 or per_seconds <= 0:
            raise ValueError("max_calls and per_seconds must be positive")
        self._max_calls = max_calls
        self._per_seconds = per_seconds
        self._tokens = float(max_calls)
        self._last_update = time.monotonic()
        self._lock = Lock()

    def acquire(self, blocking: bool = True, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._last_update = now
                self._tokens = min(self._max_calls, self._tokens + elapsed * self._rate())

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            if not blocking:
                return False

            sleep_time = 1.0 / self._rate()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                sleep_time = min(sleep_time, remaining)
            time.sleep(sleep_time)

    def _rate(self) -> float:
        return self._max_calls / self._per_seconds


class AsyncRateLimiter:
    """Async-compatible token-bucket rate limiter.

    Uses ``asyncio.Lock`` and ``asyncio.sleep`` so it does not block the event
    loop when used inside ``RivotrilAgent.run_async``.
    """

    def __init__(self, max_calls: float = 10.0, per_seconds: float = 1.0) -> None:
        if max_calls <= 0 or per_seconds <= 0:
            raise ValueError("max_calls and per_seconds must be positive")
        self._max_calls = max_calls
        self._per_seconds = per_seconds
        self._tokens = float(max_calls)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, timeout: float | None = None) -> bool:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_update
                self._last_update = now
                self._tokens = min(self._max_calls, self._tokens + elapsed * self._rate())

                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True

            sleep_time = 1.0 / self._rate()
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                sleep_time = min(sleep_time, remaining)
            await asyncio.sleep(sleep_time)

    def _rate(self) -> float:
        return self._max_calls / self._per_seconds


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpenError(Exception):
    """Raised when the circuit breaker is open."""


class CircuitBreaker:
    """Simple circuit breaker for LLM provider failures.

    After `failure_threshold` consecutive failures, the circuit opens for
    `recovery_timeout` seconds. A single success in half-open state closes it;
    a single failure reopens it.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        expected_exception: type[BaseException] | tuple[type[BaseException], ...] = Exception,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time: float | None = None
        self._half_open_probe_in_flight = False
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    def _maybe_transition_to_half_open(self) -> None:
        """Flip OPEN -> HALF_OPEN once the recovery timeout has elapsed.

        Caller must hold ``self._lock``.
        """
        if self._state != CircuitState.OPEN:
            return
        if self._last_failure_time is None:
            return
        elapsed = time.monotonic() - self._last_failure_time
        if elapsed >= self.recovery_timeout:
            self._state = CircuitState.HALF_OPEN

    def _acquire_permission(self) -> bool:
        """Atomically decide whether this call may proceed.

        While HALF_OPEN, only a single in-flight call is allowed through as a
        probe; concurrent callers are rejected until that probe resolves.
        Checking ``state`` and mutating it as two separate locked sections (as
        a naive implementation might) lets multiple threads all observe
        HALF_OPEN and dispatch a probe simultaneously, defeating the "single
        probe" guarantee under concurrent load -- so this does both atomically.
        """
        with self._lock:
            self._maybe_transition_to_half_open()
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.HALF_OPEN and not self._half_open_probe_in_flight:
                self._half_open_probe_in_flight = True
                return True
            return False

    def call(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._acquire_permission():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = fn(*args, **kwargs)
        except self.expected_exception as exc:
            self._record_failure()
            raise exc

        self._record_success()
        return result

    async def call_async(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        if not self._acquire_permission():
            raise CircuitBreakerOpenError("Circuit breaker is OPEN")

        try:
            result = await fn(*args, **kwargs)
        except self.expected_exception as exc:
            self._record_failure()
            raise exc

        self._record_success()
        return result

    def _record_failure(self) -> None:
        with self._lock:
            self._half_open_probe_in_flight = False
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._failure_count >= self.failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.warning("Circuit breaker opened after %d failures", self._failure_count)
                self._state = CircuitState.OPEN

    def _record_success(self) -> None:
        with self._lock:
            previous_state = self._state
            self._half_open_probe_in_flight = False
            self._failure_count = 0
            self._last_failure_time = None
            self._state = CircuitState.CLOSED
            if previous_state != CircuitState.CLOSED:
                logger.info("Circuit breaker closed after success")


def make_retry(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 10.0,
    exceptions: type[BaseException] | tuple[type[BaseException], ...] = Exception,
) -> Callable[[F], F]:
    """Create a tenacity retry decorator for LLM calls."""
    return tenacity_retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        reraise=True,
    )
