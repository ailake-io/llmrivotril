import asyncio
import time

import pytest

from llmrivotril.resilience import (
    AsyncRateLimiter,
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RateLimiter,
    default_retryable_exceptions,
)


def test_rate_limiter_allows_under_limit():
    limiter = RateLimiter(max_calls=10, per_seconds=1)
    assert limiter.acquire(blocking=False) is True


def test_rate_limiter_blocks_over_limit():
    limiter = RateLimiter(max_calls=1, per_seconds=10)
    assert limiter.acquire(blocking=False) is True
    assert limiter.acquire(blocking=False) is False


def test_rate_limiter_refills():
    limiter = RateLimiter(max_calls=2, per_seconds=1)
    assert limiter.acquire(blocking=False) is True
    assert limiter.acquire(blocking=False) is True
    assert limiter.acquire(blocking=False) is False
    time.sleep(0.6)
    assert limiter.acquire(blocking=False) is True


def test_circuit_breaker_closes_after_success():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=10)
    result = cb.call(lambda: "ok")
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_opens_after_failures():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=10)

    def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        cb.call(fail)
    with pytest.raises(ValueError):
        cb.call(fail)
    with pytest.raises(CircuitBreakerOpenError):
        cb.call(fail)


def test_circuit_breaker_half_open_then_closes():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

    def fail() -> None:
        raise ValueError("boom")

    with pytest.raises(ValueError):
        cb.call(fail)

    time.sleep(0.15)
    assert cb.state == CircuitState.HALF_OPEN

    result = cb.call(lambda: "ok")
    assert result == "ok"
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_only_counts_expected_exceptions():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout=10, expected_exception=ValueError)

    def key_error() -> None:
        raise KeyError("not counted")

    with pytest.raises(KeyError):
        cb.call(key_error)
    assert cb.state == CircuitState.CLOSED


def test_rate_limiter_acquire_timeout():
    limiter = RateLimiter(max_calls=1, per_seconds=10)
    assert limiter.acquire(blocking=False) is True
    start = time.monotonic()
    result = limiter.acquire(blocking=True, timeout=0.1)
    elapsed = time.monotonic() - start
    assert result is False
    assert elapsed < 0.2


def test_default_retryable_exceptions_prefers_openai_errors():
    exceptions = default_retryable_exceptions()
    import openai

    assert openai.RateLimitError in exceptions
    assert openai.APIConnectionError in exceptions
    assert openai.APITimeoutError in exceptions
    assert openai.InternalServerError in exceptions


@pytest.mark.asyncio
async def test_async_rate_limiter_allows_under_limit():
    limiter = AsyncRateLimiter(max_calls=10, per_seconds=1)
    assert await limiter.acquire() is True


@pytest.mark.asyncio
async def test_async_rate_limiter_blocks_over_limit():
    limiter = AsyncRateLimiter(max_calls=1, per_seconds=10)
    assert await limiter.acquire() is True
    assert await limiter.acquire(timeout=0.05) is False


@pytest.mark.asyncio
async def test_async_rate_limiter_refills():
    limiter = AsyncRateLimiter(max_calls=2, per_seconds=1)
    assert await limiter.acquire() is True
    assert await limiter.acquire() is True
    assert await limiter.acquire(timeout=0.05) is False
    await asyncio.sleep(0.6)
    assert await limiter.acquire() is True
