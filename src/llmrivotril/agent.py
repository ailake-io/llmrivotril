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


_UNSET = object()


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
