"""Provider base class and shared response type."""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..content import content_to_text
from ..resilience import retryable_exceptions_for_provider

if TYPE_CHECKING:
    from collections.abc import Awaitable

logger = logging.getLogger("llmrivotril")


class ProviderResponse:
    """Normalized response from any provider.

    Mimics the subset of OpenAI's ``ChatCompletion`` used by the agent.
    """

    def __init__(
        self,
        content: str | None = None,
        structured: BaseModel | None = None,
        tool_calls: Any | None = None,
        raw_response: Any | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.content = content
        self.structured = structured
        self.tool_calls = tool_calls
        self.raw_response = raw_response
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    @property
    def text(self) -> str:
        """Return the response as text.

        For structured outputs, returns the JSON serialization.
        """
        if self.structured is not None:
            return self.structured.model_dump_json()
        return self.content or ""


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    name: str = "base"

    def retryable_exceptions(self) -> tuple[type[BaseException], ...]:
        """Return transient SDK exceptions that the agent may retry."""
        return retryable_exceptions_for_provider(self.name)

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.extra_kwargs = kwargs

    def generation_limit_kwargs(self, max_output_tokens: int) -> dict[str, Any]:
        """Translate the optional per-generation output budget to native SDK fields."""
        if self.name in {"openai", "azure_openai"}:
            return {"max_completion_tokens": max_output_tokens}
        if self.name == "gemini":
            return {"config": {"max_output_tokens": max_output_tokens}}
        if self.name == "bedrock":
            return {"inferenceConfig": {"maxTokens": max_output_tokens}}
        return {"max_tokens": max_output_tokens}

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Run a synchronous chat completion."""

    @abstractmethod
    def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> Awaitable[ProviderResponse]:
        """Run an asynchronous chat completion."""

    def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        """Run a synchronous streaming chat completion.

        Returns an iterator of text chunks. Providers that do not support
        streaming should raise ``NotImplementedError``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support streaming")

    def astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Run an asynchronous streaming chat completion.

        Returns an async iterator of text chunks. Providers that do not support
        streaming should raise ``NotImplementedError``.
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support async streaming")

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Flatten message list into a single prompt string for non-chat APIs."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content_to_text(content)}")
        return "\n\n".join(parts)

    def _inject_response_model_prompt(self, prompt: str, response_model: type[BaseModel]) -> str:
        """Append JSON schema instructions for providers without native structured output."""
        schema = response_model.model_json_schema()
        return (
            f"{prompt}\n\n"
            "You must respond with a single JSON object matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Respond only with the JSON object, no markdown."
        )
