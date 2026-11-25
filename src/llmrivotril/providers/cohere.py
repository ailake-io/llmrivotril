"""Cohere provider."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..content import content_to_text

if TYPE_CHECKING:
    pass

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger("llmrivotril")


class CohereProvider(BaseProvider):
    """Cohere provider.

    Requires the ``cohere`` package. Uses the chat completion endpoint.
    Streaming filters the SDK's event stream down to ``text-generation``
    events.
    """

    name = "cohere"

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client: Any | None = None
        self._async_client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import cohere

            self._client = cohere.Client(self.api_key, **self.extra_kwargs)
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            import cohere

            self._async_client = cohere.AsyncClient(self.api_key, **self.extra_kwargs)
        return self._async_client

    def _build_cohere_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]], str]:
        """Extract the last user message as prompt and the rest as chat history."""
        system: str | None = None
        chat_history: list[dict[str, Any]] = []
        last_message: str | None = None
        for msg in messages:
            role = msg.get("role", "user")
            content = content_to_text(msg.get("content", ""))
            if role == "system":
                system = (system or "") + "\n" + content
            elif last_message is None and role == "user":
                last_message = content
            else:
                chat_history.append({"role": role, "message": content})
        return system.strip() if system else None, chat_history, last_message or ""

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        system, chat_history, message = self._build_cohere_messages(messages)

        if response_model is not None:
            message = self._inject_response_model_prompt(message, response_model)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "message": message,
            "chat_history": chat_history,
        }
        if system:
            request_kwargs["preamble"] = system

        request_kwargs.update(kwargs)
        response = client.chat(**request_kwargs)
        content = response.text

        if response_model is not None:
            parsed = json.loads(content)
            return ProviderResponse(structured=response_model.model_validate(parsed))
        return ProviderResponse(content=content)

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_async_client()
        system, chat_history, message = self._build_cohere_messages(messages)

        if response_model is not None:
            message = self._inject_response_model_prompt(message, response_model)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "message": message,
            "chat_history": chat_history,
        }
        if system:
            request_kwargs["preamble"] = system

        request_kwargs.update(kwargs)
        response = await client.chat(**request_kwargs)
        content = response.text

        if response_model is not None:
            parsed = json.loads(content)
            return ProviderResponse(structured=response_model.model_validate(parsed))
        return ProviderResponse(content=content)

    def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        client = self._get_client()
        system, chat_history, message = self._build_cohere_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "message": message,
            "chat_history": chat_history,
            **kwargs,
        }
        if system:
            request_kwargs["preamble"] = system

        for event in client.chat_stream(**request_kwargs):
            if getattr(event, "event_type", None) == "text-generation":
                yield event.text

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        client = self._get_async_client()
        system, chat_history, message = self._build_cohere_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "message": message,
            "chat_history": chat_history,
            **kwargs,
        }
        if system:
            request_kwargs["preamble"] = system

        async for event in client.chat_stream(**request_kwargs):
            if getattr(event, "event_type", None) == "text-generation":
                yield event.text
