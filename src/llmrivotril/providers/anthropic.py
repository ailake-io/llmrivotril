"""Anthropic provider."""

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


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider.

    Requires the ``anthropic`` package. Structured output is emulated by
    requesting JSON in the prompt and parsing it. Streaming uses the SDK's
    ``messages.stream()`` context manager.
    """

    name = "anthropic"

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client: Any | None = None
        self._async_client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            from anthropic import Anthropic

            self._client = Anthropic(
                api_key=self.api_key, base_url=self.base_url, **self.extra_kwargs
            )
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from anthropic import AsyncAnthropic

            self._async_client = AsyncAnthropic(
                api_key=self.api_key, base_url=self.base_url, **self.extra_kwargs
            )
        return self._async_client

    def _build_anthropic_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Separate system message from user/assistant messages."""
        system: str | None = None
        chat_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = (system or "") + "\n" + content_to_text(msg.get("content", ""))
            else:
                content = msg.get("content", "")
                if not isinstance(content, list):
                    content = str(content)
                else:
                    content = self._anthropic_content_blocks(content)
                chat_messages.append({"role": msg.get("role", "user"), "content": content})
        return system.strip() if system else None, chat_messages

    @staticmethod
    def _anthropic_content_blocks(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for part in content:
            part_type = part.get("type", "text")
            if part_type == "text":
                blocks.append({"type": "text", "text": str(part.get("text", ""))})
                continue
            if part_type == "image_url":
                image = part.get("image_url", {})
                uri = image.get("url", "") if isinstance(image, dict) else str(image)
                if not uri.startswith("data:"):
                    raise ValueError("Anthropic image content requires a data URL")
                header, encoded = uri.split(",", 1)
                media_type = header[5:].split(";", 1)[0] or "image/png"
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    }
                )
                continue
            if part_type == "document":
                document = part.get("document", {})
                uri = document.get("uri", "") if isinstance(document, dict) else str(document)
                if not uri.startswith("data:"):
                    raise ValueError("Anthropic document content requires a data URL")
                header, encoded = uri.split(",", 1)
                media_type = header[5:].split(";", 1)[0] or "application/pdf"
                blocks.append(
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": encoded,
                        },
                    }
                )
                continue
            raise ValueError(f"Unsupported multimodal content part for Anthropic: {part_type!r}")
        return blocks

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        system, chat_messages = self._build_anthropic_messages(messages)
        prompt_text = self._messages_to_prompt(chat_messages)

        if response_model is not None:
            prompt_text = self._inject_response_model_prompt(prompt_text, response_model)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            request_kwargs["system"] = system

        response = client.messages.create(**request_kwargs)
        content = "\n".join(block.text for block in response.content if hasattr(block, "text"))

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
        system, chat_messages = self._build_anthropic_messages(messages)
        prompt_text = self._messages_to_prompt(chat_messages)

        if response_model is not None:
            prompt_text = self._inject_response_model_prompt(prompt_text, response_model)

        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            request_kwargs["system"] = system

        response = await client.messages.create(**request_kwargs)
        content = "\n".join(block.text for block in response.content if hasattr(block, "text"))

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
        system, chat_messages = self._build_anthropic_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            request_kwargs["system"] = system

        with client.messages.stream(**request_kwargs) as stream:
            yield from stream.text_stream

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        client = self._get_async_client()
        system, chat_messages = self._build_anthropic_messages(messages)
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": chat_messages,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system:
            request_kwargs["system"] = system

        async with client.messages.stream(**request_kwargs) as stream:
            async for text in stream.text_stream:
                yield text
