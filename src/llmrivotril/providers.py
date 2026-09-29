"""Multi-provider adapters for ``RivotrilAgent``.

Each provider exposes a unified ``complete`` / ``acomplete`` interface so the
agent can use OpenAI-compatible APIs, Anthropic, Cohere, or Gemini without
coupling to any specific SDK.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import threading
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from threading import Lock
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ._async import run_sync
from .content import content_to_text
from .resilience import retryable_exceptions_for_provider

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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


class OpenAIProvider(BaseProvider):
    """OpenAI-compatible provider, including Ollama, vLLM, etc.

    Uses ``instructor`` when ``response_model`` is provided.
    """

    name = "openai"

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._base_client: Any | None = None
        self._client: Any | None = None
        self._async_base_client: Any | None = None
        self._async_client: Any | None = None

    def _get_base_client(self) -> Any:
        if self._base_client is None:
            from openai import OpenAI

            self._base_client = OpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.extra_kwargs
            )
        return self._base_client

    def _get_client(self) -> Any:
        if self._client is None:
            import instructor

            self._client = instructor.from_openai(self._get_base_client())
        return self._client

    def _get_async_base_client(self) -> Any:
        if self._async_base_client is None:
            from openai import AsyncOpenAI

            self._async_base_client = AsyncOpenAI(
                api_key=self.api_key, base_url=self.base_url, **self.extra_kwargs
            )
        return self._async_base_client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            import instructor

            self._async_client = instructor.from_openai(self._get_async_base_client())
        return self._async_client

    def _extract_usage(self, response: Any) -> tuple[int | None, int | None]:
        """Extract prompt/completion token counts from a provider response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None
        return (
            getattr(usage, "prompt_tokens", None),
            getattr(usage, "completion_tokens", None),
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        request_kwargs: dict[str, Any] = {}
        if tools is not None:
            request_kwargs["tools"] = tools

        create_kwargs = {**request_kwargs, **kwargs}
        if response_model is not None:
            result = self._get_client().chat.completions.create(
                model=model, response_model=response_model, messages=messages, **create_kwargs
            )
            raw = getattr(result, "_raw_response", result)
            prompt_tokens, completion_tokens = self._extract_usage(raw)
            return ProviderResponse(
                structured=result,
                raw_response=raw,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        completion = self._get_base_client().chat.completions.create(
            model=model, messages=messages, **create_kwargs
        )
        message = completion.choices[0].message
        prompt_tokens, completion_tokens = self._extract_usage(completion)
        return ProviderResponse(
            content=message.content,
            tool_calls=getattr(message, "tool_calls", None),
            raw_response=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        request_kwargs: dict[str, Any] = {}
        if tools is not None:
            request_kwargs["tools"] = tools

        create_kwargs = {**request_kwargs, **kwargs}
        if response_model is not None:
            result = await self._get_async_client().chat.completions.create(
                model=model, response_model=response_model, messages=messages, **create_kwargs
            )
            raw = getattr(result, "_raw_response", result)
            prompt_tokens, completion_tokens = self._extract_usage(raw)
            return ProviderResponse(
                structured=result,
                raw_response=raw,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

        completion = await self._get_async_base_client().chat.completions.create(
            model=model, messages=messages, **create_kwargs
        )
        message = completion.choices[0].message
        prompt_tokens, completion_tokens = self._extract_usage(completion)
        return ProviderResponse(
            content=message.content,
            tool_calls=getattr(message, "tool_calls", None),
            raw_response=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        stream_kwargs = {**kwargs, "stream": True}
        return self._get_base_client().chat.completions.create(
            model=model, messages=messages, **stream_kwargs
        )

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        stream_kwargs = {**kwargs, "stream": True}
        stream = await self._get_async_base_client().chat.completions.create(
            model=model, messages=messages, **stream_kwargs
        )
        async for chunk in stream:
            yield chunk


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI provider.

    Uses the OpenAI SDK's dedicated ``AzureOpenAI``/``AsyncAzureOpenAI``
    clients -- a different auth/endpoint shape than plain OpenAI, which is
    why Azure isn't reachable by just pointing ``base_url`` at it (see the
    "Other Providers" section of the README). No extra install needed:
    these clients ship in the ``openai`` package, already a core dependency.

    Requires ``azure_endpoint`` and ``api_version`` (Azure's REST API is
    versioned by date, e.g. ``"2024-02-01"``; check your Azure resource for
    the version it supports). The ``model`` passed to ``complete()``/
    ``acomplete()``/``run()`` should be your Azure **deployment name**, not
    the underlying model name -- that's how Azure routes the request.

    Inherits ``complete``/``acomplete``/``stream``/``astream`` from
    ``OpenAIProvider`` unchanged; only client construction differs.
    """

    name = "azure_openai"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.azure_endpoint = azure_endpoint
        self.api_version = api_version

    def _get_base_client(self) -> Any:
        if self._base_client is None:
            from openai import AzureOpenAI

            self._base_client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,  # type: ignore[arg-type]
                api_version=self.api_version,
                **self.extra_kwargs,
            )
        return self._base_client

    def _get_async_base_client(self) -> Any:
        if self._async_base_client is None:
            from openai import AsyncAzureOpenAI

            self._async_base_client = AsyncAzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.azure_endpoint,  # type: ignore[arg-type]
                api_version=self.api_version,
                **self.extra_kwargs,
            )
        return self._async_base_client


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


class GeminiProvider(BaseProvider):
    """Google Gemini provider.

    Requires the ``google-genai`` package. The adapter uses the current
    ``google.genai.Client`` API for synchronous, asynchronous, and streaming
    generation.
    """

    name = "gemini"

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import importlib

            genai = importlib.import_module("google.genai")
            self._client = genai.Client(api_key=self.api_key, **self.extra_kwargs)
        return self._client

    def _build_gemini_content(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[Any] | list[str]]:
        system: str | None = None
        contents: list[str] = []
        rich_contents: list[Any] = []
        has_rich_content = False
        for msg in messages:
            role = msg.get("role", "user")
            raw_content = msg.get("content", "")
            if role == "system":
                system = (system or "") + "\n" + content_to_text(raw_content)
            elif isinstance(raw_content, list):
                from google.genai import types

                parts: list[Any] = []
                for part in raw_content:
                    part_type = part.get("type", "text")
                    if part_type == "text":
                        parts.append(types.Part.from_text(text=str(part.get("text", ""))))
                    elif part_type == "image_url":
                        image = part.get("image_url", {})
                        uri = image.get("url", "") if isinstance(image, dict) else str(image)
                        parts.append(self._gemini_part_from_uri(types, uri, "image"))
                    elif part_type in {"input_audio", "audio"}:
                        audio = part.get("input_audio", part.get("audio", {}))
                        data = audio.get("data", "") if isinstance(audio, dict) else ""
                        mime = (
                            audio.get("format", "audio/wav")
                            if isinstance(audio, dict)
                            else "audio/wav"
                        )
                        parts.append(
                            types.Part.from_bytes(data=base64.b64decode(data), mime_type=mime)
                        )
                    elif part_type in {"file", "document"}:
                        file_data = part.get("file", part.get("document", {}))
                        uri = (
                            file_data.get("uri", "")
                            if isinstance(file_data, dict)
                            else str(file_data)
                        )
                        parts.append(
                            self._gemini_part_from_uri(types, uri, "application/octet-stream")
                        )
                rich_contents.append(
                    types.Content(role="model" if role == "assistant" else "user", parts=parts)
                )
                has_rich_content = True
            else:
                text = str(raw_content)
                if has_rich_content:
                    rich_contents.append(
                        types.Content(
                            role="model" if role == "assistant" else "user",
                            parts=[types.Part.from_text(text=text)],
                        )
                    )
                else:
                    contents.append(text)
        return system.strip() if system else None, rich_contents if has_rich_content else contents

    @staticmethod
    def _gemini_part_from_uri(types: Any, uri: str, fallback_mime: str) -> Any:
        if uri.startswith("data:"):
            header, encoded = uri.split(",", 1)
            mime = header[5:].split(";", 1)[0] or fallback_mime
            return types.Part.from_bytes(data=base64.b64decode(encoded), mime_type=mime)
        mime = mimetypes.guess_type(uri)[0] or fallback_mime
        return types.Part.from_uri(file_uri=uri, mime_type=mime)

    def _gemini_response_prompt(
        self, contents: list[Any] | list[str], model: type[BaseModel]
    ) -> Any:
        if all(isinstance(item, str) for item in contents):
            return self._inject_response_model_prompt("\n\n".join(contents), model)
        from google.genai import types

        instruction = self._inject_response_model_prompt("", model).lstrip()
        return [
            *contents,
            types.Content(role="user", parts=[types.Part.from_text(text=instruction)]),
        ]

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        system, contents = self._build_gemini_content(messages)
        prompt = (
            "\n\n".join(contents) if all(isinstance(item, str) for item in contents) else contents
        )

        if response_model is not None:
            prompt = self._gemini_response_prompt(contents, response_model)

        request_kwargs = dict(kwargs)
        if system:
            request_kwargs.setdefault("config", {"system_instruction": system})
        response = client.models.generate_content(model=model, contents=prompt, **request_kwargs)
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
        client = self._get_client()
        system, contents = self._build_gemini_content(messages)
        prompt = (
            "\n\n".join(contents) if all(isinstance(item, str) for item in contents) else contents
        )

        if response_model is not None:
            prompt = self._gemini_response_prompt(contents, response_model)

        request_kwargs = dict(kwargs)
        if system:
            request_kwargs.setdefault("config", {"system_instruction": system})
        response = await client.aio.models.generate_content(
            model=model, contents=prompt, **request_kwargs
        )
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
        system, contents = self._build_gemini_content(messages)
        prompt = (
            "\n\n".join(contents) if all(isinstance(item, str) for item in contents) else contents
        )

        request_kwargs = dict(kwargs)
        if system:
            request_kwargs.setdefault("config", {"system_instruction": system})
        response = client.models.generate_content_stream(
            model=model, contents=prompt, **request_kwargs
        )
        for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        client = self._get_client()
        system, contents = self._build_gemini_content(messages)
        prompt = (
            "\n\n".join(contents) if all(isinstance(item, str) for item in contents) else contents
        )

        request_kwargs = dict(kwargs)
        if system:
            request_kwargs.setdefault("config", {"system_instruction": system})
        response = client.aio.models.generate_content_stream(
            model=model, contents=prompt, **request_kwargs
        )
        if hasattr(response, "__await__"):
            response = await response
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text


class BedrockProvider(BaseProvider):
    """AWS Bedrock provider, via the Bedrock Runtime Converse API.

    Requires ``boto3`` (``pip install "llmrivotril[bedrock]"``) and AWS
    credentials resolved the normal boto3 way (environment variables,
    ``~/.aws/credentials``, an instance role, etc.) -- ``api_key``/
    ``base_url`` are accepted for interface symmetry with the other
    providers but unused. ``model`` should be a Bedrock model ID (e.g.
    ``"anthropic.claude-3-5-sonnet-20241022-v2:0"``) or inference profile
    ARN.

    Converse gives one request/response shape across model families on
    Bedrock (Anthropic, Meta, Amazon, Mistral, Cohere) instead of each
    family's own body schema, which is what this adapter is built on.
    Structured output is emulated the same way as the Anthropic/Cohere/
    Gemini adapters (JSON-schema instructions injected into the prompt).

    ``tools=`` is translated to Converse's ``toolConfig`` shape for
    ``complete()``/``acomplete()`` and for streaming. Converse's
    ``contentBlockStart``/``contentBlockDelta`` tool-use events are accumulated
    into a final ``ProviderResponse.tool_calls`` chunk so the generic agent
    tool loop can execute them and issue the follow-up request.

    There's no official async boto3 client, so ``acomplete`` runs the
    synchronous call in a worker thread rather than being natively
    non-blocking (``astream`` bridges it through a producer thread + queue
    instead, for genuine incremental delivery).
    """

    name = "bedrock"

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        region_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self.region_name = region_name
        self._client: Any | None = None
        self._load_lock = Lock()

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        with self._load_lock:
            if self._client is not None:
                return self._client
            import boto3

            self._client = boto3.client(
                "bedrock-runtime", region_name=self.region_name, **self.extra_kwargs
            )
            return self._client

    def _build_converse_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]]]:
        system: list[dict[str, Any]] | None = None
        converse_messages: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                system = (system or []) + [{"text": content_to_text(msg.get("content", ""))}]
            elif role == "tool":
                # agent.py's tool-calling loop appends one of these per
                # executed call; Converse expects the result correlated back
                # to the model's toolUse by id, as a "user" turn.
                converse_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "toolResult": {
                                    "toolUseId": msg.get("tool_call_id", ""),
                                    "content": [{"text": str(msg.get("content", ""))}],
                                }
                            }
                        ],
                    }
                )
            else:
                converse_role = "assistant" if role == "assistant" else "user"
                content = self._bedrock_content_blocks(msg.get("content", ""))
                converse_messages.append({"role": converse_role, "content": content})
        return system, converse_messages

    @staticmethod
    def _bedrock_content_blocks(content: Any) -> list[dict[str, Any]]:
        if not isinstance(content, list):
            return [{"text": str(content)}]

        blocks: list[dict[str, Any]] = []
        for part in content:
            part_type = part.get("type", "text")
            if part_type == "text":
                blocks.append({"text": str(part.get("text", ""))})
                continue

            if part_type == "image_url":
                image = part.get("image_url", {})
                uri = image.get("url", "") if isinstance(image, dict) else str(image)
                blocks.append({"image": BedrockProvider._binary_source(uri, "image")})
                continue

            if part_type in {"file", "document"}:
                document = part.get("file", part.get("document", {}))
                uri = document.get("uri", "") if isinstance(document, dict) else str(document)
                blocks.append({"document": BedrockProvider._binary_source(uri, "document")})
                continue

            if part_type in {"input_audio", "audio"}:
                audio = part.get("input_audio", part.get("audio", {}))
                data = audio.get("data", "") if isinstance(audio, dict) else ""
                fmt = audio.get("format", "wav") if isinstance(audio, dict) else "wav"
                blocks.append(
                    {"audio": {"format": fmt, "source": {"bytes": base64.b64decode(data)}}}
                )
                continue

            raise ValueError(f"Unsupported multimodal content part for Bedrock: {part_type!r}")
        return blocks

    @staticmethod
    def _binary_source(uri: str, block_type: str) -> dict[str, Any]:
        if not uri.startswith("data:"):
            raise ValueError(
                f"Bedrock {block_type} content requires a data URL so the SDK can receive bytes"
            )
        header, encoded = uri.split(",", 1)
        mime = header[5:].split(";", 1)[0]
        fmt = mime.split("/", 1)[-1] if "/" in mime else "bin"
        if block_type == "image":
            return {"format": fmt, "source": {"bytes": base64.b64decode(encoded)}}
        return {
            "format": fmt,
            "name": "document",
            "source": {"bytes": base64.b64decode(encoded)},
        }

    @staticmethod
    def _tools_to_tool_config(tools: list[dict[str, Any]] | None) -> dict[str, Any] | None:
        """Translate OpenAI-style tool schemas into Converse's ``toolConfig``."""
        if not tools:
            return None
        tool_specs = []
        for tool in tools:
            function = tool.get("function", tool)
            tool_specs.append(
                {
                    "toolSpec": {
                        "name": function.get("name", ""),
                        "description": function.get("description", ""),
                        "inputSchema": {"json": function.get("parameters", {"type": "object"})},
                    }
                }
            )
        return {"tools": tool_specs}

    def _build_request(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None,
        tools: list[dict[str, Any]] | None,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        system, converse_messages = self._build_converse_messages(messages)
        if response_model is not None and converse_messages:
            last_block = converse_messages[-1]["content"][0]
            if "text" in last_block:
                last_block["text"] = self._inject_response_model_prompt(
                    last_block["text"], response_model
                )
        request: dict[str, Any] = {"modelId": model, "messages": converse_messages}
        if system:
            request["system"] = system
        tool_config = self._tools_to_tool_config(tools)
        if tool_config is not None:
            request["toolConfig"] = tool_config
        request.update(kwargs)
        return request

    @staticmethod
    def _extract_tool_calls(content_blocks: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
        tool_calls = [
            {
                "id": block["toolUse"].get("toolUseId", ""),
                "name": block["toolUse"].get("name", ""),
                "arguments": block["toolUse"].get("input", {}),
            }
            for block in content_blocks
            if "toolUse" in block
        ]
        return tool_calls or None

    def _parse_result(
        self, result: dict[str, Any], response_model: type[BaseModel] | None
    ) -> ProviderResponse:
        output_message = result.get("output", {}).get("message", {})
        content_blocks = output_message.get("content", [])
        content = "".join(block.get("text", "") for block in content_blocks)
        tool_calls = self._extract_tool_calls(content_blocks)
        usage = result.get("usage", {})
        prompt_tokens = usage.get("inputTokens")
        completion_tokens = usage.get("outputTokens")

        if response_model is not None:
            parsed = json.loads(content)
            return ProviderResponse(
                structured=response_model.model_validate(parsed),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        return ProviderResponse(
            content=content,
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        request = self._build_request(messages, model, response_model, tools, kwargs)
        result = client.converse(**request)
        return self._parse_result(result, response_model)

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return await run_sync(self.complete, messages, model, response_model, tools, **kwargs)

    def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> Any:
        client = self._get_client()
        tools = kwargs.pop("tools", None)
        request = self._build_request(messages, model, None, tools, kwargs)
        response = client.converse_stream(**request)
        tool_calls: dict[int, dict[str, Any]] = {}
        for event in response["stream"]:
            start = event.get("contentBlockStart", {}).get("start", {}).get("toolUse")
            if start is not None:
                index = event.get("contentBlockStart", {}).get("contentBlockIndex", 0)
                tool_calls[index] = {
                    "id": start.get("toolUseId", ""),
                    "name": start.get("name", ""),
                    "arguments": "",
                }
                continue

            block_delta = event.get("contentBlockDelta", {})
            index = block_delta.get("contentBlockIndex", 0)
            delta = block_delta.get("delta", {})
            tool_delta = delta.get("toolUse")
            if tool_delta is not None:
                entry = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                entry["arguments"] += tool_delta.get("input", "")
                continue

            text = delta.get("text")
            if text:
                yield text

        if tool_calls:
            parsed_calls: list[dict[str, Any]] = []
            for call in tool_calls.values():
                try:
                    arguments = json.loads(call["arguments"]) if call["arguments"] else {}
                except json.JSONDecodeError:
                    arguments = {}
                parsed_calls.append(
                    {"id": call["id"], "name": call["name"], "arguments": arguments}
                )
            yield ProviderResponse(tool_calls=parsed_calls)

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        **kwargs: Any,
    ) -> AsyncIterator[Any]:
        """Bridge the sync ``stream()`` generator into an async one.

        boto3 has no async client, so the sync generator runs in a worker
        thread that pushes chunks onto a queue as they arrive, and this
        coroutine yields them as they're polled off the queue -- giving
        genuine incremental delivery rather than blocking for the whole
        response before yielding anything.
        """
        loop = asyncio.get_running_loop()
        chunk_queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()
        stopped = threading.Event()

        def _enqueue(item: Any) -> None:
            if stopped.is_set():
                return
            try:
                loop.call_soon_threadsafe(chunk_queue.put_nowait, item)
            except RuntimeError:
                # The consumer may have been cancelled and the event loop may
                # already be closing. There is no useful work left for the
                # producer in that case.
                stopped.set()

        def _produce() -> None:
            try:
                for chunk in self.stream(messages, model, **kwargs):
                    if stopped.is_set():
                        break
                    _enqueue(chunk)
            except Exception as exc:  # noqa: BLE001 - forwarded to the async caller below
                _enqueue(exc)
            finally:
                _enqueue(sentinel)

        thread = threading.Thread(target=_produce, daemon=True)
        thread.start()

        try:
            while True:
                item = await chunk_queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stopped.set()
            thread.join(timeout=1.0)


_PROVIDER_REGISTRY: dict[str, Callable[..., BaseProvider]] = {
    "openai": OpenAIProvider,
    "azure_openai": AzureOpenAIProvider,
    "anthropic": AnthropicProvider,
    "cohere": CohereProvider,
    "gemini": GeminiProvider,
    "bedrock": BedrockProvider,
}


def get_provider(name: str, **kwargs: Any) -> BaseProvider:
    """Resolve a provider name to a provider instance."""
    name = name.lower()
    if name not in _PROVIDER_REGISTRY:
        available = ", ".join(_PROVIDER_REGISTRY)
        raise ValueError(f"Unknown provider '{name}'. Available providers: {available}")
    return _PROVIDER_REGISTRY[name](**kwargs)


def available_providers() -> list[str]:
    """Return the list of supported provider names."""
    return list(_PROVIDER_REGISTRY.keys())
