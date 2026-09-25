"""Multi-provider adapters for ``RivotrilAgent``.

Each provider exposes a unified ``complete`` / ``acomplete`` interface so the
agent can use OpenAI-compatible APIs, Anthropic, Cohere, or Gemini without
coupling to any specific SDK.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger("llmrivotril")


class ProviderResponse:
    """Normalized response from any provider.

    Mimics the subset of OpenAI's ``ChatCompletion`` used by the agent.
    """

    def __init__(self, content: str | None = None, structured: BaseModel | None = None) -> None:
        self.content = content
        self.structured = structured

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
        **kwargs: Any,
    ) -> ProviderResponse:
        """Run a synchronous chat completion."""

    @abstractmethod
    def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> Awaitable[ProviderResponse]:
        """Run an asynchronous chat completion."""

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Flatten message list into a single prompt string for non-chat APIs."""
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"{role.upper()}: {content}")
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

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if response_model is not None:
            result = self._get_client().chat.completions.create(
                model=model, response_model=response_model, messages=messages, **kwargs
            )
            return ProviderResponse(structured=result)

        completion = self._get_base_client().chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return ProviderResponse(content=completion.choices[0].message.content)

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if response_model is not None:
            result = await self._get_async_client().chat.completions.create(
                model=model, response_model=response_model, messages=messages, **kwargs
            )
            return ProviderResponse(structured=result)

        completion = await self._get_async_base_client().chat.completions.create(
            model=model, messages=messages, **kwargs
        )
        return ProviderResponse(content=completion.choices[0].message.content)


class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider.

    Requires the ``anthropic`` package. Structured output is emulated by
    requesting JSON in the prompt and parsing it.
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
            from anthropic import Anthropic  # type: ignore[import-not-found]

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
                system = (system or "") + "\n" + str(msg.get("content", ""))
            else:
                chat_messages.append(
                    {"role": msg.get("role", "user"), "content": str(msg.get("content", ""))}
                )
        return system.strip() if system else None, chat_messages

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
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


class CohereProvider(BaseProvider):
    """Cohere provider.

    Requires the ``cohere`` package. Uses the chat completion endpoint.
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
            import cohere  # type: ignore[import-not-found]

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
            content = str(msg.get("content", ""))
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


class GeminiProvider(BaseProvider):
    """Google Gemini provider.

    Requires the ``google-generativeai`` package.
    """

    name = "gemini"

    def __init__(
        self, api_key: str | None = None, base_url: str | None = None, **kwargs: Any
    ) -> None:
        super().__init__(api_key=api_key, base_url=base_url, **kwargs)
        self._client: Any | None = None

    def _get_client(self) -> Any:
        if self._client is None:
            import google.generativeai as genai  # type: ignore[import-not-found]

            genai.configure(api_key=self.api_key, **self.extra_kwargs)
            self._client = genai
        return self._client

    def _build_gemini_content(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[str]]:
        system: str | None = None
        contents: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role == "system":
                system = (system or "") + "\n" + content
            else:
                contents.append(content)
        return system.strip() if system else None, contents

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        system, contents = self._build_gemini_content(messages)
        prompt = "\n\n".join(contents)

        if response_model is not None:
            prompt = self._inject_response_model_prompt(prompt, response_model)

        model_obj = client.GenerativeModel(model_name=model, system_instruction=system)
        response = model_obj.generate_content(prompt, **kwargs)
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
        **kwargs: Any,
    ) -> ProviderResponse:
        client = self._get_client()
        system, contents = self._build_gemini_content(messages)
        prompt = "\n\n".join(contents)

        if response_model is not None:
            prompt = self._inject_response_model_prompt(prompt, response_model)

        model_obj = client.GenerativeModel(model_name=model, system_instruction=system)
        response = await model_obj.generate_content_async(prompt, **kwargs)
        content = response.text

        if response_model is not None:
            parsed = json.loads(content)
            return ProviderResponse(structured=response_model.model_validate(parsed))
        return ProviderResponse(content=content)


_PROVIDER_REGISTRY: dict[str, Callable[..., BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "cohere": CohereProvider,
    "gemini": GeminiProvider,
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
