"""OpenAI and Azure OpenAI providers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    pass

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger("llmrivotril")


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
        try:
            async for chunk in stream:
                yield chunk
        finally:
            await stream.close()


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
