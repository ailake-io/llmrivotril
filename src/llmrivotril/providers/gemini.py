"""Google Gemini provider."""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from ..content import content_to_text

if TYPE_CHECKING:
    pass

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger("llmrivotril")


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
            request_kwargs["config"] = {
                "system_instruction": system,
                **request_kwargs.get("config", {}),
            }
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
            request_kwargs["config"] = {
                "system_instruction": system,
                **request_kwargs.get("config", {}),
            }
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
            request_kwargs["config"] = {
                "system_instruction": system,
                **request_kwargs.get("config", {}),
            }
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
            request_kwargs["config"] = {
                "system_instruction": system,
                **request_kwargs.get("config", {}),
            }
        response = client.aio.models.generate_content_stream(
            model=model, contents=prompt, **request_kwargs
        )
        if hasattr(response, "__await__"):
            response = await response
        async for chunk in response:
            text = getattr(chunk, "text", None)
            if text:
                yield text
