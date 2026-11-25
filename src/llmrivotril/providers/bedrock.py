"""AWS Bedrock provider."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import threading
from collections.abc import AsyncIterator
from threading import Lock
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from .._async import run_sync
from ..content import content_to_text

if TYPE_CHECKING:
    pass

from .base import BaseProvider, ProviderResponse

logger = logging.getLogger("llmrivotril")


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
