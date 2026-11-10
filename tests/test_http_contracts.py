"""Wire-level contracts: real SDKs, in-process HTTP, no network or credentials."""

import json

import httpx
import openai
import pytest

from llmrivotril.providers import AzureOpenAIProvider, OpenAIProvider

MESSAGES = [{"role": "user", "content": "local test"}]


def test_sdk_retries_transient_failure_locally():
    requests = []

    def handle(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429, headers={"retry-after-ms": "1"}, json={"error": {"message": "try again"}}
            )
        return httpx.Response(200, json=completion())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = OpenAIProvider(
            api_key="local-dummy",
            base_url="https://local.invalid/v1",
            http_client=client,
            max_retries=1,
        )
        assert provider.complete(MESSAGES, "local").text == "OK"
    assert len(requests) == 2


def test_tools_wire_contract():
    tool = {
        "type": "function",
        "function": {"name": "local_tool", "parameters": {"type": "object", "properties": {}}},
    }

    def handle(request):
        assert json.loads(request.content)["tools"] == [tool]
        body = completion()
        body["choices"][0]["message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_local",
                    "type": "function",
                    "function": {"name": "local_tool", "arguments": "{}"},
                }
            ],
        }
        body["choices"][0]["finish_reason"] = "tool_calls"
        return httpx.Response(200, json=body)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = provider_for("openai", client).complete(MESSAGES, "local", tools=[tool])
        assert result.tool_calls[0].function.name == "local_tool"


def completion():
    return {
        "id": "local",
        "object": "chat.completion",
        "created": 0,
        "model": "local",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "OK"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def provider_for(kind, client):
    if kind == "azure":
        return AzureOpenAIProvider(
            api_key="local-dummy",
            azure_endpoint="https://local.invalid",
            api_version="2024-10-21",
            http_client=client,
            max_retries=0,
        )
    return OpenAIProvider(
        api_key="local-dummy",
        base_url="https://local.invalid/v1",
        http_client=client,
        max_retries=0,
    )


@pytest.mark.parametrize("kind", ["openai", "azure"])
def test_completion_wire_contract(kind):
    def handle(request):
        body = json.loads(request.content)
        assert body["messages"] == MESSAGES
        assert body["model"] == "local"
        assert body["max_tokens"] == 8
        if kind == "azure":
            assert request.url.path == "/openai/deployments/local/chat/completions"
            assert request.url.params["api-version"] == "2024-10-21"
            assert request.headers["api-key"] == "local-dummy"
        else:
            assert request.url.path == "/v1/chat/completions"
            assert request.headers["authorization"] == "Bearer local-dummy"
        return httpx.Response(200, json=completion())

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        result = provider_for(kind, client).complete(MESSAGES, "local", max_tokens=8)
    assert result.text == "OK"
    assert (result.prompt_tokens, result.completion_tokens) == (3, 1)


@pytest.mark.parametrize("kind", ["openai", "azure"])
@pytest.mark.asyncio
async def test_async_completion_wire_contract(kind):
    def handle(request):
        assert json.loads(request.content)["messages"] == MESSAGES
        return httpx.Response(200, json=completion())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        provider = provider_for(kind, client)
        try:
            result = await provider.acomplete(MESSAGES, "local")
        finally:
            await provider._get_async_base_client().close()
    assert result.text == "OK"


@pytest.mark.parametrize(
    "status,error",
    [
        (401, openai.AuthenticationError),
        (429, openai.RateLimitError),
        (500, openai.InternalServerError),
    ],
)
def test_http_failures_are_real_sdk_exceptions(status, error):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(status, json={"error": {"message": "local failure"}})
        )
    ) as client:
        with pytest.raises(error):
            provider_for("openai", client).complete(MESSAGES, "local")


def test_timeout_is_propagated():
    def handle(request):
        assert request.extensions["timeout"]["read"] == 0.1
        raise httpx.ReadTimeout("local timeout", request=request)

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        with pytest.raises(openai.APITimeoutError):
            provider_for("openai", client).complete(MESSAGES, "local", timeout=0.1)


def stream_response(request):
    assert json.loads(request.content)["stream"] is True
    events = []
    for text in ("O", "K"):
        chunk = {
            "id": "local",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "local",
            "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
        }
        events.append("data: " + json.dumps(chunk) + "\n\n")
    events.append("data: [DONE]\n\n")
    return httpx.Response(
        200, headers={"content-type": "text/event-stream"}, content="".join(events)
    )


def test_sse_stream_parsed_by_real_sdk():
    with httpx.Client(transport=httpx.MockTransport(stream_response)) as client:
        stream = provider_for("openai", client).stream(MESSAGES, "local")
        try:
            assert "".join(chunk.choices[0].delta.content for chunk in stream) == "OK"
        finally:
            stream.close()


@pytest.mark.asyncio
async def test_async_sse_stream_parsed_by_real_sdk():
    async with httpx.AsyncClient(transport=httpx.MockTransport(stream_response)) as client:
        provider = provider_for("openai", client)
        try:
            chunks = [chunk async for chunk in provider.astream(MESSAGES, "local")]
        finally:
            await provider._get_async_base_client().close()
        assert "".join(chunk.choices[0].delta.content for chunk in chunks) == "OK"
