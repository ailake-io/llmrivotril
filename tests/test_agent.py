from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from llmrivotril import Guardrail, MemoryStore, RivotrilAgent
from llmrivotril.exceptions import GuardrailViolationError, TokenBudgetExceededError
from llmrivotril.providers import BaseProvider, ProviderResponse


def _fake_tool_call_chunk(index, tool_id=None, name=None, arguments=None):
    """Build a fake OpenAI-style streaming chunk carrying a tool-call delta fragment."""
    function = None
    if name is not None or arguments is not None:
        function = SimpleNamespace(name=name, arguments=arguments)
    tool_call_delta = SimpleNamespace(index=index, id=tool_id, function=function)
    delta = SimpleNamespace(content=None, tool_calls=[tool_call_delta])
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class Answer(BaseModel):
    text: str


class _MockProvider(BaseProvider):
    """Test double that replaces OpenAIProvider."""

    name = "mock"

    def __init__(self) -> None:
        super().__init__()
        self._complete_mock = MagicMock()
        self._acomplete_mock = AsyncMock()
        self._stream_chunks: list[str] = []
        self._astream_chunks: list[str] = []

    def complete(self, *args, **kwargs):
        return self._complete_mock(*args, **kwargs)

    def acomplete(self, *args, **kwargs):
        return self._acomplete_mock(*args, **kwargs)

    def stream(self, *args, **kwargs):
        yield from self._stream_chunks

    async def astream(self, *args, **kwargs):
        for chunk in self._astream_chunks:
            yield chunk


def _make_agent(**kwargs):
    provider = _MockProvider()
    agent = RivotrilAgent(api_key="test-key", provider=provider, **kwargs)
    return agent, provider


def _make_async_agent(**kwargs):
    provider = _MockProvider()
    agent = RivotrilAgent(api_key="test-key", provider=provider, **kwargs)
    return agent, provider


def test_agent_without_response_model():
    agent, provider = _make_agent()
    provider._complete_mock.return_value = ProviderResponse(content="Hello!")

    result = agent.run("hi")

    assert result == "Hello!"
    assert agent.memory.get_context()[-1]["role"] == "assistant"


def test_agent_with_guardrail_block():
    guardrail = Guardrail(name="safe", disallowed_keywords=["forbidden"])
    agent, _ = _make_agent(guardrails=[guardrail])

    with pytest.raises(GuardrailViolationError):
        agent.run("forbidden word")


def test_agent_with_response_model():
    agent, provider = _make_agent()
    answer = Answer(text="structured")
    provider._complete_mock.return_value = ProviderResponse(structured=answer)

    result = agent.run("question", response_model=Answer)

    assert isinstance(result, Answer)
    assert result.text == "structured"


def test_agent_memory_updates_only_on_success():
    memory = MemoryStore()
    agent, provider = _make_agent(memory=memory)
    provider._complete_mock.return_value = ProviderResponse(content="OK")

    agent.run("hello")

    assert len(memory.get_context()) == 2


@pytest.mark.asyncio
async def test_agent_run_async_without_response_model():
    agent, provider = _make_async_agent()
    provider._acomplete_mock.return_value = ProviderResponse(content="Async hello!")

    result = await agent.run_async("hi")

    assert result == "Async hello!"


@pytest.mark.asyncio
async def test_agent_run_async_with_response_model():
    agent, provider = _make_async_agent()
    answer = Answer(text="async structured")
    provider._acomplete_mock.return_value = ProviderResponse(structured=answer)

    result = await agent.run_async("question", response_model=Answer)

    assert isinstance(result, Answer)
    assert result.text == "async structured"


def test_agent_passes_base_url():
    with patch("llmrivotril.providers.OpenAIProvider.__init__", return_value=None) as mock_init:
        RivotrilAgent(api_key="test-key", base_url="http://localhost:11434/v1")

    mock_init.assert_called_once_with(api_key="test-key", base_url="http://localhost:11434/v1")


def test_agent_passes_timeout_to_llm():
    agent, provider = _make_agent(request_timeout=15.0)
    provider._complete_mock.return_value = ProviderResponse(content="OK")

    agent.run("hi")

    call_args = provider._complete_mock.call_args
    assert call_args.kwargs["timeout"] == 15.0


def test_agent_uses_isolated_metrics():
    from llmrivotril.metrics import MetricsCollector

    metrics = MetricsCollector()
    agent, provider = _make_agent(metrics=metrics)
    provider._complete_mock.return_value = ProviderResponse(content="OK")

    agent.run("hi")

    assert metrics.requests_total == 1


def test_agent_includes_system_prompt():
    agent, provider = _make_agent(system_prompt="You are a helpful assistant.")
    provider._complete_mock.return_value = ProviderResponse(content="OK")

    agent.run("hi")

    call_args = provider._complete_mock.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0] == {"role": "system", "content": "You are a helpful assistant."}
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_agent_accepts_multiple_context_sources():
    agent, provider = _make_agent()
    provider._complete_mock.return_value = ProviderResponse(content="Paris is the capital.")

    result = agent.run(
        "What is the capital?",
        context_sources=[
            "France is in Europe.",
            "The capital of France is Paris.",
        ],
    )
    assert result == "Paris is the capital."


def test_agent_includes_context_sources_in_provider_messages():
    agent, provider = _make_agent()
    provider._complete_mock.return_value = ProviderResponse(content="Paris")

    agent.run("What is the capital?", context_sources="France's capital is Paris.")

    messages = provider._complete_mock.call_args.kwargs["messages"]
    assert any(
        message["role"] == "system"
        and "France's capital is Paris." in message["content"]
        and "retrieved_context" in message["content"]
        for message in messages
    )


def test_agent_uses_env_config(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_MODEL", "gpt-4o")
    monkeypatch.setenv("RIVOTRIL_REQUEST_TIMEOUT", "20")
    monkeypatch.setenv("RIVOTRIL_API_KEY", "env-key")
    monkeypatch.setenv("RIVOTRIL_BASE_URL", "http://env.local:1234/v1")

    agent, _ = _make_agent()

    assert agent.model == "gpt-4o"
    assert agent.request_timeout == 20.0


def test_agent_kwargs_override_env_config(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_MODEL", "gpt-4o")
    monkeypatch.setenv("RIVOTRIL_REQUEST_TIMEOUT", "20")

    agent, _ = _make_agent(model="gpt-4o-mini", request_timeout=5.0)

    assert agent.model == "gpt-4o-mini"
    assert agent.request_timeout == 5.0


def test_agent_falls_back_tokenizer_for_unknown_model():
    agent, _ = _make_agent(model="some-unknown-model")

    assert agent.tokenizer.name == "cl100k_base"


def test_agent_does_not_retry_validation_error():
    agent, provider = _make_agent()
    provider._complete_mock.side_effect = ValidationError.from_exception_data(
        title="Answer",
        line_errors=[{"loc": ("text",), "msg": "field required", "type": "missing"}],
    )

    with pytest.raises(GuardrailViolationError):
        agent.run("question", response_model=Answer)

    assert provider._complete_mock.call_count == 1


def test_agent_uses_openai_provider_by_default():
    with (
        patch("llmrivotril.agent.OpenAIProvider") as mock_provider_class,
    ):
        mock_provider = MagicMock()
        mock_provider_class.return_value = mock_provider
        RivotrilAgent(api_key="test-key")

    mock_provider_class.assert_called_once_with(api_key="test-key", base_url=None)


def test_agent_accepts_provider_string():
    with patch("llmrivotril.agent.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider

        RivotrilAgent(api_key="test-key", provider="anthropic")

        mock_get_provider.assert_called_once_with("anthropic", api_key="test-key", base_url=None)


def test_agent_accepts_provider_instance():
    mock_provider = _MockProvider()
    agent = RivotrilAgent(api_key="test-key", provider=mock_provider)

    assert agent.provider is mock_provider


def test_agent_passes_provider_from_env(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_PROVIDER", "gemini")

    with patch("llmrivotril.agent.get_provider") as mock_get_provider:
        mock_provider = MagicMock()
        mock_get_provider.return_value = mock_provider
        RivotrilAgent(api_key="test-key")

        mock_get_provider.assert_called_once_with("gemini", api_key="test-key", base_url=None)


def test_agent_run_stream_yields_chunks():
    agent, provider = _make_agent()
    provider._stream_chunks = ["Hello", ", ", "world!"]

    chunks = list(agent.run_stream("hi"))

    assert chunks == ["Hello", ", ", "world!"]
    assert agent.memory.get_context()[-1]["content"] == "Hello, world!"


@pytest.mark.asyncio
async def test_agent_run_stream_async_yields_chunks():
    agent, provider = _make_async_agent()
    provider._astream_chunks = ["Async", " ", "stream"]

    chunks = [chunk async for chunk in agent.run_stream_async("hi")]

    assert chunks == ["Async", " ", "stream"]
    assert agent.memory.get_context()[-1]["content"] == "Async stream"


def test_agent_run_stream_redacts_pii_in_memory_and_metrics():
    agent, provider = _make_agent(redact_pii=True)
    provider._stream_chunks = ["Contact ", "john.doe@example.com", " for details"]

    list(agent.run_stream("My email is john.doe@example.com"))

    assert "john.doe@example.com" not in str(agent.memory.get_context())
    last_metric = agent.metrics.get_summary()
    assert "john.doe@example.com" not in str(last_metric)


@pytest.mark.asyncio
async def test_agent_run_stream_async_redacts_pii_in_memory():
    agent, provider = _make_async_agent(redact_pii=True)
    provider._astream_chunks = ["Contact ", "john.doe@example.com", " for details"]

    async for _ in agent.run_stream_async("My email is john.doe@example.com"):
        pass

    assert "john.doe@example.com" not in str(agent.memory.get_context())


def test_agent_run_stream_answers_directly_still_stream_token_by_token():
    agent, provider = _make_agent()
    provider._stream_chunks = ["Hel", "lo!"]

    chunks = list(agent.run_stream("hi", tools=[lambda: None]))

    assert chunks == ["Hel", "lo!"]


def test_agent_run_stream_executes_tool_call_and_yields_final_answer():
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent, provider = _make_agent()
    provider._stream_chunks = [
        _fake_tool_call_chunk(0, tool_id="call_1", name="get_weather", arguments=""),
        _fake_tool_call_chunk(0, arguments='{"city'),
        _fake_tool_call_chunk(0, arguments='": "SP"}'),
    ]
    provider._complete_mock.return_value = ProviderResponse(content="It's sunny in SP.")

    chunks = list(agent.run_stream("What's the weather in SP?", tools=[get_weather]))

    assert chunks == ["It's sunny in SP."]
    assert "".join(chunks) in str(agent.memory.get_context())


@pytest.mark.asyncio
async def test_agent_run_stream_async_executes_tool_call_and_yields_final_answer():
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    agent, provider = _make_async_agent()
    provider._astream_chunks = [
        _fake_tool_call_chunk(0, tool_id="call_1", name="get_weather", arguments='{"city": "SP"}'),
    ]
    provider._acomplete_mock.return_value = ProviderResponse(content="It's sunny in SP.")

    chunks = [chunk async for chunk in agent.run_stream_async("weather?", tools=[get_weather])]

    assert chunks == ["It's sunny in SP."]


def test_agent_run_stream_with_response_model_yields_json_and_sets_result():
    agent, provider = _make_agent()
    provider._stream_chunks = ['{"text": ', '"hello"}']

    stream = agent.run_stream("hi", response_model=Answer)
    chunks = list(stream)

    assert "".join(chunks) == '{"text": "hello"}'
    assert stream.result == Answer(text="hello")


@pytest.mark.asyncio
async def test_agent_run_stream_async_with_response_model_yields_json_and_sets_result():
    agent, provider = _make_async_agent()
    provider._astream_chunks = ['{"text": ', '"hello"}']

    stream = agent.run_stream_async("hi", response_model=Answer)
    chunks = [chunk async for chunk in stream]

    assert "".join(chunks) == '{"text": "hello"}'
    assert stream.result == Answer(text="hello")


def test_agent_run_stream_rejects_response_model_and_tools_together():
    agent, _ = _make_agent()
    with pytest.raises(ValueError, match="does not support combining"):
        agent.run_stream("hi", response_model=Answer, tools=[lambda: None])


def test_agent_run_stream_async_rejects_response_model_and_tools_together():
    agent, _ = _make_async_agent()
    with pytest.raises(ValueError, match="does not support combining"):
        agent.run_stream_async("hi", response_model=Answer, tools=[lambda: None])


def test_agent_run_stream_structured_invalid_json_raises_guardrail_violation():
    agent, provider = _make_agent()
    provider._stream_chunks = ["not valid json"]

    stream = agent.run_stream("hi", response_model=Answer)
    with pytest.raises(GuardrailViolationError):
        list(stream)


def test_agent_run_stream_structured_redacts_pii_before_validating():
    agent, provider = _make_agent(redact_pii=True)
    provider._stream_chunks = ['{"text": "email me at ', 'john@example.com"}']

    stream = agent.run_stream("hi", response_model=Answer)
    list(stream)

    assert "john@example.com" not in stream.result.text
    assert "[REDACTED]" in stream.result.text


def test_agent_run_stream_applies_output_guardrail():
    guardrail = Guardrail(name="short", max_tokens=5)
    agent, provider = _make_agent(guardrails=[guardrail])
    provider._stream_chunks = ["This response is way too long to pass the token limit"]

    with pytest.raises(GuardrailViolationError):
        list(agent.run_stream("hi"))


def test_agent_run_stream_blocks_input_guardrail():
    guardrail = Guardrail(name="safe", disallowed_keywords=["forbidden"])
    agent, provider = _make_agent(guardrails=[guardrail])

    with pytest.raises(GuardrailViolationError):
        list(agent.run_stream("forbidden word"))


def test_agent_enforces_max_prompt_tokens():
    agent, _ = _make_agent(max_prompt_tokens=1)

    with pytest.raises(TokenBudgetExceededError):
        agent.run("this prompt is definitely longer than one token")


def test_agent_enforces_max_session_tokens():
    agent, provider = _make_agent(max_session_tokens=4)
    provider._complete_mock.return_value = ProviderResponse(content="OK")

    agent.run("hi")

    with pytest.raises(TokenBudgetExceededError):
        agent.run("this next prompt will definitely exceed the small session budget")


def test_agent_tracks_session_tokens_across_runs():
    agent, provider = _make_agent()
    provider._complete_mock.return_value = ProviderResponse(content="OK")

    agent.run("hi")
    first_session = agent._session_tokens_used
    assert first_session > 0

    agent.run("hello again")
    assert agent._session_tokens_used > first_session


def test_agent_uses_in_memory_cache():
    from llmrivotril import InMemoryCache

    cache = InMemoryCache()
    agent, provider = _make_agent(cache=cache)
    provider._complete_mock.return_value = ProviderResponse(content="cached")

    result1 = agent.run("hi")
    agent.memory.clear()
    result2 = agent.run("hi")

    assert result1 == "cached"
    assert result2 == "cached"
    assert provider._complete_mock.call_count == 1


def test_agent_uses_disk_cache_via_env(tmp_path, monkeypatch):
    cache_path = tmp_path / "agent-cache.sqlite3"
    monkeypatch.setenv("RIVOTRIL_CACHE_PATH", str(cache_path))

    agent, provider = _make_agent()
    provider._complete_mock.return_value = ProviderResponse(content="disk")

    agent.run("hi")
    assert provider._complete_mock.call_count == 1
    assert cache_path.exists()

    agent2, provider2 = _make_agent()
    provider2._complete_mock.return_value = ProviderResponse(content="other")
    result2 = agent2.run("hi")

    assert result2 == "disk"
    assert provider2._complete_mock.call_count == 0


def test_agent_persists_memory_via_env(tmp_path, monkeypatch):
    memory_path = tmp_path / "agent-memory.json"
    monkeypatch.setenv("RIVOTRIL_MEMORY_PATH", str(memory_path))

    agent, provider = _make_agent()
    provider._complete_mock.return_value = ProviderResponse(content="hi there")
    agent.run("hello")
    assert memory_path.exists()

    agent2, _ = _make_agent()
    assert agent2.memory.get_context() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_agent_tracks_cost_when_enabled():
    from llmrivotril.metrics import MetricsCollector
    from llmrivotril.pricing import register_pricing

    register_pricing("mock", "gpt-4o-mini", 0.15, 0.60)

    metrics = MetricsCollector()
    agent, provider = _make_agent(model="gpt-4o-mini", track_costs=True, metrics=metrics)
    provider._complete_mock.return_value = ProviderResponse(
        content="ok",
        prompt_tokens=1000,
        completion_tokens=500,
    )

    agent.run("hi")

    assert metrics.total_cost_usd is not None
    assert metrics.total_cost_usd > 0


def test_agent_does_not_track_cost_by_default():
    from llmrivotril.metrics import MetricsCollector

    metrics = MetricsCollector()
    agent, provider = _make_agent(model="gpt-4o-mini", metrics=metrics)
    provider._complete_mock.return_value = ProviderResponse(content="ok")

    agent.run("hi")

    assert metrics.total_cost_usd is None


def test_agent_repairs_invalid_structured_response():
    agent, provider = _make_agent(schema_repair_attempts=2)

    class BrokenResponse(BaseModel):
        answer: str

    provider._complete_mock.side_effect = [
        ProviderResponse(content='{"answer": 123}'),
        ProviderResponse(content='{"answer": "good"}'),
    ]

    result = agent.run("question", response_model=BrokenResponse)

    assert isinstance(result, BrokenResponse)
    assert result.answer == "good"
    assert provider._complete_mock.call_count == 2


def test_agent_schema_repair_gives_up():
    agent, provider = _make_agent(schema_repair_attempts=1)

    class BrokenResponse(BaseModel):
        answer: str

    provider._complete_mock.return_value = ProviderResponse(content='{"answer": 123}')

    with pytest.raises(GuardrailViolationError):
        agent.run("question", response_model=BrokenResponse)

    assert provider._complete_mock.call_count == 2  # initial + 1 repair attempt


def test_agent_redacts_pii_in_input_and_output():
    agent, provider = _make_agent(redact_pii=True)
    provider._complete_mock.return_value = ProviderResponse(
        content="User email is john@example.com"
    )

    result = agent.run("My email is john@example.com")

    assert "john@example.com" not in result
    assert "[REDACTED]" in result
    # Memory should also be redacted
    assert "john@example.com" not in str(agent.memory.get_context())


def test_agent_redacts_pii_in_structured_response():
    agent, provider = _make_agent(redact_pii=True)
    provider._complete_mock.return_value = ProviderResponse(
        structured=Answer(text="Contact john@example.com for details")
    )

    result = agent.run("question", response_model=Answer)

    assert "john@example.com" not in result.text
    assert "[REDACTED]" in result.text


def test_agent_does_not_cache_raw_pii():
    from llmrivotril import InMemoryCache

    cache = InMemoryCache()
    agent, provider = _make_agent(redact_pii=True, cache=cache)
    provider._complete_mock.return_value = ProviderResponse(
        content="User email is john@example.com"
    )

    agent.run("hi")

    assert cache._store, "expected the response to have been cached"
    for value, _expires_at in cache._store.values():
        assert "john@example.com" not in value.text
