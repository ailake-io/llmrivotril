from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from llmrivotril import Guardrail, MemoryStore, RivotrilAgent
from llmrivotril.exceptions import GuardrailViolationError
from llmrivotril.providers import BaseProvider, ProviderResponse


class Answer(BaseModel):
    text: str


class _MockProvider(BaseProvider):
    """Test double that replaces OpenAIProvider."""

    name = "mock"

    def __init__(self) -> None:
        super().__init__()
        self._complete_mock = MagicMock()
        self._acomplete_mock = AsyncMock()

    def complete(self, *args, **kwargs):
        return self._complete_mock(*args, **kwargs)

    def acomplete(self, *args, **kwargs):
        return self._acomplete_mock(*args, **kwargs)


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
