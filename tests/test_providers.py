import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from llmrivotril.providers import (
    _PROVIDER_REGISTRY,
    AnthropicProvider,
    CohereProvider,
    GeminiProvider,
    OpenAIProvider,
    ProviderResponse,
    available_providers,
    get_provider,
)


class Answer(BaseModel):
    text: str


def _package_installed(module: str) -> bool:
    try:
        __import__(module)
        return True
    except ImportError:
        return False


def test_openai_provider_unstructured():
    provider = OpenAIProvider(api_key="test-key")
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Hello!"

    with (
        patch("openai.OpenAI") as mock_openai_class,
        patch("instructor.from_openai") as mock_from_openai,
    ):
        mock_base = MagicMock()
        mock_base.chat.completions.create.return_value = mock_completion
        mock_openai_class.return_value = mock_base
        mock_from_openai.return_value = MagicMock()

        response = provider.complete(messages=[{"role": "user", "content": "hi"}], model="gpt-4o")

        assert response.text == "Hello!"
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)


def test_openai_provider_structured():
    provider = OpenAIProvider(api_key="test-key")
    answer = Answer(text="structured")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = answer

    with (
        patch("openai.OpenAI") as mock_openai_class,
        patch("instructor.from_openai", return_value=mock_client) as mock_from_openai,
    ):
        response = provider.complete(
            messages=[{"role": "user", "content": "hi"}],
            model="gpt-4o",
            response_model=Answer,
        )

        assert response.structured is answer
        mock_from_openai.assert_called_once()
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)


@pytest.mark.asyncio
async def test_openai_provider_async_unstructured():
    provider = OpenAIProvider(api_key="test-key")
    mock_completion = MagicMock()
    mock_completion.choices[0].message.content = "Async hello!"

    with (
        patch("openai.AsyncOpenAI") as mock_async_openai_class,
        patch("instructor.from_openai") as mock_from_openai,
    ):
        mock_base = MagicMock()
        mock_base.chat.completions.create = AsyncMock(return_value=mock_completion)
        mock_async_openai_class.return_value = mock_base
        mock_from_openai.return_value = MagicMock()

        response = await provider.acomplete(
            messages=[{"role": "user", "content": "hi"}], model="gpt-4o"
        )

        assert response.text == "Async hello!"


@pytest.mark.skipif(not _package_installed("anthropic"), reason="anthropic not installed")
def test_anthropic_provider_unstructured():
    provider = AnthropicProvider(api_key="test-key")
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = "Hello from Claude"
    mock_response.content = [mock_block]

    with patch("anthropic.Anthropic") as mock_anthropic_class:
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic_class.return_value = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="claude-3-opus"
        )

        assert response.text == "Hello from Claude"


@pytest.mark.skipif(not _package_installed("cohere"), reason="cohere not installed")
def test_cohere_provider_unstructured():
    provider = CohereProvider(api_key="test-key")
    mock_response = MagicMock()
    mock_response.text = "Hello from Cohere"

    with patch("cohere.Client") as mock_cohere_class:
        mock_client = MagicMock()
        mock_client.chat.return_value = mock_response
        mock_cohere_class.return_value = mock_client

        response = provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="command-r"
        )

        assert response.text == "Hello from Cohere"


@pytest.mark.skipif(
    not _package_installed("google.generativeai"), reason="google-generativeai not installed"
)
def test_gemini_provider_unstructured():
    provider = GeminiProvider(api_key="test-key")
    mock_response = MagicMock()
    mock_response.text = "Hello from Gemini"

    with patch("google.generativeai.configure") as mock_configure:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_model_class = MagicMock(return_value=mock_model)

        with patch("google.generativeai.GenerativeModel", mock_model_class):
            response = provider.complete(
                messages=[{"role": "user", "content": "hi"}], model="gemini-pro"
            )

            assert response.text == "Hello from Gemini"
            mock_configure.assert_called_once_with(api_key="test-key")


def test_get_provider_resolves_names():
    mock_class = MagicMock()
    original = _PROVIDER_REGISTRY["openai"]
    _PROVIDER_REGISTRY["openai"] = mock_class
    try:
        provider = get_provider("openai", api_key="test-key")
        assert provider is not None
        mock_class.assert_called_once_with(api_key="test-key")
    finally:
        _PROVIDER_REGISTRY["openai"] = original


def test_get_provider_rejects_unknown():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("unknown")


def test_available_providers_returns_list():
    providers = available_providers()
    assert "openai" in providers
    assert "anthropic" in providers
    assert "cohere" in providers
    assert "gemini" in providers


def test_provider_response_text_for_structured():
    answer = Answer(text="hello")
    response = ProviderResponse(structured=answer)
    assert response.text == answer.model_dump_json()


def test_provider_response_text_for_content():
    response = ProviderResponse(content="hello")
    assert response.text == "hello"


def _build_anthropic_response(text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_block = MagicMock()
    mock_block.text = text
    mock_response.content = [mock_block]
    return mock_response


def test_anthropic_provider_unstructured_mocked():
    provider = AnthropicProvider(api_key="test-key")
    fake_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_client.messages.create.return_value = _build_anthropic_response("Hello from Claude")
    fake_anthropic.Anthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        response = provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="claude-3-opus"
        )

    assert response.text == "Hello from Claude"


@pytest.mark.asyncio
async def test_anthropic_provider_async_unstructured_mocked():
    provider = AnthropicProvider(api_key="test-key")
    fake_anthropic = MagicMock()
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_build_anthropic_response("Async Claude"))
    fake_anthropic.AsyncAnthropic.return_value = mock_client

    with patch.dict(sys.modules, {"anthropic": fake_anthropic}):
        response = await provider.acomplete(
            messages=[{"role": "user", "content": "hi"}], model="claude-3-opus"
        )

    assert response.text == "Async Claude"


def test_cohere_provider_unstructured_mocked():
    provider = CohereProvider(api_key="test-key")
    fake_cohere = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello from Cohere"
    mock_client = MagicMock()
    mock_client.chat.return_value = mock_response
    fake_cohere.Client.return_value = mock_client

    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        response = provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="command-r"
        )

    assert response.text == "Hello from Cohere"


@pytest.mark.asyncio
async def test_cohere_provider_async_unstructured_mocked():
    provider = CohereProvider(api_key="test-key")
    fake_cohere = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Async Cohere"
    mock_client = MagicMock()
    mock_client.chat = AsyncMock(return_value=mock_response)
    fake_cohere.AsyncClient.return_value = mock_client

    with patch.dict(sys.modules, {"cohere": fake_cohere}):
        response = await provider.acomplete(
            messages=[{"role": "user", "content": "hi"}], model="command-r"
        )

    assert response.text == "Async Cohere"


def test_gemini_provider_unstructured_mocked():
    provider = GeminiProvider(api_key="test-key")
    fake_genai = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Hello from Gemini"
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    fake_genai.GenerativeModel.return_value = mock_model

    with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
        response = provider.complete(
            messages=[{"role": "user", "content": "hi"}], model="gemini-pro"
        )

    assert response.text == "Hello from Gemini"
    fake_genai.configure.assert_called_once_with(api_key="test-key")


@pytest.mark.asyncio
async def test_gemini_provider_async_unstructured_mocked():
    provider = GeminiProvider(api_key="test-key")
    fake_genai = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "Async Gemini"
    mock_model = MagicMock()
    mock_model.generate_content_async = AsyncMock(return_value=mock_response)
    fake_genai.GenerativeModel.return_value = mock_model

    with patch.dict(sys.modules, {"google.generativeai": fake_genai}):
        response = await provider.acomplete(
            messages=[{"role": "user", "content": "hi"}], model="gemini-pro"
        )

    assert response.text == "Async Gemini"
