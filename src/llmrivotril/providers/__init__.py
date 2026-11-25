"""Multi-provider adapters for ``RivotrilAgent``.

Each provider exposes a unified ``complete`` / ``acomplete`` interface so the
agent can use OpenAI-compatible APIs, Anthropic, Cohere, or Gemini without
coupling to any specific SDK.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .anthropic import AnthropicProvider
from .base import BaseProvider, ProviderResponse
from .bedrock import BedrockProvider
from .cohere import CohereProvider
from .gemini import GeminiProvider
from .openai import AzureOpenAIProvider, OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "AzureOpenAIProvider",
    "BaseProvider",
    "BedrockProvider",
    "CohereProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "ProviderResponse",
    "available_providers",
    "get_provider",
]

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
