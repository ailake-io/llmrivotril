"""Cost estimation for LLM providers.

Prices are stored per 1M tokens and are best-effort approximations. They can
change at any time by the provider, so these estimates are useful for tracking
and budgeting rather than exact billing.
"""

from __future__ import annotations

from typing import Any

# Prices in USD per 1,000,000 tokens.
# "input"  = prompt tokens
# "output" = completion tokens
_PRICES: dict[str, dict[str, dict[str, float]]] = {
    "openai": {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
        "gpt-4": {"input": 30.00, "output": 60.00},
        "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    },
    "anthropic": {
        "claude-3-opus": {"input": 15.00, "output": 75.00},
        "claude-3-sonnet": {"input": 3.00, "output": 15.00},
        "claude-3-haiku": {"input": 0.25, "output": 1.25},
        "claude-3-5-sonnet": {"input": 3.00, "output": 15.00},
    },
    "cohere": {
        "command-r": {"input": 0.50, "output": 1.50},
        "command-r-plus": {"input": 3.00, "output": 15.00},
    },
    "gemini": {
        "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        "gemini-1.5-flash": {"input": 0.35, "output": 1.05},
    },
}


def _normalize_model(model: str) -> str:
    """Strip common date/version suffixes to improve match rates."""
    model = model.lower().strip()
    for suffix in ("-latest", "-preview", ":", "-"):
        if model.endswith(suffix.rstrip("-")):
            continue
    return model


def get_model_pricing(provider: str, model: str) -> dict[str, float] | None:
    """Return pricing for a specific provider/model, or ``None`` if unknown."""
    provider_prices = _PRICES.get(provider.lower())
    if provider_prices is None:
        return None

    model_lower = model.lower()
    if model_lower in provider_prices:
        return provider_prices[model_lower]

    # Try prefix matches (e.g. "gpt-4o-2024-08-06" -> "gpt-4o")
    for known_model, prices in provider_prices.items():
        if model_lower.startswith(known_model):
            return prices

    return None


def estimate_cost(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
) -> float | None:
    """Estimate the cost in USD for a completion.

    Returns ``None`` when pricing for the provider/model is not known.
    """
    prices = get_model_pricing(provider, model)
    if prices is None:
        return None

    input_cost = (prompt_tokens / 1_000_000) * prices["input"]
    output_cost = (completion_tokens / 1_000_000) * prices["output"]
    return round(input_cost + output_cost, 10)


def register_pricing(provider: str, model: str, input_price: float, output_price: float) -> None:
    """Register or override pricing for a provider/model at runtime."""
    provider_lower = provider.lower()
    if provider_lower not in _PRICES:
        _PRICES[provider_lower] = {}
    _PRICES[provider_lower][model.lower()] = {
        "input": input_price,
        "output": output_price,
    }


def list_supported_models(provider: str | None = None) -> dict[str, Any]:
    """Return a dict of supported provider/model pricing."""
    if provider is None:
        return {k: dict(v) for k, v in _PRICES.items()}
    return dict(_PRICES.get(provider.lower(), {}))
