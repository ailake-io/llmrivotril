import pytest

from llmrivotril.pricing import estimate_cost, get_model_pricing, register_pricing


def test_estimate_cost_for_known_openai_model():
    cost = estimate_cost("openai", "gpt-4o-mini", 1_000_000, 1_000_000)
    assert cost is not None
    assert cost > 0


def test_estimate_cost_for_prefix_match():
    cost = estimate_cost("openai", "gpt-4o-2024-08-06", 1_000_000, 0)
    assert cost is not None
    assert cost > 0


def test_estimate_cost_returns_none_for_unknown_provider():
    assert estimate_cost("unknown", "model", 1_000, 1_000) is None


def test_estimate_cost_returns_none_for_unknown_model():
    assert estimate_cost("openai", "unknown-model", 1_000, 1_000) is None


def test_register_pricing():
    register_pricing("custom", "custom-model", 1.0, 2.0)
    cost = estimate_cost("custom", "custom-model", 1_000_000, 1_000_000)
    assert cost == pytest.approx(3.0)


def test_get_model_pricing():
    prices = get_model_pricing("openai", "gpt-4o-mini")
    assert prices is not None
    assert "input" in prices
    assert "output" in prices
