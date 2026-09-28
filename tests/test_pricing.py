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


def test_azure_openai_falls_back_to_openai_pricing_by_model_name():
    # Only works when the deployment name matches/prefixes a known OpenAI
    # model name -- azure_openai has no way to know what a custom deployment
    # name maps to otherwise.
    assert get_model_pricing("azure_openai", "gpt-4o-mini") == get_model_pricing(
        "openai", "gpt-4o-mini"
    )


def test_azure_openai_unknown_deployment_name_returns_none():
    assert get_model_pricing("azure_openai", "my-custom-deployment") is None


def test_bedrock_anthropic_model_id_resolves_via_vendor_prefix():
    assert get_model_pricing(
        "bedrock", "anthropic.claude-3-5-sonnet-20241022-v2:0"
    ) == get_model_pricing("anthropic", "claude-3-5-sonnet")


def test_bedrock_unknown_vendor_prefix_returns_none():
    assert get_model_pricing("bedrock", "meta.llama3-70b-instruct-v1:0") is None
