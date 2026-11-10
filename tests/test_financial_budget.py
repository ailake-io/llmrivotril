from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import httpx
import pytest

from llmrivotril import CostBudgetExceededError, RivotrilAgent
from llmrivotril.budget import SessionBudget
from llmrivotril.metrics import MetricsCollector
from llmrivotril.providers import OpenAIProvider


def test_reservations_are_atomic():
    budget = SessionBudget(0.001, 1, 1)

    def reserve(_):
        try:
            return budget.reserve(400, 100)
        except CostBudgetExceededError:
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(8)))
    assert sum(item is not None for item in results) == 2
    assert budget.summary()["reserved_usd"] == 0.001


def test_failed_call_keeps_reservation_and_reported_overrun_is_visible():
    budget = SessionBudget(0.001, 1, 1)
    amount = budget.reserve(100, 100)
    budget.settle(amount, None)
    assert budget.summary()["committed_usd"] == 0.0002
    amount = budget.reserve(100, 100)
    budget.settle(amount, Decimal("0.002"))
    assert budget.summary()["overrun"] is True
    with pytest.raises(CostBudgetExceededError):
        budget.reserve(1, 1)


@pytest.mark.parametrize(
    "maximum,input_price,output_price",
    [
        (float("nan"), 1, 1),
        (1, float("inf"), 1),
        (1, -1, 1),
        (0, 1, 1),
    ],
)
def test_invalid_financial_values(maximum, input_price, output_price):
    with pytest.raises(ValueError):
        SessionBudget(maximum, input_price, output_price)


def test_budget_stops_call_before_http_and_disables_hidden_sdk_retries(monkeypatch):
    class Tokenizer:
        def encode(self, text):
            return list(text)

    monkeypatch.setattr("llmrivotril.agent._get_tokenizer", lambda model: Tokenizer())
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "local failure"}})

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        provider = OpenAIProvider(
            api_key="dummy", base_url="https://local.invalid/v1", http_client=client
        )
        metrics = MetricsCollector(include_content=False)
        agent = RivotrilAgent(
            provider=provider,
            max_output_tokens=10,
            max_session_cost_usd=0.0006,
            input_price_per_million=1,
            output_price_per_million=1,
            retry_max_attempts=3,
            retry_min_wait=0,
            retry_max_wait=0,
            metrics=metrics,
        )
        with pytest.raises(CostBudgetExceededError):
            agent.run("hello")
        assert len(requests) == 1
        assert provider._get_base_client().max_retries == 0
        usage = metrics.get_summary()["provider_usage"]
        assert usage["calls"] == usage["failures"] == 1
        assert agent.session_budget.summary()["reserved_usd"] == 0


def test_provider_usage_roundtrip_and_reset():
    metrics = MetricsCollector()
    metrics.log_provider_usage(5, 3, False, False, 0.01)
    loaded = MetricsCollector()
    loaded.from_dict(metrics.to_dict())
    assert loaded.get_summary()["provider_usage"]["calls"] == 1
    loaded.reset()
    assert loaded.get_summary()["provider_usage"]["calls"] == 0


def test_reported_usage_settles_reserve_and_cache_is_free(monkeypatch):
    from llmrivotril import InMemoryCache

    class Tokenizer:
        def encode(self, text):
            return list(text)

    monkeypatch.setattr("llmrivotril.agent._get_tokenizer", lambda model: Tokenizer())
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "local",
                "object": "chat.completion",
                "created": 0,
                "model": "local",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "OK"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as client:
        metrics = MetricsCollector()
        agent = RivotrilAgent(
            provider=OpenAIProvider(
                api_key="dummy", base_url="https://local.invalid/v1", http_client=client
            ),
            max_output_tokens=10,
            max_session_cost_usd=0.01,
            input_price_per_million=1,
            output_price_per_million=1,
            cache=InMemoryCache(),
            metrics=metrics,
        )
        assert agent.run("hello") == "OK"
        assert agent.session_budget.summary()["committed_usd"] == 0.000004
        agent.memory.clear()
        assert agent.run("hello") == "OK"
        assert len(requests) == 1
        assert agent.session_budget.summary()["committed_usd"] == 0.000004
        assert agent._summarize_history("history") == "OK"
        assert agent.session_budget.summary()["committed_usd"] == 0.000008
        usage = metrics.get_summary()["provider_usage"]
        assert usage["calls"] == 2
        assert usage["input_tokens"] == 6 and usage["output_tokens"] == 2
        assert usage["estimated_calls"] == 0
