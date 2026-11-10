"""Deterministic offline comparison; synthetic savings are not provider billing."""

import argparse
import json
import statistics
import time
import tracemalloc
from typing import Any

from llmrivotril import Guardrail, InMemoryCache, RivotrilAgent
from llmrivotril.exceptions import GuardrailViolationError
from llmrivotril.metrics import MetricsCollector
from llmrivotril.providers import BaseProvider, ProviderResponse


class LocalProvider(BaseProvider):
    name = "local-benchmark"

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def complete(self, **kwargs: Any) -> ProviderResponse:
        self.calls += 1
        time.sleep(0.001)
        return ProviderResponse(content="Paris", prompt_tokens=12, completion_tokens=2)

    async def acomplete(self, **kwargs: Any) -> ProviderResponse:
        return self.complete(**kwargs)


def compare(iterations: int = 100) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    rows = []
    # Warm tokenization before measuring allocations or latency.
    warm = RivotrilAgent(provider=LocalProvider(), metrics=MetricsCollector(max_logs=0))
    warm._count_tokens("What is the capital of France?")
    for scenario in ("direct", "agent", "cache", "guardrail"):
        provider = LocalProvider()
        metrics = MetricsCollector(max_logs=0, include_content=False)
        agent = RivotrilAgent(
            provider=provider,
            metrics=metrics,
            cache=InMemoryCache() if scenario == "cache" else None,
            guardrails=[Guardrail(name="local", disallowed_keywords=["blocked"])],
        )
        durations, answers, blocked = [], [], 0
        tracemalloc.start()
        for _ in range(iterations):
            agent.memory.clear()
            start = time.perf_counter()
            try:
                if scenario == "direct":
                    answers.append(provider.complete().text)
                else:
                    answers.append(
                        agent.run(
                            "blocked"
                            if scenario == "guardrail"
                            else "What is the capital of France?"
                        )
                    )
            except GuardrailViolationError:
                blocked += 1
            durations.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        rows.append(
            {
                "scenario": scenario,
                "requests": iterations,
                "provider_calls": provider.calls,
                "synthetic_provider_tokens": provider.calls * 14,
                "blocked": blocked,
                "correct_answers": answers.count("Paris"),
                "latency_median_ms": round(statistics.median(durations) * 1000, 3),
                "peak_traced_bytes": peak,
            }
        )
    return {"synthetic": True, "billing_or_real_quality_claim": False, "results": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    print(json.dumps(compare(args.iterations), indent=2))
