import threading

from llmrivotril.metrics import MetricsCollector


def test_empty_summary():
    metrics = MetricsCollector()
    summary = metrics.get_summary()
    assert summary["requests_total"] == 0
    assert summary["success_rate"] == 100.0


def test_log_execution_counts():
    metrics = MetricsCollector()
    metrics.log_execution("prompt", "response", tokens=10, latency=0.5)
    metrics.log_execution("prompt2", "response2", tokens=5, latency=0.2, guardrail_blocked=True)

    summary = metrics.get_summary()
    assert summary["requests_total"] == 2
    assert summary["guardrail_blocks"] == 1
    assert summary["errors_total"] == 0
    assert summary["total_tokens_consumed"] == 15
    assert summary["success_rate"] == 50.0


def test_success_rate_never_negative():
    metrics = MetricsCollector()
    metrics.log_execution(
        "p", "r", tokens=1, latency=0.1, guardrail_blocked=True, hallucination_blocked=True
    )
    summary = metrics.get_summary()
    assert summary["success_rate"] == 0.0


def test_errors_reduce_success_rate_without_being_policy_blocks():
    metrics = MetricsCollector()
    metrics.log_execution("p", "", tokens=1, latency=0.1, error="provider failed")

    summary = metrics.get_summary()
    assert summary["errors_total"] == 1
    assert summary["guardrail_blocks"] == 0
    assert summary["hallucinations_detected"] == 0
    assert summary["success_rate"] == 0.0


def test_log_history_cap():
    metrics = MetricsCollector()
    for i in range(110):
        metrics.log_execution(f"prompt {i}", "response", tokens=1, latency=0.01)
    summary = metrics.get_summary()
    assert len(summary["logs"]) == 100


def test_reset():
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=1, latency=0.1)
    metrics.reset()
    summary = metrics.get_summary()
    assert summary["requests_total"] == 0
    assert summary["errors_total"] == 0
    assert summary["total_tokens_consumed"] == 0
    assert summary["logs"] == []


def test_cache_hit_tracked_in_summary_and_log():
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=1, latency=0.1, cache_hit=False)
    metrics.log_execution("p", "r", tokens=1, latency=0.0, cache_hit=True)

    summary = metrics.get_summary()
    assert summary["cache_hits"] == 1
    assert summary["cache_hit_rate"] == 50.0
    assert summary["logs"][0]["cache_hit"] is True
    assert summary["logs"][1]["cache_hit"] is False


def test_cache_hit_defaults_to_false():
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=1, latency=0.1)

    summary = metrics.get_summary()
    assert summary["cache_hits"] == 0
    assert summary["cache_hit_rate"] == 0.0
    assert summary["logs"][0]["cache_hit"] is False


def test_cache_hits_survive_reset_and_round_trip():
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=1, latency=0.1, cache_hit=True)
    metrics.reset()
    assert metrics.get_summary()["cache_hits"] == 0

    metrics.log_execution("p", "r", tokens=1, latency=0.1, cache_hit=True)
    loaded = MetricsCollector()
    loaded.from_dict(metrics.to_dict())
    assert loaded.cache_hits == 1


def test_thread_safety():
    metrics = MetricsCollector()

    def worker() -> None:
        for _ in range(100):
            metrics.log_execution("p", "r", tokens=1, latency=0.001)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert metrics.requests_total == 1000


def test_metrics_save_and_load(tmp_path):
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=10, latency=0.5, guardrail_blocked=True)

    path = tmp_path / "metrics.json"
    metrics.save_to_json(path)

    loaded = MetricsCollector()
    loaded.load_from_json(path)

    assert loaded.requests_total == 1
    assert loaded.guardrail_blocks == 1
    assert loaded.errors_total == 0
    assert loaded.total_tokens_consumed == 10
    assert len(loaded.logs) == 1


def test_metrics_to_dict_and_from_dict():
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=5, latency=0.2)

    snapshot = metrics.to_dict()
    loaded = MetricsCollector()
    loaded.from_dict(snapshot)

    assert loaded.requests_total == 1
    assert loaded.total_tokens_consumed == 5
