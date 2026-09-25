import json

from llmrivotril.metrics import MetricsCollector


def test_metrics_auto_save_via_constructor(tmp_path):
    path = tmp_path / "metrics.json"
    metrics = MetricsCollector(auto_save_path=path)

    metrics.log_execution("p", "r", tokens=10, latency=0.5)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["requests_total"] == 1
    assert data["total_tokens_consumed"] == 10


def test_metrics_auto_save_via_env_var(tmp_path, monkeypatch):
    path = tmp_path / "metrics.json"
    monkeypatch.setenv("RIVOTRIL_METRICS_PATH", str(path))
    metrics = MetricsCollector()

    metrics.log_execution("p", "r", tokens=5, latency=0.2)

    assert path.exists()
    data = json.loads(path.read_text())
    assert data["requests_total"] == 1
    assert data["total_tokens_consumed"] == 5


def test_metrics_constructor_path_overrides_env_var(tmp_path, monkeypatch):
    env_path = tmp_path / "env_metrics.json"
    ctor_path = tmp_path / "ctor_metrics.json"
    monkeypatch.setenv("RIVOTRIL_METRICS_PATH", str(env_path))
    metrics = MetricsCollector(auto_save_path=ctor_path)

    metrics.log_execution("p", "r", tokens=3, latency=0.1)

    assert ctor_path.exists()
    assert not env_path.exists()


def test_metrics_load_metrics_alias(tmp_path):
    path = tmp_path / "metrics.json"
    metrics = MetricsCollector()
    metrics.log_execution("p", "r", tokens=7, latency=0.3)
    metrics.save_to_json(path)

    loaded = MetricsCollector()
    loaded.load_metrics(path)

    assert loaded.requests_total == 1
    assert loaded.total_tokens_consumed == 7


def test_metrics_auto_save_creates_parent_directories(tmp_path):
    path = tmp_path / "nested" / "dir" / "metrics.json"
    metrics = MetricsCollector(auto_save_path=path)

    metrics.log_execution("p", "r", tokens=1, latency=0.01)

    assert path.exists()
