import asyncio

import httpx
import pytest

from llmrivotril import server


class _SyncASGIClient:
    """Small sync facade over httpx's async ASGI transport.

    Starlette's ``TestClient`` relies on AnyIO's blocking portal. That portal
    can hang under Python 3.13 in the managed test environment, while the
    async transport exercises the same application without an extra portal
    thread.
    """

    def __init__(self, app) -> None:
        self._app = app

    def get(self, url: str, **kwargs):
        async def request():
            transport = httpx.ASGITransport(app=self._app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                return await client.get(url, **kwargs)

        return asyncio.run(request())


@pytest.fixture(autouse=True)
def _reset_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKEN", None)
    monkeypatch.setattr(server, "DASHBOARD_TOKENS", {})


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    """The rate limiter is process-global state, shared across every test in
    this file; without resetting it, exhausting the budget in one test would
    make unrelated later tests fail with 429."""
    server._rate_limiters.clear()
    yield
    server._rate_limiters.clear()


def test_dashboard_without_auth():
    client = _SyncASGIClient(server.app)
    response = client.get("/")
    assert response.status_code == 200


def test_metrics_without_auth():
    client = _SyncASGIClient(server.app)
    response = client.get("/api/metrics")
    assert response.status_code == 200


def test_dashboard_with_required_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKENS", {"secret-token": "default"})
    client = _SyncASGIClient(server.app)

    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_metrics_with_required_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKENS", {"secret-token": "default"})
    client = _SyncASGIClient(server.app)

    assert client.get("/api/metrics").status_code == 401
    assert (
        client.get("/api/metrics", headers={"Authorization": "Bearer secret-token"}).status_code
        == 200
    )


def test_env_var_populates_token(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_DASHBOARD_TOKEN", "env-token")
    import importlib

    importlib.reload(server)
    assert server.DASHBOARD_TOKEN == "env-token"
    assert server.DASHBOARD_TOKENS == {"env-token": "default"}


def test_named_tokens_env_var_accepts_multiple_labeled_tokens(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_DASHBOARD_TOKENS", "alice:tok-a, bob:tok-b")
    import importlib

    importlib.reload(server)
    assert server.DASHBOARD_TOKENS == {"tok-a": "alice", "tok-b": "bob"}

    client = _SyncASGIClient(server.app)
    assert client.get("/", headers={"Authorization": "Bearer tok-a"}).status_code == 200
    assert client.get("/", headers={"Authorization": "Bearer tok-b"}).status_code == 200
    assert client.get("/", headers={"Authorization": "Bearer tok-c"}).status_code == 401


def test_named_tokens_env_var_accepts_unlabeled_entries(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_DASHBOARD_TOKENS", "just-a-token")
    import importlib

    importlib.reload(server)
    assert server.DASHBOARD_TOKENS == {"just-a-token": "just-a-token"}


def test_health_endpoint():
    client = _SyncASGIClient(server.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"]
    assert data["timestamp"]


def test_dashboard_does_not_use_cdn():
    client = _SyncASGIClient(server.app)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "cdn.tailwindcss.com" not in html
    assert "/static/tailwind.min.js" in html
    assert "innerHTML" not in html
    assert "textContent" in html


def test_parse_dashboard_tokens_empty_when_unset(monkeypatch):
    monkeypatch.delenv("RIVOTRIL_DASHBOARD_TOKEN", raising=False)
    monkeypatch.delenv("RIVOTRIL_DASHBOARD_TOKENS", raising=False)
    assert server._parse_dashboard_tokens() == {}


def test_parse_dashboard_tokens_combines_both_env_vars(monkeypatch):
    monkeypatch.setenv("RIVOTRIL_DASHBOARD_TOKENS", "alice:tok-a")
    monkeypatch.setenv("RIVOTRIL_DASHBOARD_TOKEN", "legacy-tok")
    assert server._parse_dashboard_tokens() == {"tok-a": "alice", "legacy-tok": "default"}


def test_dashboard_access_authenticated_as_matching_label(monkeypatch, caplog):
    monkeypatch.setattr(server, "DASHBOARD_TOKENS", {"tok-a": "alice"})
    client = _SyncASGIClient(server.app)

    with caplog.at_level("INFO", logger="llmrivotril"):
        client.get("/", headers={"Authorization": "Bearer tok-a"})

    assert any("alice" in record.message for record in caplog.records)


def test_static_tailwind_file_is_served():
    client = _SyncASGIClient(server.app)
    response = client.get("/static/tailwind.min.js")
    assert response.status_code == 200
    assert "tailwind" in response.text.lower() or "@tailwind" in response.text


def test_prometheus_metrics_endpoint():
    client = _SyncASGIClient(server.app)
    response = client.get("/api/metrics/prometheus")
    assert response.status_code == 200
    text = response.text
    assert "llmrivotril_requests_total" in text
    assert "llmrivotril_guardrail_blocks_total" in text
    assert "llmrivotril_hallucinations_detected_total" in text
    assert "llmrivotril_errors_total" in text
    assert "llmrivotril_tokens_consumed_total" in text
    assert "llmrivotril_success_rate" in text
    assert "llmrivotril_avg_latency_seconds" in text


def test_rate_limit_blocks_after_exceeding_budget(monkeypatch):
    monkeypatch.setattr(server, "_RATE_LIMIT_MAX_CALLS", 2.0)
    monkeypatch.setattr(server, "_RATE_LIMIT_PER_SECONDS", 60.0)
    client = _SyncASGIClient(server.app)

    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/health").status_code == 429


def test_rate_limit_is_tracked_per_client(monkeypatch):
    monkeypatch.setattr(server, "_RATE_LIMIT_MAX_CALLS", 1.0)
    monkeypatch.setattr(server, "_RATE_LIMIT_PER_SECONDS", 60.0)

    server._get_rate_limiter("1.2.3.4")
    assert len(server._rate_limiters) == 1
    server._get_rate_limiter("5.6.7.8")
    assert len(server._rate_limiters) == 2


def test_prometheus_metrics_requires_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKENS", {"secret-token": "default"})
    client = _SyncASGIClient(server.app)

    assert client.get("/api/metrics/prometheus").status_code == 401
    assert (
        client.get(
            "/api/metrics/prometheus", headers={"Authorization": "Bearer secret-token"}
        ).status_code
        == 200
    )
