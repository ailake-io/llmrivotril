import pytest
from fastapi.testclient import TestClient

from llmrivotril import server


@pytest.fixture(autouse=True)
def _reset_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKEN", None)


def test_dashboard_without_auth():
    client = TestClient(server.app)
    response = client.get("/")
    assert response.status_code == 200


def test_metrics_without_auth():
    client = TestClient(server.app)
    response = client.get("/api/metrics")
    assert response.status_code == 200


def test_dashboard_with_required_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKEN", "secret-token")
    client = TestClient(server.app)

    assert client.get("/").status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/", headers={"Authorization": "Bearer secret-token"}).status_code == 200


def test_metrics_with_required_token(monkeypatch):
    monkeypatch.setattr(server, "DASHBOARD_TOKEN", "secret-token")
    client = TestClient(server.app)

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


def test_health_endpoint():
    client = TestClient(server.app)
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"]
    assert data["timestamp"]


def test_dashboard_does_not_use_cdn():
    client = TestClient(server.app)
    response = client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "cdn.tailwindcss.com" not in html
    assert "/static/tailwind.min.js" in html


def test_static_tailwind_file_is_served():
    client = TestClient(server.app)
    response = client.get("/static/tailwind.min.js")
    assert response.status_code == 200
    assert "tailwind" in response.text.lower() or "@tailwind" in response.text
