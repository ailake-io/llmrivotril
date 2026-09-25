import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .metrics import global_metrics

DASHBOARD_TOKEN: str | None = os.environ.get("RIVOTRIL_DASHBOARD_TOKEN")

app = FastAPI(title="LLM-Rivotril Local Dashboard")

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
HTML_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _verify_dashboard_token(authorization: str | None = Header(None)) -> None:
    """Require a Bearer token when ``RIVOTRIL_DASHBOARD_TOKEN`` is set."""
    if not DASHBOARD_TOKEN:
        return
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token != DASHBOARD_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


@app.get("/", response_class=HTMLResponse, dependencies=[Depends(_verify_dashboard_token)])
def get_dashboard() -> str:
    return HTML_TEMPLATE


@app.get("/api/metrics", dependencies=[Depends(_verify_dashboard_token)])
def get_metrics() -> dict[str, Any]:
    return global_metrics.get_summary()


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
