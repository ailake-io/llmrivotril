import logging
import os
import secrets
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .metrics import global_metrics
from .resilience import RateLimiter

logger = logging.getLogger("llmrivotril")


def _parse_dashboard_tokens() -> dict[str, str]:
    """Return {token: label} from ``RIVOTRIL_DASHBOARD_TOKENS``/``_TOKEN``.

    ``RIVOTRIL_DASHBOARD_TOKENS`` holds one or more named tokens as
    ``"label1:token1,label2:token2"``, so each client can carry its own
    revocable token and requests can be attributed to a label in logs --
    there's still no per-user permissions, just per-token identity.
    ``RIVOTRIL_DASHBOARD_TOKEN`` (singular) is kept for backward
    compatibility as a single unlabeled token.
    """
    tokens: dict[str, str] = {}
    raw = os.environ.get("RIVOTRIL_DASHBOARD_TOKENS")
    if raw:
        for entry in raw.split(","):
            entry = entry.strip()
            if not entry:
                continue
            label, sep, token = entry.partition(":")
            if not sep:
                label, token = "", label
            token = token.strip()
            if token:
                tokens[token] = label.strip() or token

    single = os.environ.get("RIVOTRIL_DASHBOARD_TOKEN")
    if single:
        tokens.setdefault(single, "default")

    return tokens


DASHBOARD_TOKENS: dict[str, str] = _parse_dashboard_tokens()
# Deprecated: kept for backward compatibility with anything that reads this
# module attribute directly. Prefer DASHBOARD_TOKENS.
DASHBOARD_TOKEN: str | None = os.environ.get("RIVOTRIL_DASHBOARD_TOKEN")

# Per-client-IP rate limiting. This dashboard has a single shared bearer
# token and no per-user accounts -- it's meant for local/trusted-network use,
# not multi-tenant or public exposure (put a real auth/rate-limiting proxy in
# front of it for that). This limiter exists to blunt brute-forcing
# RIVOTRIL_DASHBOARD_TOKEN and casual abuse, not to replace proper access
# control.
_RATE_LIMIT_MAX_CALLS = float(os.environ.get("RIVOTRIL_DASHBOARD_RATE_LIMIT_MAX_CALLS", "60"))
_RATE_LIMIT_PER_SECONDS = float(os.environ.get("RIVOTRIL_DASHBOARD_RATE_LIMIT_PER_SECONDS", "60"))
_MAX_TRACKED_CLIENTS = 10_000

_rate_limiters: OrderedDict[str, RateLimiter] = OrderedDict()
_rate_limiters_lock = Lock()


def _get_rate_limiter(client_id: str) -> RateLimiter:
    with _rate_limiters_lock:
        limiter = _rate_limiters.get(client_id)
        if limiter is not None:
            _rate_limiters.move_to_end(client_id)
            return limiter
        limiter = RateLimiter(max_calls=_RATE_LIMIT_MAX_CALLS, per_seconds=_RATE_LIMIT_PER_SECONDS)
        _rate_limiters[client_id] = limiter
        if len(_rate_limiters) > _MAX_TRACKED_CLIENTS:
            _rate_limiters.popitem(last=False)
        return limiter


def _enforce_rate_limit(request: Request) -> None:
    client_id = request.client.host if request.client else "unknown"
    limiter = _get_rate_limiter(client_id)
    if not limiter.acquire(blocking=False):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
        )


app = FastAPI(title="LLM-Rivotril Local Dashboard")

_STATIC_DIR = Path(__file__).parent / "static"
_TEMPLATE_PATH = Path(__file__).parent / "templates" / "dashboard.html"
HTML_TEMPLATE = _TEMPLATE_PATH.read_text(encoding="utf-8")

app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _verify_dashboard_token(authorization: str | None = Header(None)) -> None:
    """Require a Bearer token matching one of ``DASHBOARD_TOKENS`` when set."""
    if not DASHBOARD_TOKENS:
        return
    if authorization is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    # Compare against every known token with a constant-time check each time
    # (not a dict lookup) so response timing doesn't reveal which, if any,
    # token prefix matched.
    matched_label = None
    for candidate_token, label in DASHBOARD_TOKENS.items():
        if secrets.compare_digest(token, candidate_token):
            matched_label = label
    if matched_label is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )
    logger.info("Dashboard access authenticated as %r", matched_label)


@app.get(
    "/",
    response_class=HTMLResponse,
    dependencies=[Depends(_enforce_rate_limit), Depends(_verify_dashboard_token)],
)
def get_dashboard() -> str:
    return HTML_TEMPLATE


@app.get(
    "/api/metrics",
    dependencies=[Depends(_enforce_rate_limit), Depends(_verify_dashboard_token)],
)
def get_metrics() -> dict[str, Any]:
    return global_metrics.get_summary()


@app.get("/api/health", dependencies=[Depends(_enforce_rate_limit)])
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "version": __version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _format_prometheus_line(name: str, value: float | int, help_text: str, type_: str) -> str:
    return f"# HELP {name} {help_text}\n# TYPE {name} {type_}\n{name} {value}\n"


@app.get(
    "/api/metrics/prometheus",
    dependencies=[Depends(_enforce_rate_limit), Depends(_verify_dashboard_token)],
)
def get_prometheus_metrics() -> str:
    """Return telemetry in Prometheus exposition format."""
    summary = global_metrics.get_summary()
    output = ""
    output += _format_prometheus_line(
        "llmrivotril_requests_total",
        summary["requests_total"],
        "Total number of agent runs",
        "counter",
    )
    output += _format_prometheus_line(
        "llmrivotril_guardrail_blocks_total",
        summary["guardrail_blocks"],
        "Total number of requests blocked by guardrails",
        "counter",
    )
    output += _format_prometheus_line(
        "llmrivotril_hallucinations_detected_total",
        summary["hallucinations_detected"],
        "Total number of responses blocked by grounding verification",
        "counter",
    )
    output += _format_prometheus_line(
        "llmrivotril_errors_total",
        summary["errors_total"],
        "Total number of requests that failed without a policy block",
        "counter",
    )
    output += _format_prometheus_line(
        "llmrivotril_tokens_consumed_total",
        summary["total_tokens_consumed"],
        "Total tokens consumed across all requests",
        "counter",
    )
    output += _format_prometheus_line(
        "llmrivotril_success_rate",
        summary["success_rate"],
        "Percentage of successful requests",
        "gauge",
    )
    output += _format_prometheus_line(
        "llmrivotril_avg_latency_seconds",
        summary["avg_latency"],
        "Average latency in seconds",
        "gauge",
    )
    return output


def run_dashboard(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)
