import json
import os
import tempfile
import time
from pathlib import Path
from threading import Lock
from typing import Any


class MetricsCollector:
    """Thread-safe telemetry store.

    Tracks requests, tokens, guardrail blocks, hallucinations, and cache hits.
    """

    def __init__(self, auto_save_path: str | Path | None = None) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.guardrail_blocks = 0
        self.hallucinations_detected = 0
        self.errors_total = 0
        self.cache_hits = 0
        self.total_tokens_consumed = 0
        self.total_cost_usd: float | None = None
        self.latencies: list[float] = []
        self.logs: list[dict[str, Any]] = []
        self.auto_save_path = auto_save_path

    def log_execution(
        self,
        prompt: str,
        response: str,
        tokens: int,
        latency: float,
        guardrail_blocked: bool = False,
        hallucination_blocked: bool = False,
        error: str | None = None,
        cost_usd: float | None = None,
        cache_hit: bool = False,
    ) -> None:
        with self._lock:
            self.requests_total += 1
            self.total_tokens_consumed += tokens
            self.latencies.append(latency)

            if cost_usd is not None:
                if self.total_cost_usd is None:
                    self.total_cost_usd = 0.0
                self.total_cost_usd += cost_usd

            if guardrail_blocked:
                self.guardrail_blocks += 1
            if hallucination_blocked:
                self.hallucinations_detected += 1
            if error is not None and not guardrail_blocked and not hallucination_blocked:
                self.errors_total += 1
            if cache_hit:
                self.cache_hits += 1

            self.logs.insert(
                0,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "tokens": tokens,
                    "latency_sec": round(latency, 3),
                    "guardrail_blocked": guardrail_blocked,
                    "hallucination_blocked": hallucination_blocked,
                    "error": error,
                    "cost_usd": cost_usd,
                    "cache_hit": cache_hit,
                },
            )
            # Keep history capped at 100 items
            if len(self.logs) > 100:
                self.logs.pop()

        self._maybe_auto_save()

    def reset(self) -> None:
        with self._lock:
            self.requests_total = 0
            self.guardrail_blocks = 0
            self.hallucinations_detected = 0
            self.errors_total = 0
            self.cache_hits = 0
            self.total_tokens_consumed = 0
            self.total_cost_usd = None
            self.latencies.clear()
            self.logs.clear()

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            failed = self.guardrail_blocks + self.hallucinations_detected + self.errors_total
            if self.requests_total > 0:
                success_rate = max(0.0, (self.requests_total - failed) / self.requests_total * 100)
                cache_hit_rate = self.cache_hits / self.requests_total * 100
            else:
                success_rate = 100.0
                cache_hit_rate = 0.0
            return {
                "requests_total": self.requests_total,
                "guardrail_blocks": self.guardrail_blocks,
                "hallucinations_detected": self.hallucinations_detected,
                "errors_total": self.errors_total,
                "cache_hits": self.cache_hits,
                "cache_hit_rate": round(cache_hit_rate, 2),
                "total_tokens_consumed": self.total_tokens_consumed,
                "total_cost_usd": self.total_cost_usd,
                "avg_latency": round(avg_latency, 3),
                "success_rate": round(success_rate, 2),
                "logs": self.logs,
            }

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the collector state."""
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "guardrail_blocks": self.guardrail_blocks,
                "hallucinations_detected": self.hallucinations_detected,
                "cache_hits": self.cache_hits,
                "total_tokens_consumed": self.total_tokens_consumed,
                "total_cost_usd": self.total_cost_usd,
                "latencies": self.latencies.copy(),
                "logs": self.logs.copy(),
            }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore collector state from a dictionary."""
        with self._lock:
            self.requests_total = data.get("requests_total", 0)
            self.guardrail_blocks = data.get("guardrail_blocks", 0)
            self.hallucinations_detected = data.get("hallucinations_detected", 0)
            self.errors_total = data.get("errors_total", 0)
            self.cache_hits = data.get("cache_hits", 0)
            self.total_tokens_consumed = data.get("total_tokens_consumed", 0)
            self.total_cost_usd = data.get("total_cost_usd", None)
            self.latencies = data.get("latencies", []).copy()
            self.logs = data.get("logs", []).copy()

    def _maybe_auto_save(self) -> None:
        """Persist state if an automatic save path is configured."""
        path = self.auto_save_path
        if path is None:
            env_path = os.environ.get("RIVOTRIL_METRICS_PATH")
            if env_path:
                path = Path(env_path)
        if path:
            self.save_to_json(path)

    def save_to_json(self, path: str | Path) -> None:
        """Persist the current state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self.to_dict(), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)

    def load_from_json(self, path: str | Path) -> None:
        """Restore state from a JSON file created by ``save_to_json``."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.from_dict(data)

    def load_metrics(self, path: str | Path) -> None:
        """Restore state from a JSON file (alias for ``load_from_json``)."""
        self.load_from_json(path)


# Global singleton metrics instance
global_metrics = MetricsCollector()
