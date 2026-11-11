import json
import os
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any


class MetricsCollector:
    """Thread-safe telemetry store.

    Tracks requests, tokens, guardrail blocks, hallucinations, and cache hits.
    """

    def __init__(
        self,
        auto_save_path: str | Path | None = None,
        max_logs: int = 100,
        include_content: bool = True,
    ) -> None:
        if max_logs < 0:
            raise ValueError("max_logs must be non-negative")
        self._lock = Lock()
        self.requests_total = 0
        self.guardrail_blocks = 0
        self.hallucinations_detected = 0
        self.errors_total = 0
        self.cache_hits = 0
        self.total_tokens_consumed = 0
        self.total_cost_usd: float | None = None
        self.latencies: list[float] = []
        self._latency_total = 0.0
        self._latency_count = 0
        self.logs: list[dict[str, Any]] = []
        self.auto_save_path = auto_save_path
        self.max_logs = max_logs
        self.include_content = include_content
        self.provider_usage: dict[str, float | int] = {
            "calls": 0,
            "failures": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "estimated_calls": 0,
            "unpriced_calls": 0,
            "cost_usd": 0.0,
        }

    def log_provider_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        failed: bool,
        estimated: bool,
        cost_usd: float | None,
    ) -> None:
        """Track every visible SDK invocation separately from user-facing runs."""
        with self._lock:
            for key, value in {
                "calls": 1,
                "failures": int(failed),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "estimated_calls": int(estimated),
                "unpriced_calls": int(cost_usd is None),
                "cost_usd": cost_usd or 0.0,
            }.items():
                self.provider_usage[key] += value
        self._maybe_auto_save()

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
        if not self.include_content:
            prompt = response = "[omitted]"
            error = "[omitted]" if error is not None else None
        with self._lock:
            self.requests_total += 1
            self.total_tokens_consumed += tokens
            self.latencies.append(latency)
            del self.latencies[:-1000]
            self._latency_total += latency
            self._latency_count += 1

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
            # Keep history bounded so long-running processes do not grow
            # indefinitely. A zero limit disables per-request log retention.
            if len(self.logs) > self.max_logs:
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
            self._latency_total = 0.0
            self._latency_count = 0
            self.logs.clear()
            self.provider_usage = dict.fromkeys(self.provider_usage, 0)

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = self._latency_total / self._latency_count if self._latency_count else 0.0
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
                "provider_usage": self.provider_usage.copy(),
                "success_rate": round(success_rate, 2),
                "logs": deepcopy(self.logs),
            }

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot of the collector state."""
        with self._lock:
            return {
                "requests_total": self.requests_total,
                "errors_total": self.errors_total,
                "guardrail_blocks": self.guardrail_blocks,
                "hallucinations_detected": self.hallucinations_detected,
                "cache_hits": self.cache_hits,
                "total_tokens_consumed": self.total_tokens_consumed,
                "total_cost_usd": self.total_cost_usd,
                "latencies": self.latencies.copy(),
                "latency_total": self._latency_total,
                "latency_count": self._latency_count,
                "provider_usage": self.provider_usage.copy(),
                "logs": deepcopy(self.logs),
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
            self.provider_usage = {
                key: data.get("provider_usage", {}).get(key, 0) for key in self.provider_usage
            }
            samples = data.get("latencies", [])
            self._latency_total = data.get("latency_total", sum(samples))
            self._latency_count = data.get("latency_count", len(samples))
            self.latencies = samples[-1000:].copy()
            self.logs = deepcopy(data.get("logs", [])[: self.max_logs])
            if not self.include_content:
                for entry in self.logs:
                    entry["prompt"] = entry["response"] = "[omitted]"
                    if entry.get("error") is not None:
                        entry["error"] = "[omitted]"

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
