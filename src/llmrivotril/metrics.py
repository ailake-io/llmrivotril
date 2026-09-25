import json
import time
from pathlib import Path
from threading import Lock
from typing import Any


class MetricsCollector:
    """Thread-safe telemetry store.

    Tracks requests, tokens, guardrail blocks, and hallucinations.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self.requests_total = 0
        self.guardrail_blocks = 0
        self.hallucinations_detected = 0
        self.total_tokens_consumed = 0
        self.latencies: list[float] = []
        self.logs: list[dict[str, Any]] = []

    def log_execution(
        self,
        prompt: str,
        response: str,
        tokens: int,
        latency: float,
        guardrail_blocked: bool = False,
        hallucination_blocked: bool = False,
        error: str | None = None,
    ) -> None:
        with self._lock:
            self.requests_total += 1
            self.total_tokens_consumed += tokens
            self.latencies.append(latency)

            if guardrail_blocked:
                self.guardrail_blocks += 1
            if hallucination_blocked:
                self.hallucinations_detected += 1

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
                },
            )
            # Keep history capped at 100 items
            if len(self.logs) > 100:
                self.logs.pop()

    def reset(self) -> None:
        with self._lock:
            self.requests_total = 0
            self.guardrail_blocks = 0
            self.hallucinations_detected = 0
            self.total_tokens_consumed = 0
            self.latencies.clear()
            self.logs.clear()

    def get_summary(self) -> dict[str, Any]:
        with self._lock:
            avg_latency = sum(self.latencies) / len(self.latencies) if self.latencies else 0.0
            blocked = self.guardrail_blocks + self.hallucinations_detected
            if self.requests_total > 0:
                success_rate = max(0.0, (self.requests_total - blocked) / self.requests_total * 100)
            else:
                success_rate = 100.0
            return {
                "requests_total": self.requests_total,
                "guardrail_blocks": self.guardrail_blocks,
                "hallucinations_detected": self.hallucinations_detected,
                "total_tokens_consumed": self.total_tokens_consumed,
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
                "total_tokens_consumed": self.total_tokens_consumed,
                "latencies": self.latencies.copy(),
                "logs": self.logs.copy(),
            }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore collector state from a dictionary."""
        with self._lock:
            self.requests_total = data.get("requests_total", 0)
            self.guardrail_blocks = data.get("guardrail_blocks", 0)
            self.hallucinations_detected = data.get("hallucinations_detected", 0)
            self.total_tokens_consumed = data.get("total_tokens_consumed", 0)
            self.latencies = data.get("latencies", []).copy()
            self.logs = data.get("logs", []).copy()

    def save_to_json(self, path: str | Path) -> None:
        """Persist the current state to a JSON file."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load_from_json(self, path: str | Path) -> None:
        """Restore state from a JSON file created by ``save_to_json``."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.from_dict(data)


# Global singleton metrics instance
global_metrics = MetricsCollector()
