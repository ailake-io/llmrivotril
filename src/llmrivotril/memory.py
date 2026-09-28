import json
import os
from pathlib import Path
from typing import cast


class MemoryStore:
    """Manages working memory and sliding windows to prevent context drift.

    Pass ``auto_save_path=`` (or set ``RIVOTRIL_MEMORY_PATH``) to persist
    conversation history to a JSON file after every turn and reload it on
    construction, so history survives a process restart -- the same pattern
    ``MetricsCollector``/``RIVOTRIL_METRICS_PATH`` already uses.
    """

    def __init__(
        self, retention_window: int = 10, auto_save_path: str | Path | None = None
    ) -> None:
        self.retention_window = retention_window
        self.history: list[dict[str, str]] = []
        self.auto_save_path = auto_save_path
        load_path = self._resolve_auto_save_path()
        if load_path is not None and load_path.exists():
            self.load_from_json(load_path)

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > (self.retention_window * 2):
            self.history = self.history[-(self.retention_window * 2) :]
        self._maybe_auto_save()

    def get_context(self) -> list[dict[str, str]]:
        return self.history

    def clear(self) -> None:
        self.history.clear()
        self._maybe_auto_save()

    def _resolve_auto_save_path(self) -> Path | None:
        path = self.auto_save_path
        if path is None:
            env_path = os.environ.get("RIVOTRIL_MEMORY_PATH")
            if env_path:
                path = env_path
        return Path(path) if path else None

    def _maybe_auto_save(self) -> None:
        path = self._resolve_auto_save_path()
        if path is not None:
            self.save_to_json(path)

    def to_dict(self) -> dict[str, object]:
        """Return a serializable snapshot of the memory state."""
        return {
            "retention_window": self.retention_window,
            "history": self.history.copy(),
        }

    def from_dict(self, data: dict[str, object]) -> None:
        """Restore memory state from a dictionary."""
        raw_retention_window = data.get("retention_window", 10)
        if isinstance(raw_retention_window, int):
            self.retention_window = raw_retention_window
        else:
            self.retention_window = int(cast("str | float", raw_retention_window))

        history = data.get("history", [])
        if isinstance(history, list):
            self.history = [dict(turn) for turn in history if isinstance(turn, dict)]
        else:
            self.history = []

    def save_to_json(self, path: str | Path) -> None:
        """Persist the current conversation history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load_from_json(self, path: str | Path) -> None:
        """Restore conversation history from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.from_dict(data)
