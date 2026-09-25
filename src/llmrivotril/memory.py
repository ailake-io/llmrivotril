import json
from pathlib import Path
from typing import cast


class MemoryStore:
    """Manages working memory and sliding windows to prevent context drift."""

    def __init__(self, retention_window: int = 10) -> None:
        self.retention_window = retention_window
        self.history: list[dict[str, str]] = []

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > (self.retention_window * 2):
            self.history = self.history[-(self.retention_window * 2) :]

    def get_context(self) -> list[dict[str, str]]:
        return self.history

    def clear(self) -> None:
        self.history.clear()

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
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def load_from_json(self, path: str | Path) -> None:
        """Restore conversation history from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.from_dict(data)
