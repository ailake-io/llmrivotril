import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from threading import Lock
from typing import cast

logger = logging.getLogger("llmrivotril")


def _default_count_tokens(text: str) -> int:
    """Whitespace-word-count fallback when no real tokenizer is supplied.

    A rough proxy, not an exact token count -- ``RivotrilAgent`` passes its
    own tiktoken-backed counter for its default ``MemoryStore``, so this only
    matters for a ``MemoryStore`` built standalone without one.
    """
    return len(text.split())


class MemoryStore:
    """Manages working memory and sliding windows to prevent context drift.

    Pass ``auto_save_path=`` (or set ``RIVOTRIL_MEMORY_PATH``) to persist
    conversation history to a JSON file after every turn and reload it on
    construction, so history survives a process restart -- the same pattern
    ``MetricsCollector``/``RIVOTRIL_METRICS_PATH`` already uses.

    ``retention_window`` caps history by turn count (kept for backward
    compatibility and as a cheap absolute ceiling). ``max_tokens`` layers an
    additional, tighter cap by actual token count on top of it -- turns vary
    a lot in length, so a token budget bounds prompt growth far more directly
    than a fixed turn count. ``summarize`` is fully optional (``None`` by
    default, no behavior change unless explicitly supplied): a callable,
    usually backed by an LLM call, that compacts old turns into a running
    summary instead of silently dropping them once ``summarize_trigger_turns``
    is exceeded.
    """

    def __init__(
        self,
        retention_window: int = 10,
        auto_save_path: str | Path | None = None,
        max_tokens: int | None = None,
        count_tokens: Callable[[str], int] | None = None,
        summarize: Callable[[str], str] | None = None,
        summarize_trigger_turns: int = 20,
    ) -> None:
        if retention_window <= 0:
            raise ValueError("retention_window must be positive")
        self.retention_window = retention_window
        self.max_tokens = max_tokens
        self.count_tokens = count_tokens or _default_count_tokens
        self.summarize = summarize
        self.summarize_trigger_turns = summarize_trigger_turns
        self.history: list[dict[str, str]] = []
        self._summary: str | None = None
        self.auto_save_path = auto_save_path
        self._lock = Lock()
        load_path = self._resolve_auto_save_path()
        if load_path is not None and load_path.exists():
            self.load_from_json(load_path)

    def add_turn(self, role: str, content: str) -> None:
        with self._lock:
            self.history.append({"role": role, "content": content})
            to_summarize = self._trim_locked()
        if to_summarize:
            self._apply_summary(to_summarize)
        self._maybe_auto_save()

    def _trim_locked(self) -> list[dict[str, str]] | None:
        """Cap ``self.history``, returning any turns handed off for summarization.

        Must be called with ``self._lock`` held. Runs in three stages: first
        either summarization-aware trimming (only if ``summarize`` was
        supplied) or a plain turn-count cap, never both -- summarizing already
        bounds history to the same size the plain cap would -- then a
        token-budget cap layered on top of whichever ran.
        """
        to_summarize: list[dict[str, str]] | None = None
        keep = self.retention_window * 2
        if self.summarize is not None and len(self.history) > self.summarize_trigger_turns:
            # A trigger below the retention cap must still fire: never keep
            # more raw turns than the trigger itself allows.
            keep = min(keep, max(self.summarize_trigger_turns, 0))
            if keep > 0 and len(self.history) > keep:
                to_summarize = self.history[:-keep]
                self.history = self.history[-keep:]
            elif keep <= 0:
                to_summarize = self.history[:]
                self.history = []
        elif len(self.history) > keep:
            self.history = self.history[-keep:]

        if self.max_tokens is not None:
            while len(self.history) > 1 and self._total_tokens_locked() > self.max_tokens:
                self.history.pop(0)
        return to_summarize

    def _total_tokens_locked(self) -> int:
        return sum(self.count_tokens(turn["content"]) for turn in self.history)

    def _apply_summary(self, turns: list[dict[str, str]]) -> None:
        """Summarize ``turns`` (already evicted from history) outside the lock.

        Runs the (usually network-bound) ``summarize`` call without holding
        ``self._lock``, so a slow summarization doesn't block concurrent
        ``add_turn``/``get_context`` calls. A failure here just skips this
        round's summary rather than losing the turn or crashing the caller --
        those turns are already gone from ``self.history`` either way.
        """
        assert self.summarize is not None
        text = "\n".join(f"{turn['role']}: {turn['content']}" for turn in turns)
        try:
            new_summary = self.summarize(text)
        except Exception:
            logger.warning("Memory history summarization failed; skipping.", exc_info=True)
            return
        with self._lock:
            self._summary = f"{self._summary}\n{new_summary}" if self._summary else new_summary

    def get_context(self) -> list[dict[str, str]]:
        with self._lock:
            turns = [turn.copy() for turn in self.history]
            summary = self._summary
        if summary:
            summary_turn = {
                "role": "system",
                "content": f"Summary of earlier conversation:\n{summary}",
            }
            return [summary_turn, *turns]
        return turns

    def clear(self) -> None:
        with self._lock:
            self.history.clear()
            self._summary = None
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
        """Return a serializable snapshot of the memory state.

        ``max_tokens``/``count_tokens``/``summarize`` are runtime
        configuration supplied by the constructing code (e.g. ``RivotrilAgent``
        wiring in its own tokenizer/provider) -- not part of the persisted
        conversation state, and ``count_tokens``/``summarize`` aren't
        JSON-serializable anyway.
        """
        with self._lock:
            return {
                "retention_window": self.retention_window,
                "history": [turn.copy() for turn in self.history],
                "summary": self._summary,
            }

    def from_dict(self, data: dict[str, object]) -> None:
        """Restore memory state from a dictionary."""
        raw_retention_window = data.get("retention_window", 10)
        if isinstance(raw_retention_window, int):
            retention_window = raw_retention_window
        else:
            retention_window = int(cast("str | float", raw_retention_window))
        if retention_window <= 0:
            raise ValueError("retention_window must be positive")

        history = data.get("history", [])
        summary = data.get("summary")
        with self._lock:
            self.retention_window = retention_window
            if isinstance(history, list):
                self.history = [dict(turn) for turn in history if isinstance(turn, dict)]
            else:
                self.history = []
            self._summary = summary if isinstance(summary, str) else None

    def save_to_json(self, path: str | Path) -> None:
        """Persist the current conversation history to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary_path.replace(path)

    def load_from_json(self, path: str | Path) -> None:
        """Restore conversation history from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        self.from_dict(data)
