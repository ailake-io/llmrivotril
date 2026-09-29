from llmrivotril import MemoryStore


def test_add_turn_and_retrieve():
    memory = MemoryStore(retention_window=2)
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "hi")

    context = memory.get_context()
    assert len(context) == 2
    assert context[0]["role"] == "user"
    assert context[0]["content"] == "hello"


def test_retention_window():
    memory = MemoryStore(retention_window=1)
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "first reply")
    memory.add_turn("user", "second")
    memory.add_turn("assistant", "second reply")
    memory.add_turn("user", "third")

    context = memory.get_context()
    assert len(context) == 2
    assert context[-1]["content"] == "third"


def test_clear():
    memory = MemoryStore()
    memory.add_turn("user", "hello")
    memory.clear()
    assert memory.get_context() == []


def test_to_dict_and_from_dict():
    memory = MemoryStore(retention_window=3)
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "hi")

    snapshot = memory.to_dict()
    assert snapshot["retention_window"] == 3
    assert len(snapshot["history"]) == 2

    restored = MemoryStore()
    restored.from_dict(snapshot)
    assert restored.get_context() == memory.get_context()
    assert restored.retention_window == 3


def test_save_and_load_from_json(tmp_path):
    memory = MemoryStore(retention_window=2)
    memory.add_turn("user", "hello")
    memory.add_turn("assistant", "hi there")

    path = tmp_path / "memory.json"
    memory.save_to_json(path)

    restored = MemoryStore()
    restored.load_from_json(path)

    assert restored.retention_window == 2
    assert restored.get_context() == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_auto_save_path_persists_after_every_turn(tmp_path):
    path = tmp_path / "auto_memory.json"
    memory = MemoryStore(auto_save_path=path)

    memory.add_turn("user", "hello")
    assert path.exists()

    # A fresh instance pointed at the same path picks up the saved turn --
    # this proves it's written after every add_turn, not just on demand.
    reloaded = MemoryStore(auto_save_path=path)
    assert reloaded.get_context() == [{"role": "user", "content": "hello"}]


def test_auto_save_path_via_env_var(tmp_path, monkeypatch):
    path = tmp_path / "env_memory.json"
    monkeypatch.setenv("RIVOTRIL_MEMORY_PATH", str(path))

    memory = MemoryStore()
    memory.add_turn("user", "hi")

    assert path.exists()
    reloaded = MemoryStore()
    assert reloaded.get_context() == [{"role": "user", "content": "hi"}]


def test_auto_save_path_creates_missing_parent_directory(tmp_path):
    path = tmp_path / "nested" / "dir" / "memory.json"
    memory = MemoryStore(auto_save_path=path)

    memory.add_turn("user", "hello")

    assert path.exists()


def test_max_tokens_trims_oldest_turns_beyond_budget():
    # Default word-count fallback: each turn below is one word (one "token").
    memory = MemoryStore(retention_window=10, max_tokens=2)
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "second")
    memory.add_turn("user", "third")

    context = memory.get_context()
    assert [turn["content"] for turn in context] == ["second", "third"]


def test_max_tokens_never_drops_the_last_turn():
    memory = MemoryStore(retention_window=10, max_tokens=1)
    memory.add_turn("user", "one two three four five")

    context = memory.get_context()
    assert len(context) == 1
    assert context[0]["content"] == "one two three four five"


def test_max_tokens_uses_custom_count_tokens_callable():
    memory = MemoryStore(retention_window=10, max_tokens=5, count_tokens=len)
    memory.add_turn("user", "aaaaaa")  # 6 chars, alone already over budget but kept (last turn)
    memory.add_turn("assistant", "b")  # 1 char

    context = memory.get_context()
    # Adding "b" makes the total 7 > 5, so the oldest ("aaaaaa") is dropped.
    assert [turn["content"] for turn in context] == ["b"]


def test_no_max_tokens_keeps_retention_window_behavior_unchanged():
    memory = MemoryStore(retention_window=1, max_tokens=None)
    memory.add_turn("user", "a very long message with many words in it")
    memory.add_turn("assistant", "reply")

    assert len(memory.get_context()) == 2


def test_summarize_is_off_by_default():
    memory = MemoryStore(retention_window=1, summarize_trigger_turns=1)
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "first reply")
    memory.add_turn("user", "second")

    # No summarize= callable given: falls back to the plain turn-count cap,
    # no summary is ever produced even past summarize_trigger_turns.
    context = memory.get_context()
    assert all(turn["role"] != "system" for turn in context)


def test_summarize_replaces_dropped_turns_with_a_summary():
    calls = []

    def fake_summarize(text: str) -> str:
        calls.append(text)
        return "SUMMARY"

    memory = MemoryStore(
        retention_window=1,
        summarize=fake_summarize,
        summarize_trigger_turns=2,
    )
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "first reply")
    memory.add_turn("user", "second")

    context = memory.get_context()
    assert context[0] == {
        "role": "system",
        "content": "Summary of earlier conversation:\nSUMMARY",
    }
    # retention_window=1 keeps the last 2 raw turns verbatim; only what's
    # pushed beyond that ("first") gets summarized.
    assert [turn["content"] for turn in context[1:]] == ["first reply", "second"]
    assert calls == ["user: first"]


def test_summarize_failure_is_swallowed_and_turns_stay_dropped():
    def failing_summarize(text: str) -> str:
        raise RuntimeError("boom")

    memory = MemoryStore(
        retention_window=1,
        summarize=failing_summarize,
        summarize_trigger_turns=2,
    )
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "first reply")
    memory.add_turn("user", "second")

    # Summarization failed, but the oldest turn was already evicted by the
    # trim step -- no summary is added, and no exception propagates.
    context = memory.get_context()
    assert [turn["content"] for turn in context] == ["first reply", "second"]


def test_summary_persists_through_to_dict_and_from_dict():
    memory = MemoryStore(
        retention_window=1,
        summarize=lambda text: "SUMMARY",
        summarize_trigger_turns=2,
    )
    memory.add_turn("user", "first")
    memory.add_turn("assistant", "first reply")
    memory.add_turn("user", "second")

    snapshot = memory.to_dict()
    assert snapshot["summary"] == "SUMMARY"

    restored = MemoryStore()
    restored.from_dict(snapshot)
    assert restored.get_context()[0]["content"] == "Summary of earlier conversation:\nSUMMARY"
