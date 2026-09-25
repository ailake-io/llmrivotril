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
