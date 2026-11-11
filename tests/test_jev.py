import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from llmrivotril import JevGuardrail
from llmrivotril.exceptions import GuardrailViolationError


@pytest.fixture(autouse=True)
def fake_sdk(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", SimpleNamespace(Noul=MagicMock()))


def _resp(p: float) -> MagicMock:
    return MagicMock(answers={"unsafe": MagicMock(noul=p)})


def _run(guardrail, p=None, exc=None, phase="validate_input"):
    with patch.object(guardrail, "_get_client") as get:
        sys_one = get.return_value.system_one
        if exc:
            sys_one.side_effect = exc
        else:
            sys_one.return_value = _resp(p)
        getattr(guardrail, phase)("text")


def test_allows_below_threshold():
    _run(JevGuardrail(), p=0.1)


def test_blocks_input_at_threshold():
    with pytest.raises(GuardrailViolationError, match="Input blocked"):
        _run(JevGuardrail(), p=0.7)


def test_blocks_output():
    with pytest.raises(GuardrailViolationError, match="Output blocked"):
        _run(JevGuardrail(), p=0.9, phase="validate_output")


def test_fails_closed_then_open():
    with pytest.raises(GuardrailViolationError, match="unavailable"):
        _run(JevGuardrail(), exc=RuntimeError("down"))
    _run(JevGuardrail(fail_open=True), exc=RuntimeError("down"))


def test_check_flags_skip():
    g = JevGuardrail(check_input=False)
    with patch.object(g, "_get_client") as get:
        g.validate_input("x")
        get.assert_not_called()


def test_output_rule_blocks_output_only():
    g = JevGuardrail(output_rules=["promises a refund"])
    with patch.object(g, "_get_client") as get:
        get.return_value.system_one.return_value = MagicMock(
            answers={"unsafe": MagicMock(noul=0.0), "rule0": MagicMock(noul=0.9)}
        )
        g.validate_input("x")  # rules not asked on input
        with pytest.raises(GuardrailViolationError, match="output rule"):
            g.validate_output("x")


def test_context_selector_drops_irrelevant_keeps_last():
    from llmrivotril import JevContextSelector

    sel = JevContextSelector()
    turns = [{"role": "user", "content": c} for c in "abc"]
    with patch.object(sel._jev, "_get_client") as get:
        get.return_value.system_one.return_value = MagicMock(
            answers={"t0": MagicMock(noul=0.0), "t1": MagicMock(noul=0.9)}
        )
        assert sel(turns, "q") == turns[1:]


def test_memory_select_fails_open_and_gets_query():
    from llmrivotril import MemoryStore

    def boom(turns, query):
        raise RuntimeError

    m = MemoryStore(select=boom)
    m.add_turn("user", "a")
    assert m.get_context(query="q") == [{"role": "user", "content": "a"}]
    m.select = lambda turns, q: turns[:0]
    assert m.get_context(query="q") == []
    assert len(m.get_context()) == 1  # no query -> no selection


def test_parallel_guardrails_overlap_and_raise():
    from threading import Barrier

    from llmrivotril import RivotrilAgent

    barrier = Barrier(3, timeout=3)

    class Slow:
        name = "slow"

        def validate_input(self, p):
            barrier.wait()

        def validate_output(self, t, model=""):
            pass

    class Bad(Slow):
        name = "bad"

        def validate_input(self, p):
            raise GuardrailViolationError("no")

    a = RivotrilAgent(api_key="k", guardrails=[Slow(), Slow(), Slow()], parallel_guardrails=True)
    a._run_preflight("hi")
    b = RivotrilAgent(api_key="k", guardrails=[Bad(), Bad()], parallel_guardrails=True)
    with pytest.raises(GuardrailViolationError):
        b._run_preflight("hi")


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_scores_fail_closed(score):
    with pytest.raises(GuardrailViolationError, match="unavailable"):
        _run(JevGuardrail(), p=score)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_thresholds_rejected(threshold):
    with pytest.raises(ValueError):
        JevGuardrail(threshold=threshold)


def test_jev_failure_does_not_disclose_sdk_error_or_key(caplog):
    guardrail = JevGuardrail(api_key="secret-key")
    assert "secret-key" not in repr(guardrail)
    assert "secret-key" not in str(guardrail.model_dump())
    with pytest.raises(GuardrailViolationError) as error:
        _run(guardrail, exc=RuntimeError("secret-input-and-key"))
    assert "secret-input-and-key" not in str(error.value)
    assert "secret-input-and-key" not in caplog.text


def test_local_base_url_is_passed_to_sdk(monkeypatch):
    from llmrivotril import JevContextSelector

    sdk = SimpleNamespace(
        Noul=MagicMock(),
        RetryPolicy=MagicMock(),
        TypeSafeClient=MagicMock(return_value=MagicMock()),
    )
    monkeypatch.setitem(sys.modules, "typesafe_sdk", sdk)
    guardrail = JevGuardrail(
        base_url="http://127.0.0.1:8765", api_key="local", model="llm-qwen3.5-4b"
    )
    guardrail._get_client()
    kwargs = sdk.TypeSafeClient.call_args.kwargs
    assert kwargs["base_url"] == "http://127.0.0.1:8765"
    assert kwargs["api_key"] == "local"
    assert kwargs["model"] == "llm-qwen3.5-4b"

    selector = JevContextSelector(base_url="http://127.0.0.1:8765", api_key="local")
    assert selector._jev.base_url == "http://127.0.0.1:8765"


def test_selector_keeps_user_assistant_pairs_and_last_group():
    from llmrivotril import JevContextSelector

    turns = [
        {"role": role, "content": text}
        for role, text in [
            ("user", "old"),
            ("assistant", "old answer"),
            ("user", "recent"),
            ("assistant", "recent answer"),
        ]
    ]
    selector = JevContextSelector(client=MagicMock())
    selector._jev.client.system_one.return_value = MagicMock(answers={"t0": MagicMock(noul=0.0)})
    assert selector(turns, "new question") == turns[2:]
    selector._jev.client.system_one.side_effect = RuntimeError("private")
    assert selector(turns, "new question") == turns


def test_memory_selector_cannot_inject_or_mutate_history():
    from llmrivotril import MemoryStore

    def mutate(turns, query):
        turns[0]["content"] = "injected"
        return turns

    memory = MemoryStore(select=mutate)
    memory.add_turn("user", "original")
    assert memory.get_context(query="q") == [{"role": "user", "content": "original"}]
    assert memory.get_context() == [{"role": "user", "content": "original"}]


def test_parallel_pool_is_bounded_and_block_prevents_generation():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event, Lock

    from llmrivotril import RivotrilAgent

    release, started = Event(), Event()
    lock = Lock()
    active = peak = 0

    class Check:
        name = "local"

        def validate_input(self, text):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
                if active == 2:
                    started.set()
            try:
                assert release.wait(timeout=3)
            finally:
                with lock:
                    active -= 1

    agent = RivotrilAgent(
        api_key="dummy",
        guardrails=[Check() for _ in range(8)],
        parallel_guardrails=True,
        guardrail_workers=2,
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent._run_preflight, "local")
        try:
            assert started.wait(timeout=2)
        finally:
            release.set()
        future.result(timeout=3)
    assert peak == 2
    with patch.object(agent.provider, "complete") as complete:
        agent.guardrails = [JevGuardrail()]
        with patch.object(agent.guardrails[0], "_get_client") as client:
            client.return_value.system_one.return_value = _resp(0.9)
            with pytest.raises(GuardrailViolationError):
                agent.run("blocked locally")
        complete.assert_not_called()
