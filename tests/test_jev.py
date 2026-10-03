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
        with pytest.raises(GuardrailViolationError, match="promises a refund"):
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
    import time

    from llmrivotril import RivotrilAgent

    class Slow:
        name = "slow"

        def validate_input(self, p):
            time.sleep(0.3)

        def validate_output(self, t, model=""):
            pass

    class Bad(Slow):
        name = "bad"

        def validate_input(self, p):
            raise GuardrailViolationError("no")

    a = RivotrilAgent(api_key="k", guardrails=[Slow(), Slow(), Slow()], parallel_guardrails=True)
    t = time.perf_counter()
    a._run_preflight("hi")
    assert time.perf_counter() - t < 0.7
    b = RivotrilAgent(api_key="k", guardrails=[Slow(), Bad()], parallel_guardrails=True)
    with pytest.raises(GuardrailViolationError):
        b._run_preflight("hi")
