from unittest.mock import patch

import pytest

from llmrivotril import Guardrail, RivotrilAgent, discover_plugins, load_plugins
from llmrivotril.exceptions import GuardrailViolationError, HallucinationDetectedError
from llmrivotril.providers import BaseProvider, ProviderResponse
from llmrivotril.verifier import Verifier


class _MockProvider(BaseProvider):
    name = "mock"

    def __init__(self):
        super().__init__()
        self._complete_mock = lambda *args, **kwargs: ProviderResponse(content="OK")

    def complete(self, *args, **kwargs):
        return self._complete_mock(*args, **kwargs)

    async def acomplete(self, *args, **kwargs):
        return self._complete_mock(*args, **kwargs)


def _always_ungrounded(response, context_sources=None):
    return False


def _always_grounded(response, context_sources=None):
    return True


def test_load_plugins_accepts_guardrail_instance():
    guardrail = Guardrail(name="test", disallowed_keywords=["bad"])
    guardrails, verifiers = load_plugins([guardrail])

    assert len(guardrails) == 1
    assert len(verifiers) == 0


def test_load_plugins_accepts_verifier_callable():
    guardrails, verifiers = load_plugins([_always_grounded])

    assert len(guardrails) == 0
    assert len(verifiers) == 1
    assert isinstance(verifiers[0], Verifier)


def test_load_plugins_accepts_verifier_instance():
    verifier = Verifier(check_fn=_always_grounded)
    guardrails, verifiers = load_plugins([verifier])

    assert len(guardrails) == 0
    assert len(verifiers) == 1
    assert verifiers[0] is verifier


def test_load_plugins_auto_loads_discovered_plugins():
    with patch(
        "llmrivotril.plugins.discover_plugins",
        return_value={
            "guardrails": {"safe": Guardrail(name="safe", disallowed_keywords=["bad"])},
            "verifiers": {"grounded": _always_grounded},
        },
    ):
        guardrails, verifiers = load_plugins("auto")

    assert len(guardrails) == 1
    assert len(verifiers) == 1


def test_load_plugins_by_name():
    with patch(
        "llmrivotril.plugins.discover_plugins",
        return_value={
            "guardrails": {"safe": Guardrail(name="safe", disallowed_keywords=["bad"])},
            "verifiers": {},
        },
    ):
        guardrails, verifiers = load_plugins(["safe"])

    assert len(guardrails) == 1
    assert guardrails[0].name == "safe"
    assert len(verifiers) == 0


def test_load_plugins_unknown_name_raises():
    with patch(
        "llmrivotril.plugins.discover_plugins",
        return_value={"guardrails": {}, "verifiers": {}},
    ):
        with pytest.raises(ValueError, match="Unknown plugin name"):
            load_plugins(["missing"])


def test_agent_uses_plugin_guardrail():
    guardrail = Guardrail(name="safe", disallowed_keywords=["bad"])
    agent = RivotrilAgent(api_key="test", provider=_MockProvider(), plugins=[guardrail])

    with pytest.raises(GuardrailViolationError):
        agent.run("bad word")


def test_agent_uses_plugin_verifier():
    agent = RivotrilAgent(
        api_key="test",
        provider=_MockProvider(),
        plugins=[_always_ungrounded],
    )

    with pytest.raises(HallucinationDetectedError):
        agent.run("hi", context_sources="any context")


def test_agent_explicit_verifier_overrides_plugins():
    explicit = Verifier(check_fn=_always_grounded)
    agent = RivotrilAgent(
        api_key="test",
        provider=_MockProvider(),
        verifier=explicit,
        plugins=[_always_ungrounded],
    )

    # Should succeed because explicit verifier wins over plugins
    result = agent.run("hi", context_sources="any context")
    assert result == "OK"


def test_discover_plugins_returns_empty_when_no_entry_points():
    plugins = discover_plugins()
    assert "guardrails" in plugins
    assert "verifiers" in plugins
