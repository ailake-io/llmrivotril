from unittest.mock import MagicMock, patch

import pytest

from llmrivotril import ModerationGuardrail
from llmrivotril.exceptions import GuardrailViolationError


def _mock_moderation_result(flagged: bool) -> MagicMock:
    result = MagicMock()
    result.results = [MagicMock(flagged=flagged)]
    return result


def test_allows_clean_input():
    guardrail = ModerationGuardrail()
    with patch.object(guardrail, "_get_client") as mock_get_client:
        mock_get_client.return_value.moderations.create.return_value = _mock_moderation_result(
            flagged=False
        )
        guardrail.validate_input("What's a good recipe for banana bread?")


def test_blocks_flagged_input():
    guardrail = ModerationGuardrail()
    with patch.object(guardrail, "_get_client") as mock_get_client:
        mock_get_client.return_value.moderations.create.return_value = _mock_moderation_result(
            flagged=True
        )
        with pytest.raises(GuardrailViolationError, match="Input blocked"):
            guardrail.validate_input("something adversarial")


def test_blocks_flagged_output():
    guardrail = ModerationGuardrail()
    with patch.object(guardrail, "_get_client") as mock_get_client:
        mock_get_client.return_value.moderations.create.return_value = _mock_moderation_result(
            flagged=True
        )
        with pytest.raises(GuardrailViolationError, match="Output blocked"):
            guardrail.validate_output("something adversarial")


def test_check_input_false_skips_input_check():
    guardrail = ModerationGuardrail(check_input=False)
    with patch.object(guardrail, "_get_client") as mock_get_client:
        guardrail.validate_input("anything")
        mock_get_client.assert_not_called()


def test_check_output_false_skips_output_check():
    guardrail = ModerationGuardrail(check_output=False)
    with patch.object(guardrail, "_get_client") as mock_get_client:
        guardrail.validate_output("anything")
        mock_get_client.assert_not_called()


def test_fails_closed_by_default_when_api_call_errors():
    guardrail = ModerationGuardrail()
    with patch.object(guardrail, "_get_client") as mock_get_client:
        mock_get_client.return_value.moderations.create.side_effect = RuntimeError("API down")
        with pytest.raises(GuardrailViolationError, match="moderation check unavailable"):
            guardrail.validate_input("anything")


def test_fail_open_lets_request_through_when_api_call_errors():
    guardrail = ModerationGuardrail(fail_open=True)
    with patch.object(guardrail, "_get_client") as mock_get_client:
        mock_get_client.return_value.moderations.create.side_effect = RuntimeError("API down")
        guardrail.validate_input("anything")  # must not raise


def test_get_client_constructs_and_caches_openai_client():
    guardrail = ModerationGuardrail(api_key="test-key")
    with patch("openai.OpenAI") as mock_openai_class:
        client1 = guardrail._get_client()
        client2 = guardrail._get_client()

        mock_openai_class.assert_called_once_with(api_key="test-key")
        assert client1 is client2


def test_empty_text_skips_the_api_call():
    guardrail = ModerationGuardrail()
    with patch.object(guardrail, "_get_client") as mock_get_client:
        guardrail.validate_input("")
        mock_get_client.assert_not_called()
