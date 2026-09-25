from unittest.mock import MagicMock, patch

from llmrivotril.verifier import (
    CitationVerifier,
    KeywordOverlapVerifier,
    ModelBasedFaithfulnessVerifier,
    Verifier,
)


def test_default_verifier_accepts_without_context():
    verifier = Verifier()
    assert verifier.verify_grounding("any response") is True


def test_default_verifier_accepts_with_context_and_overlapping_response():
    verifier = Verifier()
    assert (
        verifier.verify_grounding(
            "Paris is the capital of France.",
            context_sources="The capital of France is Paris.",
        )
        is True
    )


def test_default_verifier_rejects_unrelated_response_with_context():
    verifier = Verifier()
    assert (
        verifier.verify_grounding(
            "I love playing football on weekends.",
            context_sources="The capital of France is Paris.",
        )
        is False
    )


def test_default_verifier_rejects_empty_response_with_context():
    verifier = Verifier()
    assert verifier.verify_grounding("   ", context_sources="source text") is False


def test_custom_check_fn():
    def strict_check(response: str, context: str | None) -> bool:
        return response == "yes"

    verifier = Verifier(check_fn=strict_check)
    assert verifier.verify_grounding("yes") is True
    assert verifier.verify_grounding("no") is False


def test_keyword_overlap_accepts_grounded_response():
    verifier = KeywordOverlapVerifier(threshold=0.1)
    context = "The capital of France is Paris and the Eiffel Tower is there."
    response = "Paris is the capital of France."
    assert verifier.verify(response, context) is True


def test_keyword_overlap_rejects_ungrounded_response():
    verifier = KeywordOverlapVerifier(threshold=0.1)
    context = "The capital of France is Paris."
    response = "I love playing football on weekends."
    assert verifier.verify(response, context) is False


def test_citation_verifier_accepts_quoted_text():
    verifier = CitationVerifier()
    context = "The speed of light is 299,792 km/s."
    response = 'As noted, "the speed of light is 299,792 km/s".'
    assert verifier.verify(response, context) is True


def test_citation_verifier_rejects_without_citation():
    verifier = CitationVerifier()
    context = "The speed of light is 299,792 km/s."
    response = "The speed of light is very fast."
    assert verifier.verify(response, context) is False


def test_model_based_verifier_accepts_grounded_response():
    verifier = ModelBasedFaithfulnessVerifier(api_key="test-key")
    mock_verdict = MagicMock()
    mock_verdict.verdict = True
    mock_verdict.confidence = 0.9
    mock_verdict.reason = "Grounded."

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_verdict

    with (
        patch("openai.OpenAI") as mock_openai_class,
        patch("instructor.from_openai", return_value=mock_client) as mock_from_openai,
    ):
        result = verifier.verify("Paris is the capital.", "The capital is Paris.")

        assert result is True
        mock_openai_class.assert_called_once_with(api_key="test-key", base_url=None)
        mock_from_openai.assert_called_once()


def test_model_based_verifier_rejects_ungrounded_response():
    verifier = ModelBasedFaithfulnessVerifier(api_key="test-key")
    mock_verdict = MagicMock()
    mock_verdict.verdict = False
    mock_verdict.confidence = 0.95
    mock_verdict.reason = "Not grounded."

    with patch.object(verifier, "_get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.return_value = mock_verdict

        result = verifier.verify("London is the capital.", "The capital is Paris.")

        assert result is False


def test_model_based_verifier_fails_open_on_judge_error():
    verifier = ModelBasedFaithfulnessVerifier(api_key="test-key")

    with patch.object(verifier, "_get_client") as mock_get_client:
        mock_get_client.return_value.chat.completions.create.side_effect = RuntimeError("API down")

        result = verifier.verify("Any response.", "Any context.")

        assert result is True


def test_model_based_verifier_accepts_without_context():
    verifier = ModelBasedFaithfulnessVerifier(api_key="test-key")
    assert verifier.verify("Any response.", None) is True
