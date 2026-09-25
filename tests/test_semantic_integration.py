"""Integration tests for semantic components using a real local embedding model.

These tests are skipped unless ``sentence-transformers`` is installed and the
``--run-slow`` pytest flag is passed, because they download/load the model on
the first run.
"""

import pytest

from llmrivotril.exceptions import GuardrailViolationError
from llmrivotril.semantic import EmbeddingFaithfulnessVerifier, SemanticTopicGuardrail

try:
    import sentence_transformers  # noqa: F401

    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover
    _SENTENCE_TRANSFORMERS_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(
        not _SENTENCE_TRANSFORMERS_AVAILABLE, reason="sentence-transformers missing"
    ),
    pytest.mark.slow,
]


def test_semantic_topic_guardrail_real_model():
    guardrail = SemanticTopicGuardrail(
        name="real-semantic",
        allowed_topics=["machine learning", "baking bread"],
        similarity_threshold=0.35,
    )

    # "neural networks" is semantically close to "machine learning".
    guardrail.validate_input("explain neural networks")

    # "how to bake sourdough" is semantically close to "baking bread".
    guardrail.validate_input("how to bake sourdough")

    # An unrelated prompt should be blocked.
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_input("what is the stock market forecast")


def test_embedding_faithfulness_verifier_real_model():
    verifier = EmbeddingFaithfulnessVerifier(similarity_threshold=0.5)

    context = "The capital of France is Paris."
    grounded = "Paris is the capital of France."
    ungrounded = "The speed of light is 299,792 km per second."

    assert verifier.verify(grounded, context) is True
    assert verifier.verify(ungrounded, context) is False
    assert verifier.verify("any answer", None) is True
