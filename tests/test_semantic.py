import pytest

from llmrivotril.exceptions import GuardrailViolationError
from llmrivotril.semantic import EmbeddingFaithfulnessVerifier, SemanticTopicGuardrail


@pytest.fixture
def guardrail():
    gr = SemanticTopicGuardrail(
        name="semantic-test",
        allowed_topics=["science", "sports"],
        disallowed_keywords=["password"],
        similarity_threshold=0.9,
    )
    embeddings = {
        "science": [1.0, 0.0],
        "sports": [0.0, 1.0],
        "physics is interesting": [0.98, 0.02],
        "baking a cake": [0.1, 0.1],
    }
    gr._embed_fn = lambda text: embeddings.get(text, [0.0, 0.0])
    gr._topic_embeddings = [embeddings[t] for t in gr.allowed_topics]
    return gr


def test_semantic_guardrail_accepts_close_prompt(guardrail):
    guardrail.validate_input("physics is interesting")


def test_semantic_guardrail_blocks_unrelated_prompt(guardrail):
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_input("baking a cake")


def test_semantic_guardrail_blocks_disallowed_keyword(guardrail):
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_input("what is the password?")


def test_semantic_guardrail_allows_anything_without_topics():
    gr = SemanticTopicGuardrail(name="no-topics")
    gr._embed_fn = lambda text: [0.0, 0.0]
    gr._topic_embeddings = []
    gr.validate_input("anything goes")


def test_semantic_guardrail_output_token_limit():
    gr = SemanticTopicGuardrail(name="token-test", max_tokens=5)
    gr.validate_output("short")
    with pytest.raises(GuardrailViolationError):
        gr.validate_output("this text is way longer than five tokens for sure")


def test_semantic_guardrail_output_uses_fallback_for_unknown_model():
    gr = SemanticTopicGuardrail(name="fallback-token-test", max_tokens=5)
    gr.validate_output("short", model="unknown-model")


@pytest.fixture
def embedding_verifier():
    verifier = EmbeddingFaithfulnessVerifier(similarity_threshold=0.9)
    embeddings = {
        "the capital of france is paris": [1.0, 0.0],
        "paris is the capital of france": [0.98, 0.02],
        "the speed of light is fast": [0.1, 0.1],
    }
    verifier._embed_fn = lambda text: embeddings.get(text, [0.0, 0.0])
    return verifier


def test_embedding_verifier_accepts_grounded_response(embedding_verifier):
    assert (
        embedding_verifier.verify(
            "paris is the capital of france",
            "the capital of france is paris",
        )
        is True
    )


def test_embedding_verifier_rejects_ungrounded_response(embedding_verifier):
    assert (
        embedding_verifier.verify(
            "the speed of light is fast",
            "the capital of france is paris",
        )
        is False
    )


def test_embedding_verifier_allows_without_context(embedding_verifier):
    assert embedding_verifier.verify("anything", None) is True


def test_embedding_verifier_as_callable(embedding_verifier):
    check_fn = embedding_verifier.as_callable()
    assert check_fn("paris is the capital of france", "the capital of france is paris") is True
    assert check_fn("the speed of light is fast", "the capital of france is paris") is False
