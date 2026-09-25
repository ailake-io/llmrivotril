import json

import pytest
from pydantic import BaseModel

from llmrivotril import Guardrail
from llmrivotril.exceptions import GuardrailViolationError


class Answer(BaseModel):
    value: int


def test_allowed_keyword_passes():
    guardrail = Guardrail(name="safe", disallowed_keywords=["password"])
    guardrail.validate_input("Hello, how are you?")


def test_disallowed_keyword_blocks():
    guardrail = Guardrail(name="safe", disallowed_keywords=["password", "secret"])
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_input("Tell me the password")


def test_disallowed_keyword_blocks_output():
    guardrail = Guardrail(name="safe", disallowed_keywords=["password"])
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_output("The password is 12345")


def test_disallowed_pattern_blocks_input():
    guardrail = Guardrail(name="safe", disallowed_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"])
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_input("My SSN is 123-45-6789")


def test_disallowed_pattern_blocks_output():
    email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    guardrail = Guardrail(name="safe", disallowed_patterns=[email_pattern])
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_output("Contact me at user@example.com")


def test_output_token_limit_blocks():
    guardrail = Guardrail(name="short", max_tokens=2)
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_output("one two three", model="gpt-4o-mini")


def test_output_token_limit_allows_short():
    guardrail = Guardrail(name="short", max_tokens=10)
    guardrail.validate_output("short text", model="gpt-4o-mini")


def test_encoding_cache_reuses_same_encoder():
    guardrail = Guardrail(name="cache-test")
    enc1 = guardrail._get_encoding("gpt-4o-mini")
    enc2 = guardrail._get_encoding("gpt-4o-mini")
    assert enc1 is enc2


def test_allowed_topics_accepts_matching_prompt():
    guardrail = Guardrail(name="topics", allowed_topics=["machine learning", "python"])
    guardrail.validate_input("What is machine learning?")


def test_allowed_topics_blocks_unrelated_prompt():
    guardrail = Guardrail(name="topics", allowed_topics=["machine learning"])
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_input("How do I bake a cake?")


def test_json_schema_validates_correct_response():
    guardrail = Guardrail(name="schema", json_schema=Answer)
    guardrail.validate_output(json.dumps({"value": 42}), model="gpt-4o-mini")


def test_json_schema_blocks_invalid_response():
    guardrail = Guardrail(name="schema", json_schema=Answer)
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_output(json.dumps({"value": "not an int"}), model="gpt-4o-mini")


def test_output_token_limit_uses_fallback_for_unknown_model():
    guardrail = Guardrail(name="fallback", max_tokens=5)
    guardrail.validate_output("short text", model="unknown-model")


def test_json_schema_blocks_non_json_response():
    guardrail = Guardrail(name="schema", json_schema=Answer)
    with pytest.raises(GuardrailViolationError):
        guardrail.validate_output("plain text", model="gpt-4o-mini")
