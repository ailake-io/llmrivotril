from llmrivotril import PIIRedactor


def test_redact_email():
    redactor = PIIRedactor()
    text = "Contact me at john.doe@example.com please."
    assert redactor.redact(text) == "Contact me at [REDACTED] please."


def test_redact_cpf():
    redactor = PIIRedactor()
    text = "My CPF is 123.456.789-09."
    assert redactor.redact(text) == "My CPF is [REDACTED]."


def test_redact_cnpj():
    redactor = PIIRedactor()
    text = "CNPJ: 12.345.678/0001-95"
    assert redactor.redact(text) == "CNPJ: [REDACTED]"


def test_redact_phone():
    redactor = PIIRedactor()
    text = "Call me at (11) 91234-5678."
    assert redactor.redact(text) == "Call me at [REDACTED]."


def test_redact_credit_card_valid():
    redactor = PIIRedactor()
    # Valid Luhn number
    text = "Card: 4532015112830366"
    assert redactor.redact(text) == "Card: [REDACTED]"


def test_redact_credit_card_invalid_skipped_with_luhn():
    redactor = PIIRedactor()
    # Invalid Luhn number
    text = "Card: 4532015112830367"
    assert redactor.redact(text) == "Card: 4532015112830367"


def test_detect_returns_findings():
    redactor = PIIRedactor()
    findings = redactor.detect("Email: a@b.com and CPF 123.456.789-09")
    types = {f["type"] for f in findings}
    assert "email" in types
    assert "cpf" in types


def test_custom_replacement():
    redactor = PIIRedactor()
    assert redactor.redact("a@b.com", replacement="***") == "***"


def test_no_pii_returns_original():
    redactor = PIIRedactor()
    text = "Just a regular sentence."
    assert redactor.redact(text) == text


def test_redact_merges_overlapping_spans_from_different_patterns():
    # Two patterns whose matches overlap ("abc123" and "123xyz" share "123").
    # Replacing each span independently by raw offsets corrupts the string;
    # they must be merged into a single span first.
    redactor = PIIRedactor(patterns={"a": r"abc\d\d\d", "b": r"\d\d\dxyz"})
    assert redactor.redact("abc123xyz") == "[REDACTED]"
    assert redactor.redact("before abc123xyz after") == "before [REDACTED] after"
