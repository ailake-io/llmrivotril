from scripts.compare_local import compare


def test_local_comparison_counts_calls_without_claiming_billing_savings():
    result = compare(3)
    assert result["synthetic"] is True
    assert result["billing_or_real_quality_claim"] is False
    rows = {row["scenario"]: row for row in result["results"]}
    assert rows["direct"]["provider_calls"] == 3
    assert rows["agent"]["provider_calls"] == 3
    assert rows["cache"]["provider_calls"] == 1
    assert rows["cache"]["correct_answers"] == 3
    assert rows["guardrail"]["provider_calls"] == 0
    assert rows["guardrail"]["blocked"] == 3
