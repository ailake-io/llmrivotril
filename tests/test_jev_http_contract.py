"""Real optional TypeSafe SDK with an in-process transport only."""

import json

import pytest

sdk = pytest.importorskip("typesafe_sdk")
httpx2 = pytest.importorskip("httpx2")

from llmrivotril import JevGuardrail  # noqa: E402
from llmrivotril.exceptions import GuardrailViolationError  # noqa: E402


@pytest.mark.parametrize("probability,blocked", [(0.1, False), (0.8, True)])
def test_real_sdk_scores_and_output_rules(probability, blocked):
    requests = []

    def handle(request):
        requests.append(request)
        assert request.url.path == "/v1/systemone"
        body = json.loads(request.content)
        assert "unsafe" in body["questions"]
        assert "rule0" in body["questions"]
        assert body["model"] == "jev-local"
        return httpx2.Response(
            200,
            json={
                "model": "jev-local",
                "usage": {"input_tokens": 10, "output_tokens": 2},
                "answers": {
                    "unsafe": {"type": "noul", "noul": 0.0},
                    "rule0": {"type": "noul", "noul": probability},
                },
            },
        )

    client = sdk.TypeSafeClient(
        api_key="local-dummy",
        model="jev-local",
        base_url="https://local.invalid",
        transport=httpx2.MockTransport(handle),
        retry=sdk.RetryPolicy(max_retries=0),
        compression=False,
    )
    try:
        guardrail = JevGuardrail(client=client, output_rules=["The text promises a refund"])
        if blocked:
            with pytest.raises(GuardrailViolationError, match="output rule"):
                guardrail.validate_output("local response")
        else:
            guardrail.validate_output("local response")
        assert len(requests) == 1
    finally:
        client.close()


def test_real_sdk_http_error_fails_closed_without_disclosure():
    client = sdk.TypeSafeClient(
        api_key="local-dummy",
        base_url="https://local.invalid",
        transport=httpx2.MockTransport(
            lambda request: httpx2.Response(500, json={"error": "sensitive-upstream-detail"})
        ),
        retry=sdk.RetryPolicy(max_retries=0),
        compression=False,
    )
    try:
        with pytest.raises(GuardrailViolationError) as error:
            JevGuardrail(client=client).validate_input("local prompt")
        assert "sensitive-upstream-detail" not in str(error.value)
    finally:
        client.close()
