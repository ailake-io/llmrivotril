"""Side-by-side comparison: plain LLM call vs. LLM-Rivotril.

Usage:
    python examples/comparison.py --mock

The script runs the same scenarios twice:
1. "Without llmrivotril" — direct mock LLM call, no guardrails, no grounding check.
2. "With llmrivotril" — same calls wrapped by RivotrilAgent with guardrails,
   grounding verification, JSON schema validation, and telemetry.
"""

import argparse
from typing import Any

from pydantic import BaseModel

from llmrivotril import Guardrail, RivotrilAgent
from llmrivotril.exceptions import GuardrailViolationError, HallucinationDetectedError
from llmrivotril.metrics import global_metrics
from llmrivotril.verifier import KeywordOverlapVerifier, Verifier


class FakeLLM:
    """Deterministic mock responses."""

    def __init__(self) -> None:
        self.responses: list[tuple[str, str]] = [
            ("password", "The default admin password is 'admin123'."),
            (
                "chocolate cake",
                "Preheat the oven to 180°C and mix flour, cocoa, sugar, eggs, and butter.",
            ),
            ("speed of light", "The speed of light is 299,792 km/s."),
            ("capital of france", "The capital of France is Berlin."),
            (
                "cite the speed",
                'According to the source, "the speed of light is 299,792 km/s".',
            ),
            ("json object", '{"value": 42}'),
            ("hello", "Hello! How can I help you today?"),
        ]

    def chat_completions_create(self, *, model: str, messages: list[dict[str, str]]) -> Any:
        prompt = messages[-1]["content"].lower()
        response_text = "I'm a mock LLM."
        for key, text in self.responses:
            if key in prompt:
                response_text = text
                break

        class Choice:
            class Message:
                content = response_text

            message = Message()

        class Completion:
            choices = [Choice()]

        return Completion()


class FakeInstructor:
    def chat_completions_create(self, **kwargs: Any) -> Any:
        response_model = kwargs.get("response_model")
        if response_model is not None:
            return response_model(value=42)

        class Dummy:
            def model_dump_json(self) -> str:
                return '{"value": 42}'

        return Dummy()


class PlainLLM:
    """Simulates a raw LLM call without any safety layer."""

    def __init__(self) -> None:
        self.fake = FakeLLM()

    def call(self, prompt: str) -> str:
        completion = self.fake.chat_completions_create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content


class Answer(BaseModel):
    value: int


def _without_package(scenarios: list[tuple[str, str, str | None]]) -> None:
    print("\n" + "=" * 60)
    print("WITHOUT llmrivotril — direct LLM call")
    print("=" * 60)

    llm = PlainLLM()
    blocks = 0
    for label, prompt, _ctx in scenarios:
        response = llm.call(prompt)
        print(f"\n[{label}] Prompt: {prompt}")
        print(f"  → Response: {response}")
        # Manual sanity check after the fact
        if "password" in prompt.lower():
            print("  ⚠️  Policy violation: leaked a password")
            blocks += 1
        if "capital of france" in prompt.lower() and "Berlin" in response:
            print("  ⚠️  Hallucination: wrong capital")
            blocks += 1

    print(f"\nManual issues detected after the fact: {blocks}")
    print("No telemetry, no schema validation, no memory, no audit trail.")


def _with_package(scenarios: list[tuple[str, str, str | None]]) -> None:
    print("\n" + "=" * 60)
    print("WITH llmrivotril — guarded, grounded, observable")
    print("=" * 60)

    agent = RivotrilAgent(
        api_key="mock-key",
        guardrails=[
            Guardrail(
                name="content-safety",
                allowed_topics=[
                    "AI",
                    "physics",
                    "speed",
                    "light",
                    "france",
                    "cake",
                    "json",
                    "hello",
                ],
                disallowed_keywords=["password", "secret", "token"],
                max_tokens=200,
            )
        ],
        verifier=Verifier(check_fn=KeywordOverlapVerifier(threshold=0.1).as_callable()),
    )

    fake_llm = FakeLLM()
    fake_instructor = FakeInstructor()
    agent.base_client.chat.completions.create = fake_llm.chat_completions_create
    agent.client.chat.completions.create = fake_instructor.chat_completions_create

    for label, prompt, ctx in scenarios:
        print(f"\n[{label}] Prompt: {prompt}")
        try:
            if label == "schema":
                result = agent.run(prompt, response_model=Answer, context_sources=ctx)
            else:
                result = agent.run(prompt, context_sources=ctx)
            print(f"  → Response: {result}")
        except GuardrailViolationError as exc:
            print(f"  → BLOCKED by guardrail: {exc}")
        except HallucinationDetectedError as exc:
            print(f"  → BLOCKED by verifier: {exc}")
        except Exception as exc:
            print(f"  → ERROR: {exc}")

    summary = global_metrics.get_summary()
    print("\nTelemetry summary:")
    print(f"  Requests:          {summary['requests_total']}")
    print(f"  Guardrail blocks:  {summary['guardrail_blocks']}")
    print(f"  Hallucinations:    {summary['hallucinations_detected']}")
    print(f"  Tokens consumed:   {summary['total_tokens_consumed']}")
    print(f"  Success rate:      {summary['success_rate']}%")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare plain LLM calls with llmrivotril")
    parser.add_argument("--mock", action="store_true", help="Use deterministic mock LLM responses.")
    args = parser.parse_args()

    if not args.mock:
        parser.error("This demo currently only supports --mock mode (no API cost).")

    context = "The speed of light is 299,792 km/s."
    scenarios: list[tuple[str, str, str | None]] = [
        ("normal", "Say hello briefly.", None),
        ("guardrail", "What is the default admin password?", None),
        ("topic", "How do I bake a chocolate cake?", None),
        ("grounded", "What is the speed of light?", context),
        ("ungrounded", "What is the capital of France?", context),
        ("cited", "Cite the speed of light.", context),
        ("schema", "Return a JSON object with value 42.", None),
    ]

    _without_package(scenarios)
    _with_package(scenarios)

    print("\n" + "=" * 60)
    print("Key takeaway")
    print("=" * 60)
    print(
        "Without llmrivotril, harmful or hallucinated responses are returned and "
        "only noticed manually afterwards.\n"
        "With llmrivotril, they are blocked at runtime, validated against schemas, "
        "and recorded as structured telemetry you can audit in the dashboard."
    )


if __name__ == "__main__":
    main()
