"""Side-by-side comparison: plain LLM call vs. LLM-Rivotril.

Usage:
    python examples/comparison.py --mock
    python examples/comparison.py --api-key sk-... --base-url https://api.openai.com/v1 \\
        --model gpt-4o-mini
    OPENROUTER_API_KEY=sk-or-... python examples/comparison.py \\
        --base-url https://openrouter.ai/api/v1 --model openai/gpt-4o-mini

The script runs the same scenarios twice:
1. "Without llmrivotril" — direct LLM call, no guardrails, no grounding check.
2. "With llmrivotril" — same calls wrapped by RivotrilAgent with guardrails,
   grounding verification, JSON schema validation, and telemetry.

In real (non-mock) mode, the "without" side still runs the same grounding
check the "with" side uses internally (KeywordOverlapVerifier) -- not
because a plain call would do that on its own, but to show concretely what
"with llmrivotril" catches automatically that a plain pipeline would have to
remember to implement and call itself.
"""

import argparse
import os
from typing import Any

from pydantic import BaseModel

from llmrivotril import Guardrail, RivotrilAgent
from llmrivotril.exceptions import GuardrailViolationError, HallucinationDetectedError
from llmrivotril.metrics import global_metrics
from llmrivotril.providers import BaseProvider, OpenAIProvider, ProviderResponse
from llmrivotril.verifier import KeywordOverlapVerifier, Verifier


class _MockProvider(BaseProvider):
    """Deterministic mock provider for the comparison demo."""

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

    def _lookup(self, messages: list[dict[str, str]]) -> str:
        prompt = messages[-1].get("content", "").lower()
        for key, text in self.responses:
            if key in prompt:
                return text
        return "I'm a mock LLM."

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        if response_model is not None:
            return ProviderResponse(structured=response_model(value=42))
        return ProviderResponse(content=self._lookup(messages))

    async def acomplete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        response_model: type[BaseModel] | None = None,
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        return self.complete(messages, model, response_model=response_model, tools=tools, **kwargs)


class PlainLLM:
    """Simulates a raw LLM call without any safety layer."""

    def __init__(self, provider: BaseProvider, model: str) -> None:
        self.provider = provider
        self.model = model

    def call(self, prompt: str) -> str:
        response = self.provider.complete(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.text


class Answer(BaseModel):
    value: int


def _without_package(
    scenarios: list[tuple[str, str, str | None]], provider: BaseProvider, model: str, mock: bool
) -> None:
    print("\n" + "=" * 60)
    print("WITHOUT llmrivotril — direct LLM call")
    print("=" * 60)

    llm = PlainLLM(provider, model)
    grounding_check = KeywordOverlapVerifier(threshold=0.1)
    blocks = 0
    for label, prompt, ctx in scenarios:
        response = llm.call(prompt)
        print(f"\n[{label}] Prompt: {prompt}")
        print(f"  → Response: {response}")

        if "password" in prompt.lower():
            print("  ⚠️  Risky prompt (would-be secret/policy leak) went straight to the model")
            blocks += 1

        if not mock and ctx is not None:
            # Same check the "with" side runs automatically -- shown here to
            # make clear it doesn't happen unless you call it yourself.
            if not grounding_check.verify(response, ctx):
                print("  ⚠️  Response isn't grounded in the provided context (not checked)")
                blocks += 1
        elif mock and "capital of france" in prompt.lower() and "Berlin" in response:
            print("  ⚠️  Hallucination: wrong capital")
            blocks += 1

    print(f"\nIssues an unguarded pipeline would have missed: {blocks}")
    print("No telemetry, no schema validation, no memory, no audit trail.")


def _with_package(
    scenarios: list[tuple[str, str, str | None]], provider: BaseProvider, model: str
) -> None:
    print("\n" + "=" * 60)
    print("WITH llmrivotril — guarded, grounded, observable")
    print("=" * 60)

    agent = RivotrilAgent(
        model=model,
        provider=provider,
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
    parser.add_argument(
        "--api-key", default=None, help="Real API key (or set OPENAI_API_KEY/OPENROUTER_API_KEY)."
    )
    parser.add_argument(
        "--base-url", default=None, help="OpenAI-compatible base URL (e.g. OpenRouter)."
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model name to use in real mode.")
    args = parser.parse_args()

    api_key = (
        args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
    )
    if not args.mock and not api_key:
        parser.error(
            "Real mode needs --api-key or OPENAI_API_KEY/OPENROUTER_API_KEY set. "
            "Use --mock instead?"
        )

    if args.mock:
        provider: BaseProvider = _MockProvider()
        model = "gpt-4o-mini"
    else:
        provider = OpenAIProvider(api_key=api_key, base_url=args.base_url)
        model = args.model

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

    _without_package(scenarios, provider, model, args.mock)
    _with_package(scenarios, provider, model)

    print("\n" + "=" * 60)
    print("Key takeaway")
    print("=" * 60)
    print(
        "Without llmrivotril, harmful, ungrounded, or malformed responses are returned and "
        "only noticed manually afterwards -- if you remember to check at all.\n"
        "With llmrivotril, they are blocked at runtime, validated against schemas, "
        "and recorded as structured telemetry you can audit in the dashboard."
    )


if __name__ == "__main__":
    main()
