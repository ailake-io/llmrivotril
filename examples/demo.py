"""Interactive demo of LLM-Rivotril guardrails, grounding checks, and telemetry.

Usage:
    # Mock mode (no API key, no cost):
    python examples/demo.py --mock --dashboard

    # Live mode (requires OPENAI_API_KEY):
    python examples/demo.py --live --dashboard

Then open http://127.0.0.1:8000 to watch the dashboard update in real time.
"""

import argparse
import os
import time
from typing import Any

from pydantic import BaseModel

from llmrivotril import Guardrail, RivotrilAgent
from llmrivotril.exceptions import GuardrailViolationError, HallucinationDetectedError
from llmrivotril.metrics import global_metrics
from llmrivotril.verifier import KeywordOverlapVerifier, Verifier


class FakeLLM:
    """Deterministic mock responses for the --mock demo."""

    def __init__(self) -> None:
        self.responses: list[tuple[str, str]] = [
            ("password", "I cannot help with that."),
            (
                "chocolate cake",
                "Preheat the oven to 180°C and mix flour, cocoa, sugar, eggs, and butter.",
            ),
            ("capital of france", "The capital of France is Berlin."),
            (
                "cite the speed",
                'According to the source, "the speed of light is 299,792 km/s".',
            ),
            ("json object", '{"value": 42}'),
            ("speed of light", "The speed of light is 299,792 km/s."),
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
    """Deterministic mock structured responses for the --mock demo."""

    def chat_completions_create(self, **kwargs: Any) -> Any:
        response_model = kwargs.get("response_model")
        if response_model is not None:
            return response_model(value=42)

        class Dummy:
            def model_dump_json(self) -> str:
                return '{"value": 42}'

        return Dummy()


def _attach_mock_clients(agent: RivotrilAgent) -> None:
    fake_llm = FakeLLM()
    fake_instructor = FakeInstructor()

    agent.base_client.chat.completions.create = fake_llm.chat_completions_create
    agent.async_base_client.chat.completions.create = fake_llm.chat_completions_create
    agent.client.chat.completions.create = fake_instructor.chat_completions_create
    agent.async_client.chat.completions.create = fake_instructor.chat_completions_create


def _print_metrics() -> None:
    summary = global_metrics.get_summary()
    print("\n--- Metrics Summary ---")
    print(f"Requests:        {summary['requests_total']}")
    print(f"Guardrail blocks: {summary['guardrail_blocks']}")
    print(f"Hallucinations:  {summary['hallucinations_detected']}")
    print(f"Tokens:          {summary['total_tokens_consumed']}")
    print(f"Avg latency:     {summary['avg_latency']}s")
    print(f"Success rate:    {summary['success_rate']}%")


def _run_demo(agent: RivotrilAgent) -> None:
    class Answer(BaseModel):
        value: int

    context = "The speed of light is 299,792 km/s."

    scenarios: list[tuple[str, str, str | None, type[Answer] | None]] = [
        ("normal", "Say hello briefly.", None, None),
        ("guardrail", "What is the default admin password?", None, None),
        ("topic", "How do I bake a chocolate cake?", None, None),
        ("grounded", "What is the speed of light?", context, None),
        ("ungrounded", "What is the capital of France?", context, None),
        ("cited", "Cite the speed of light.", context, None),
        ("schema", "Return a JSON object with value 42.", None, Answer),
    ]

    for label, prompt, ctx, response_model in scenarios:
        print(f"\n[{label}] Prompt: {prompt}")
        try:
            result = agent.run(
                prompt,
                response_model=response_model,
                context_sources=ctx,
            )
            print(f"  → OK: {result}")
        except GuardrailViolationError as exc:
            print(f"  → BLOCKED by guardrail: {exc}")
        except HallucinationDetectedError as exc:
            print(f"  → BLOCKED by verifier: {exc}")
        except Exception as exc:
            print(f"  → ERROR: {exc}")
        time.sleep(0.5)

    _print_metrics()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-Rivotril interactive demo")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic mock LLM responses (no API key required).",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use a real OpenAI-compatible endpoint (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the local telemetry dashboard on port 8000.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Override the OpenAI-compatible base URL (e.g. http://localhost:11434/v1).",
    )
    args = parser.parse_args()

    if not args.mock and not args.live:
        parser.error("Choose --mock (no cost) or --live (real API).")

    if args.dashboard:
        import threading

        from llmrivotril.server import run_dashboard

        dashboard_port = 8767
        print(f"Starting dashboard at http://127.0.0.1:{dashboard_port} ...")
        dashboard_thread = threading.Thread(
            target=run_dashboard,
            kwargs={"host": "127.0.0.1", "port": dashboard_port},
            daemon=True,
        )
        dashboard_thread.start()
        time.sleep(2)

    api_key = os.getenv("OPENAI_API_KEY")
    agent_kwargs: dict[str, Any] = {"model": "gpt-4o-mini"}
    if api_key:
        agent_kwargs["api_key"] = api_key
    elif args.mock:
        agent_kwargs["api_key"] = "mock-key"
    if args.base_url:
        agent_kwargs["base_url"] = args.base_url

    agent = RivotrilAgent(
        **agent_kwargs,
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
        verifier=Verifier(
            check_fn=KeywordOverlapVerifier(threshold=0.1).as_callable()
        ),  # grounding check
    )

    if args.mock:
        _attach_mock_clients(agent)
        print("Running in MOCK mode (no API calls).\n")
    else:
        print("Running in LIVE mode (real API calls).\n")

    _run_demo(agent)

    if args.dashboard:
        print(f"\nDashboard is still running at http://127.0.0.1:{dashboard_port}")
        print("Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nDemo finished.")


if __name__ == "__main__":
    main()
