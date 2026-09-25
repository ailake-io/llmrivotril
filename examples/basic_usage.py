"""Basic usage example for LLM-Rivotril."""

import os

from pydantic import BaseModel

from llmrivotril import Guardrail, MemoryStore, RivotrilAgent


class Summary(BaseModel):
    title: str
    bullet_points: list[str]


def main():
    agent = RivotrilAgent(
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
        memory=MemoryStore(retention_window=5),
        guardrails=[
            Guardrail(
                name="safe-prompts",
                disallowed_keywords=["password", "secret", "token"],
                max_tokens=300,
            )
        ],
    )

    response = agent.run(
        "Summarize the concept of LLM guardrails in two bullet points.",
        response_model=Summary,
    )
    print(response)


if __name__ == "__main__":
    main()
