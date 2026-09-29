# Guardrails & Safety

*[Português](guardrails-and-safety.pt-BR.md)*

[← Back to README](../README.md)

The basic `Guardrail` class (keywords, allowed topics, JSON schema
validation, output size) is covered in the main [README](../README.md#quick-start).
This page covers the rest.

## Semantic Guardrails

Use dense embeddings to accept semantically related prompts even when they do not contain exact topic keywords (requires `llmrivotril[semantic]`):

```python
from llmrivotril import SemanticTopicGuardrail

guardrail = SemanticTopicGuardrail(
    name="semantic-topics",
    allowed_topics=["machine learning", "software engineering"],
    similarity_threshold=0.5,
)

agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

## Moderation Guardrail

The keyword/regex/topic guardrails above are cheap local heuristics, not a
moderation system. `ModerationGuardrail` calls OpenAI's moderation endpoint
for adversarial/unsafe content they aren't meant to catch (no extra install
needed, `openai` is already a core dependency):

```python
from llmrivotril import ModerationGuardrail

guardrail = ModerationGuardrail(api_key=os.getenv("OPENAI_API_KEY"))
agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

It fails closed by default: if the moderation call itself errors (network,
rate limit), the request is blocked rather than silently let through. Pass
`fail_open=True` to let requests through instead when the moderation call
fails, or `check_input=False`/`check_output=False` to only check one side.

The model-based grounding verifier also exposes the same explicit policy. It
fails closed by default, which is appropriate for safety-sensitive or
regulated responses; opt into availability over strict grounding with
`ModelBasedFaithfulnessVerifier(fail_open=True)`.

## PII Redaction

Redact sensitive information from inputs and outputs before they reach the LLM
or logs:

```python
from llmrivotril import RivotrilAgent

agent = RivotrilAgent(
    api_key="sk-...",
    redact_pii=True,
)

agent.run("My email is alice@example.com")
```

Redaction runs before a response is written to `cache=` (`InMemoryCache`,
`DiskCache`, or `RedisCache`), so raw PII is never persisted at rest when
`redact_pii=True`.

You can also use `PIIRedactor` directly to scan or sanitize text:

```python
from llmrivotril import PIIRedactor

redactor = PIIRedactor()
text = redactor.redact("CPF: 123.456.789-09")
```

## Token Budget

Cap prompt and per-session token usage to avoid runaway costs:

```python
agent = RivotrilAgent(
    api_key="sk-...",
    max_prompt_tokens=2000,
    max_session_tokens=10000,
)
```

`max_prompt_tokens` rejects a single prompt that exceeds the limit.
`max_session_tokens` tracks cumulative token use across runs on the same agent instance.
Both raise `TokenBudgetExceededError` when exceeded.
