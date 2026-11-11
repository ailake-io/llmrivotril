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

## Jev Triage Guardrail

[Jev](https://docs.typesafe.ai/go/system-one) returns typed probabilities.
The optional check defaults to TypeSafe's hosted API, and can also use a local
System One-compatible server such as
[`local-jev`](https://github.com/amithgc/local-jev). It stays disabled unless
explicitly added.
`JevGuardrail` asks one yes/no question and blocks when the probability is
`>= threshold` (default `0.7`). Install `pip install llmrivotril[jev]`; the
hosted endpoint needs `TYPESAFE_API_KEY`, while a local endpoint needs its URL
and a placeholder key:

```python
from llmrivotril import JevGuardrail

guardrail = JevGuardrail(question="The text asks for medical dosage advice", threshold=0.8)
agent = RivotrilAgent(api_key=os.getenv("OPENAI_API_KEY"), guardrails=[guardrail])
```

Same `fail_open` / `check_input` / `check_output` flags as `ModerationGuardrail`.

To use `local-jev`, start its server on `127.0.0.1:8765` following its README
and select an installed local model. Pass the URL and a dummy local key (the SDK
requires a key; local inference has no per-call charge):

```python
guardrail = JevGuardrail(
    base_url="http://127.0.0.1:8765",
    api_key="local",
    question="The text requests disallowed content",
)
```

Use the same `base_url`, `api_key` and `model` with `JevContextSelector`. The
local server selects an installed model by default; set `model=` to choose a
different one. It downloads model weights on first use; inference then runs on
your machine. It still uses disk, memory, CPU/GPU and electricity. The server
and model weights have separate licenses; check the license for the model you
choose. Its accuracy and calibration differ from hosted Jev, so evaluate it on
your own cases.

Ollama 0.35+ also exposes this API directly. Pull a decision model (for example,
`ollama pull tev1:0.8b`), then configure the adapter like this:

```python
guardrail = JevGuardrail(
    base_url="http://127.0.0.1:11434",
    api_key="ollama",
    model="tev1:0.8b",
)
```

Ollama's `/v1/systemone` route is local and does not charge per call. Model
quality varies: in our local smoke test, `tev1:0.8b` allowed a benign question
and blocked a harmful instruction at `0.89`, but retained all turns in one
context-selection example. This small test is not an accuracy or calibration
benchmark.

**Semantic output rules.** `output_rules=["The reply promises a refund"]` adds
one Noul question per rule to the same Jev call on output. Use it for meaning
(a field's value contradicts another); structural JSON checks stay
deterministic in `Guardrail(json_schema=...)`.

**Context selection.** `JevContextSelector` scores conversation groups, retaining
user/assistant pairs together and always keeping the latest group and system
instructions. Failures or invalid probabilities keep the full history. It
changes the prompt, not persisted memory. Token-budget checks still consider
the full stored context conservatively, and do not trigger an extra selection:

```python
from llmrivotril import JevContextSelector, MemoryStore

agent = RivotrilAgent(memory=MemoryStore(select=JevContextSelector()))
```

**Parallel guardrails.** `RivotrilAgent(parallel_guardrails=True,
guardrail_workers=4)` runs input checks with a bounded pool. All finish before
generation; one failing check prevents the provider call. Output checks remain
sequential. Custom guardrails used in parallel must support concurrent calls.
Jev checks and context selection run in worker threads in async APIs; an
in-flight synchronous SDK call cannot be forcibly cancelled by Python.
Parallel mode may send text to a remote checker even when a local guardrail
rejects it. Keep sequential mode if local rejection must prevent remote checks.

Validated with `typesafe-sdk==0.7.5`, behind `llmrivotril[jev]`; it is not a core
dependency. New clients use a 10-second timeout and no SDK retries. Configure
`timeout=` or inject an owned `client=` (close injected clients yourself).
Thresholds and probabilities must be finite and within `[0, 1]`. Errors block
by default; `fail_open=True` is explicit. Credentials are excluded from repr
and serialization; wrapper errors omit upstream details.

With the hosted endpoint, checked text goes to TypeSafe; context selection also
sends history and query, potentially adding a call on every turn, including
cache hits. External calls are outside `provider_usage` and the estimated
session cost limit. With `local-jev`, content stays on the local host after
weights are downloaded. SDK debug logging may include bodies. llmrivotril's
local tests validate SDK serialization and policies, not model quality or
latency.

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
