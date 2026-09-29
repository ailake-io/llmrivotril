# Framework Integrations

*[Português](integrations.pt-BR.md)*

[← Back to README](../README.md)

`llmrivotril` has no framework-integration code baked into its core -- `RivotrilAgent`
is a single-agent, prompt-in/text-out orchestrator. The adapters in
`llmrivotril.integrations` bridge that orchestrator into four popular agent
frameworks, so every call those frameworks make still runs through
`llmrivotril`'s guardrails, PII redaction, RAG, and observability.

Each adapter is optional and only imported when you use it -- none of these
frameworks are core dependencies.

| Framework | Adapter | Interface implemented | Extra |
|---|---|---|---|
| CrewAI | `llmrivotril.integrations.crewai.CrewAILLM` | `BaseLLM.call()` | `llmrivotril[crewai]` |
| AG2 / pyautogen | `llmrivotril.integrations.autogen.RivotrilModelClient` | `ModelClient` protocol | none (structural typing) |
| LangChain / LangGraph | `llmrivotril.integrations.langchain.RivotrilChatModel` | `BaseChatModel._generate()` | `llmrivotril[langchain]` |
| Google ADK | `llmrivotril.integrations.adk.RivotrilLlm` | `BaseLlm.generate_content_async()` | `llmrivotril[adk]` |

Install one, several, or all:

```bash
pip install "llmrivotril[crewai]"
pip install "llmrivotril[langchain]"
pip install "llmrivotril[adk]"
pip install "llmrivotril[integrations]"  # all three + no-extra AutoGen support
```

## Why message history is flattened

Every one of these frameworks owns and resends the *entire* conversation
history on every call (CrewAI's task context, AutoGen's `params["messages"]`,
LangChain's message list, ADK's `llm_request.contents`). `RivotrilAgent.run()`
takes a single prompt string and manages its own continuity only through an
optional `memory=` store.

Each adapter reconciles this by flattening the framework's full message list
into one prompt block (`role: content` per line) before calling
`agent.run()`/`run_async()`. This is why **`RivotrilAgent`'s own `memory=`
should stay unset when it's wrapped by one of these adapters** -- the
orchestrator already resends full history every call, so giving the agent its
own memory store on top of that would track two divergent histories instead
of one.

## CrewAI

```python
from crewai import Agent
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.crewai import CrewAILLM

researcher = RivotrilAgent(model="gpt-4o-mini", api_key="...")
agent = Agent(
    role="Researcher",
    goal="...",
    backstory="...",
    llm=CrewAILLM(researcher),
)
```

Tool execution: pass callables via CrewAI's `available_functions=` and they're
registered with `llmrivotril`'s own `ToolRegistry`, so tool calls run under the
same guardrails as the rest of the turn. Bare `tools=` schemas without a
matching callable are forwarded for the provider to see but can't be
auto-executed by `RivotrilAgent`. `response_model=` (CrewAI's structured-output
param) is passed straight through to `agent.run(response_model=...)`.

Streaming is available through CrewAI's `stream_events()` protocol when the
LLM is configured with `stream=True`. Stop sequences remain unsupported
(`supports_stop_words()` reports `False` instead of silently ignoring them).

## AG2 / pyautogen

No extra install: `ModelClient` is a structural `Protocol`, satisfied by
matching methods rather than a subclass, so this adapter has zero dependency
on the `ag2`/`pyautogen` package itself.

```python
import autogen
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.autogen import RivotrilModelClient

worker = RivotrilAgent(model="gpt-4o-mini", api_key="...")
llm_config = {
    "config_list": [{"model": worker.model, "model_client_cls": "RivotrilModelClient"}],
}
assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
assistant.register_model_client(model_client_cls=RivotrilModelClient, agent=worker)
```

Targets the AG2/pyautogen `register_model_client` mechanism specifically --
**not** compatible with the newer `autogen-core`/AgentChat `ChatCompletionClient`
interface, which is async and has a different, richer protocol. Cost/usage
tracking is left at zero/empty (`llmrivotril` already tracks cost and tokens
through its own `metrics`/`track_costs`).

## LangChain / LangGraph

```python
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.langchain import RivotrilChatModel

writer = RivotrilAgent(model="gpt-4o-mini", api_key="...")
llm = RivotrilChatModel(agent=writer)
llm.invoke("Draft a release note.")
```

`RivotrilChatModel` is a real `langchain_core.language_models.chat_models.BaseChatModel`,
so it drops into any chain, agent, or LangGraph node that accepts a chat
model. Both sync (`_generate`) and async (`_agenerate`, calling
`agent.run_async` directly rather than LangChain's default thread-pool
wrapper) are implemented.

Streaming is available through `stream()`/`astream()`, and `bind_tools()`
forwards schemas and compatible LangChain tools to `RivotrilAgent`.
`tool_choice` is limited to `None` or `"auto"`.

## Google ADK

```python
from llmrivotril import RivotrilAgent
from llmrivotril.integrations.adk import RivotrilLlm
from google.adk.agents import LlmAgent

planner = RivotrilAgent(model="gemini-2.0-flash", api_key="...")
agent = LlmAgent(name="planner", model=RivotrilLlm(agent=planner), instruction="...")
```

`stream=True` yields one ADK response per agent stream chunk. Text, image,
audio, and file parts are translated to the normalized LLM-Rivotril content
format. Function-response parts still need explicit tool wiring by the ADK
orchestrator and are not executed by this adapter itself.

## Multi-agent setups (crews, group chats, graphs, multi-agent ADK trees)

`llmrivotril` has no multi-agent concept of its own -- it doesn't know it's
being used inside a crew, group chat, or graph. All routing, delegation, and
hand-off logic is entirely the orchestrating framework's job. What you need
to do on the `llmrivotril` side is purely mechanical:

- **One `RivotrilAgent` (wrapped by one adapter instance) per framework
  agent/role/node.** A five-member CrewAI crew needs five `CrewAILLM`
  instances wrapping five `RivotrilAgent` instances (they can share the same
  `model=`/config, or differ per role -- e.g. a cheaper model for a
  "summarizer" role, a stronger one for a "critic" role).
- **Guardrails can be shared or per-role.** Pass the same `Guardrail` list to
  every `RivotrilAgent` for a crew-wide policy, or different lists per role
  for role-specific restrictions.
- **Leave `memory=` unset on each `RivotrilAgent`** (see above) -- the
  orchestrator already resends full history per call.
- **Metrics aggregate automatically if you don't override them.** Every
  `RivotrilAgent` defaults to the same module-level `global_metrics` collector
  unless you pass `metrics=` explicitly, so a crew's token usage, cost, and
  guardrail blocks show up together in one dashboard/log by default. Pass the
  same explicit `MetricsCollector` instance to every agent if you want that
  guarantee to be explicit rather than implicit.

## Not verified against a live multi-agent run

These adapters are built from each framework's documented interface and
covered by mocked-agent unit tests (`tests/test_integrations.py`), plus a
one-time real-package smoke test against actually-installed `crewai`,
`langchain-core`, and `google-adk` (confirming the classes subclass/import
correctly and a single call round-trips). None of them have been exercised
inside a real multi-agent crew, group chat, or graph run. Please report
issues if something doesn't match your framework version's behavior.
