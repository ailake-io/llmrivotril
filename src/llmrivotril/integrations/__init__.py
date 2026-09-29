"""Optional adapters exposing ``RivotrilAgent`` to other agent frameworks.

Each submodule requires its own optional dependency (or none) and is only
imported when explicitly used, so none of these are hard dependencies of
the core package:

- ``llmrivotril.integrations.crewai`` -- CrewAI ``BaseLLM``.
  ``pip install "llmrivotril[crewai]"``
- ``llmrivotril.integrations.autogen`` -- AG2/pyautogen ``ModelClient``.
  Pure structural protocol, no extra package required.
- ``llmrivotril.integrations.langchain`` -- LangChain ``BaseChatModel``.
  ``pip install "llmrivotril[langchain]"``
- ``llmrivotril.integrations.adk`` -- Google ADK ``BaseLlm``.
  ``pip install "llmrivotril[adk]"``

All three flatten the orchestrator's full message history into a single
prompt string before calling ``RivotrilAgent.run()``/``run_async()``, since
each framework owns and resends that history itself on every call, while
``RivotrilAgent`` expects one prompt and manages its own continuity only
when constructed with a ``memory=`` store.
"""
