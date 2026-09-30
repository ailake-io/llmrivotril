"""Tests for the CrewAI/AutoGen/LangChain adapters in ``llmrivotril.integrations``."""

from unittest.mock import MagicMock

import pytest

from llmrivotril.agent import RivotrilAgent


def _fake_agent(response: str = "hello from rivotril") -> RivotrilAgent:
    agent = MagicMock(spec=RivotrilAgent)
    agent.model = "gpt-4o-mini"
    agent.run.return_value = response
    agent.run_async = MagicMock(return_value=response)
    return agent


class TestCrewAI:
    def test_call_flattens_messages_and_delegates_to_agent_run(self) -> None:
        pytest.importorskip("crewai")
        from llmrivotril.integrations.crewai import CrewAILLM

        agent = _fake_agent()
        llm = CrewAILLM(agent)

        result = llm.call(
            [{"role": "system", "content": "be nice"}, {"role": "user", "content": "hi"}]
        )

        assert result == "hello from rivotril"
        prompt = agent.run.call_args.args[0]
        assert "System: be nice" in prompt
        assert "User: hi" in prompt

    def test_available_functions_take_priority_over_bare_tool_schemas(self) -> None:
        pytest.importorskip("crewai")
        from llmrivotril.integrations.crewai import CrewAILLM

        agent = _fake_agent()
        llm = CrewAILLM(agent)

        def my_tool(x: int) -> int:
            return x

        llm.call("hi", tools=[{"type": "function"}], available_functions={"my_tool": my_tool})

        assert agent.run.call_args.kwargs["tools"] == [my_tool]

    def test_supports_stop_words_is_false(self) -> None:
        pytest.importorskip("crewai")
        from llmrivotril.integrations.crewai import CrewAILLM

        assert CrewAILLM(_fake_agent()).supports_stop_words() is False

    def test_missing_crewai_raises_clear_import_error(self) -> None:
        with pytest.MonkeyPatch.context() as mp:
            import sys

            mp.setitem(sys.modules, "crewai", None)
            mp.setitem(sys.modules, "crewai.llms.base_llm", None)
            sys.modules.pop("llmrivotril.integrations.crewai", None)
            with pytest.raises(ImportError, match="llmrivotril\\[crewai\\]"):
                import llmrivotril.integrations.crewai  # noqa: F401
            sys.modules.pop("llmrivotril.integrations.crewai", None)


class TestAutoGen:
    """Targets AG2 1.x's Agent(config=...) extension point.

    AG2<1.0's AssistantAgent/register_model_client mechanism (what this
    adapter used to target) no longer exists in any currently-installable
    package -- see docs/integrations.md.
    """

    @pytest.mark.asyncio
    async def test_llm_client_flattens_messages_and_wraps_response(self) -> None:
        pytest.importorskip("ag2")
        from ag2.events import HumanMessage

        from llmrivotril.integrations.autogen import RivotrilLLMClient

        agent = _fake_agent()

        async def fake_run_async(prompt: str) -> str:
            fake_run_async.seen_prompt = prompt  # type: ignore[attr-defined]
            return "hello from rivotril"

        agent.run_async = fake_run_async
        client = RivotrilLLMClient(agent)

        response = await client(
            [HumanMessage("hi")],
            context=None,
            tools=[],
            response_schema=None,
            serializer=None,
        )

        assert response.content == "hello from rivotril"
        assert "hi" in fake_run_async.seen_prompt  # type: ignore[attr-defined]

    def test_model_config_exposes_provider_and_model(self) -> None:
        pytest.importorskip("ag2")
        from ag2.config.config import ModelProvider

        from llmrivotril.integrations.autogen import RivotrilModelConfig

        agent = _fake_agent()
        agent.provider = MagicMock()
        agent.provider.name = "openai"
        config = RivotrilModelConfig(agent)

        assert config.model == "gpt-4o-mini"
        assert config.provider == ModelProvider.OPENAI
        copied = config.copy()
        assert isinstance(copied, RivotrilModelConfig)
        assert copied.agent is agent

    def test_model_config_unknown_provider_falls_back_to_openai(self) -> None:
        pytest.importorskip("ag2")
        from ag2.config.config import ModelProvider

        from llmrivotril.integrations.autogen import RivotrilModelConfig

        agent = _fake_agent()
        agent.provider = MagicMock()
        agent.provider.name = "cohere"  # not in AG2's ModelProvider enum
        config = RivotrilModelConfig(agent)

        assert config.provider == ModelProvider.OPENAI

    @pytest.mark.asyncio
    async def test_model_config_create_returns_working_client(self) -> None:
        pytest.importorskip("ag2")
        from ag2.events import HumanMessage

        from llmrivotril.integrations.autogen import RivotrilLLMClient, RivotrilModelConfig

        agent = _fake_agent()

        async def fake_run_async(prompt: str) -> str:
            return "hello from rivotril"

        agent.run_async = fake_run_async
        client = RivotrilModelConfig(agent).create()

        assert isinstance(client, RivotrilLLMClient)
        response = await client(
            [HumanMessage("hi")], context=None, tools=[], response_schema=None, serializer=None
        )
        assert response.content == "hello from rivotril"


class TestLangChain:
    def test_generate_flattens_messages_and_wraps_response(self) -> None:
        pytest.importorskip("langchain_core")
        from langchain_core.messages import HumanMessage, SystemMessage

        from llmrivotril.integrations.langchain import RivotrilChatModel

        agent = _fake_agent()
        model = RivotrilChatModel(agent=agent)

        result = model._generate([SystemMessage(content="be nice"), HumanMessage(content="hi")])

        assert result.generations[0].message.content == "hello from rivotril"
        prompt = agent.run.call_args.args[0]
        assert "System: be nice" in prompt
        assert "Human: hi" in prompt

    @pytest.mark.asyncio
    async def test_agenerate_calls_run_async(self) -> None:
        pytest.importorskip("langchain_core")
        from langchain_core.messages import HumanMessage

        from llmrivotril.integrations.langchain import RivotrilChatModel

        agent = _fake_agent()

        async def fake_run_async(prompt: str) -> str:
            return "async response"

        agent.run_async = fake_run_async
        model = RivotrilChatModel(agent=agent)

        result = await model._agenerate([HumanMessage(content="hi")])

        assert result.generations[0].message.content == "async response"

    def test_llm_type(self) -> None:
        pytest.importorskip("langchain_core")
        from llmrivotril.integrations.langchain import RivotrilChatModel

        assert RivotrilChatModel(agent=_fake_agent())._llm_type == "rivotril"

    def test_stream_delegates_chunks_to_agent(self) -> None:
        pytest.importorskip("langchain_core")
        from langchain_core.messages import HumanMessage

        from llmrivotril.integrations.langchain import RivotrilChatModel

        agent = _fake_agent()
        agent.run_stream.return_value = iter(["hello", " world"])
        model = RivotrilChatModel(agent=agent)

        chunks = list(model._stream([HumanMessage(content="hi")]))

        assert [chunk.message.content for chunk in chunks] == ["hello", " world"]
        assert "Human: hi" in agent.run_stream.call_args.args[0]

    @pytest.mark.asyncio
    async def test_astream_delegates_chunks_to_agent(self) -> None:
        pytest.importorskip("langchain_core")
        from langchain_core.messages import HumanMessage

        from llmrivotril.integrations.langchain import RivotrilChatModel

        agent = _fake_agent()

        async def fake_stream(prompt: str):
            yield "hello"
            yield " world"

        agent.run_stream_async = fake_stream
        model = RivotrilChatModel(agent=agent)

        chunks = [chunk async for chunk in model._astream([HumanMessage(content="hi")])]

        assert [chunk.message.content for chunk in chunks] == ["hello", " world"]

    def test_bind_tools_forwards_tools_to_agent(self) -> None:
        pytest.importorskip("langchain_core")
        from langchain_core.messages import HumanMessage

        from llmrivotril.integrations.langchain import RivotrilChatModel

        def lookup(city: str) -> str:
            return city

        agent = _fake_agent()
        model = RivotrilChatModel(agent=agent).bind_tools([lookup])
        model._generate([HumanMessage(content="weather?")])

        assert agent.run.call_args.kwargs["tools"] == [lookup]


class TestADK:
    @pytest.mark.asyncio
    async def test_generate_content_async_flattens_contents_and_wraps_response(self) -> None:
        pytest.importorskip("google.adk")
        from google.genai import types

        from llmrivotril.integrations.adk import RivotrilLlm

        agent = _fake_agent()

        async def fake_run_async(prompt: str) -> str:
            fake_run_async.seen_prompt = prompt  # type: ignore[attr-defined]
            return "hello from rivotril"

        agent.run_async = fake_run_async
        llm = RivotrilLlm(agent=agent)

        llm_request = MagicMock()
        llm_request.contents = [
            types.Content(role="user", parts=[types.Part(text="hi")]),
        ]

        responses = [response async for response in llm.generate_content_async(llm_request)]

        assert len(responses) == 1
        assert responses[0].content.parts[0].text == "hello from rivotril"
        assert "User: hi" in fake_run_async.seen_prompt  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_generate_content_async_streams_when_requested(self) -> None:
        pytest.importorskip("google.adk")
        from google.genai import types

        from llmrivotril.integrations.adk import RivotrilLlm

        agent = _fake_agent()

        async def fake_stream(prompt: str):
            yield "hello"
            yield " world"

        agent.run_stream_async = fake_stream
        llm = RivotrilLlm(agent=agent)
        llm_request = MagicMock()
        llm_request.contents = [types.Content(role="user", parts=[types.Part(text="hi")])]

        responses = [
            response async for response in llm.generate_content_async(llm_request, stream=True)
        ]

        assert [response.content.parts[0].text for response in responses] == ["hello", " world"]
