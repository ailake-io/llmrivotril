"""AG2 (pyautogen) custom ModelClient adapter.

Wraps a ``RivotrilAgent`` as an AG2 ``ModelClient`` -- the
``autogen.oai.client.ModelClient`` structural protocol. No import of, or
dependency on, the ag2/pyautogen package is required here: it's a
``Protocol`` satisfied by matching methods alone, not by subclassing.

Not compatible with the newer `autogen-core`/AgentChat ``ChatCompletionClient``
interface, which is async and has a different, richer protocol -- this
targets the AG2/pyautogen ``register_model_client`` mechanism only.

AG2 resends the full conversation on every ``create()`` call via
``params["messages"]``; this adapter flattens it into one prompt via
``flatten_messages`` rather than relying on ``RivotrilAgent``'s own
``memory=`` store. Cost/usage tracking is left at zero/empty -- llmrivotril
already tracks cost and tokens through its own ``metrics``/``track_costs``.

Usage::

    agent = RivotrilAgent(...)
    llm_config = {
        "config_list": [{"model": agent.model, "model_client_cls": "RivotrilModelClient"}],
    }
    assistant = autogen.AssistantAgent("assistant", llm_config=llm_config)
    assistant.register_model_client(model_client_cls=RivotrilModelClient, agent=agent)
"""

from dataclasses import dataclass
from typing import Any

from ..agent import RivotrilAgent
from ._common import flatten_messages


@dataclass
class _Message:
    content: str | None


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    """Satisfies AG2's ``ModelClientResponseProtocol``."""

    choices: list[_Choice]
    model: str


class RivotrilModelClient:
    """AG2 ``ModelClient`` backed by a ``RivotrilAgent``."""

    def __init__(self, config: dict[str, Any], agent: RivotrilAgent) -> None:
        self.agent = agent

    def create(self, params: dict[str, Any]) -> _Response:
        prompt = flatten_messages(params.get("messages", []))
        text = self.agent.run(prompt)
        return _Response(choices=[_Choice(message=_Message(content=text))], model=self.agent.model)

    def message_retrieval(self, response: _Response) -> list[str]:
        return [choice.message.content or "" for choice in response.choices]

    def cost(self, response: _Response) -> float:
        return 0.0

    @staticmethod
    def get_usage(response: _Response) -> dict[str, Any]:
        return {}
