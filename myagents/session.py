"""
Session management for myagents.
"""


from typing import Protocol

from typing_extensions import override


from .agent import Agent, AgentRunContext
from .tools.core import Tool, ToolManager
from .tools.core.provider import ToolProvider
from .utils import truncate
from .events import *
from .history import History


class SessionBuildinToolProvider:

    _tools: list[Tool]

    def __init__(self, tools: list[Tool]):
        self._tools = tools

    def get_tools(self) -> list[Tool]:
        return self._tools


class Session:

    _history: History
    _agent: Agent
    _tool_manager: ToolManager

    def __init__(self, 
                 agent: Agent, 
                 tool_providers: list[ToolProvider],
                 middlewares: list['SessionMiddleware'] = []) -> None:

        self._history = History()
        self._agent = agent
        self._tool_manager = ToolManager(initial_providers=tool_providers)

        for middleware in middlewares:
            middleware.post_init(self)

    def add_tool_provider(self, provider: ToolProvider):
        self._tool_manager.add_provider(provider)

    def stream(self, prompt: str):
        # Append user turn
        self._history.append("user", prompt)
        yield UserPromptEvent(prompt=prompt)

        # Run the agent loop until it stops
        yield from self._agent.run(AgentRunContext(
            self._history,
            self._tool_manager, 
        ))

    def close(self):
        print("Exiting.")
        self._agent.close()


class SessionMiddleware(Protocol):

    def post_init(self, session: Session) -> None: ...
