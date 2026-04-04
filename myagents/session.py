"""
Session management for myagents.
"""


from json import tool
from mailbox import Message
from typing import Literal, Protocol, Sequence

from anthropic.types import Message
from typing_extensions import override

from .tools.core.capability import CapabilityRule


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


class _AgentRunContext(AgentRunContext):

    _session: 'Session'

    def __init__(self, session: 'Session') -> None:
        self._session = session

    @override
    def get_inputs(self) -> Sequence[dict]:
        """get inputs
        """
        return self._session._history.messages

    @override
    def append_message(self, role: Literal["user", "assistant"], content) -> None:
        """append message to history
        """
        self._session.append_message(role, content)

    @override
    def resolve_tools(self, allowed_capabilities: list[CapabilityRule]) -> tuple[Tool, ...]:
        """resolve tools by allowed capabilities
        """
        return self._session._tool_manager.resolve_tools_by_capabilities(allowed_capabilities)

    @override
    def use_tool(self, allowed_capabilities: list[CapabilityRule], tool_to_use: str, **tool_kwargs) -> str:
        """use tools subjected to allowed capabilities
        """
        return self._session._tool_manager.execute(allowed_capabilities, tool_to_use, **tool_kwargs)

    @override
    def post_step(self, resp: Message, loop_completed: bool) -> tuple[list[Event], bool]:
        return self._session._post_step(resp, loop_completed)


class Session:

    _history: History
    _agent: Agent
    _tool_manager: ToolManager
    _middlewares: list['SessionMiddleware']

    def __init__(self, 
                 agent: Agent, 
                 tool_providers: list[ToolProvider],
                 middlewares: list['SessionMiddleware'] = []) -> None:

        self._history = History()
        self._agent = agent
        self._tool_manager = ToolManager(initial_providers=tool_providers)
        self._middlewares = middlewares

        self._post_init()

    def _post_init(self):
        for middleware in self._middlewares:
            middleware.post_session_init(self)

    def _post_step(self, resp: Message, loop_completed: bool) -> tuple[list[Event], bool]:
        all_events = []
        for middleware in self._middlewares:
            events, completed = middleware.post_agent_step(self, resp, loop_completed)
            if events:
                all_events.extend(events)
            if not completed:
                # the loop will continue if any middleware want it to be
                loop_completed = False
        return all_events, loop_completed

    def add_tool_provider(self, provider: ToolProvider):
        self._tool_manager.add_provider(provider)

    def append_message(self, role: Literal["user", "assistant"], content):
        self._history.append(role, content)

    def stream(self, prompt: str):
        # Append user turn
        self._history.append("user", prompt)
        yield UserPromptEvent(prompt=prompt)

        # Run the agent loop until it stops
        yield from self._agent.run(_AgentRunContext(self))

    def close(self):
        print("Exiting.")
        self._agent.close()


class SessionMiddleware(Protocol):

    def post_session_init(self, session: 'Session') -> None: ...

    def post_agent_step(self, session: 'Session', resp: Message, loop_completed: bool) -> tuple[list[Event], bool]:
        return [], loop_completed

