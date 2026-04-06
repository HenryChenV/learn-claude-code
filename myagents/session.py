"""
Session management for myagents.
"""


from mailbox import Message
from typing import Generator, Literal, Optional, Protocol, Self, Sequence, Any

from anthropic.types import Message
from typing_extensions import override

from .ids import HierarchicalID
from .tools.core.tool import FunctionTool

from .tools.core.capability import CapabilityRule


from .agent import Agent, AgentRunContext
from .tools.core import Tool, ToolManager
from .tools.core.provider import ToolProvider
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
        return self._session.history

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
    def post_step(self, resp: Message) -> Generator[Event, None, Optional[bool]]:
        return (yield from self._session._post_step(resp))

    @override
    def extend_paths(self, *parts: str) -> list[str]:
        return self._session.extend_paths(*parts)

    @override
    def get_final_message(self) -> str:
        final_message = self._session.latest_message
        if final_message:
            return str(final_message)
        return "(no message yet)"

    @override
    def get_evnet_extra(self) -> dict[str, Any]:
        return self._session.event_extra


class SessionID(HierarchicalID):

    pass


class Session:

    _sid: SessionID
    _history: History
    _agent: Agent
    _tool_providers: list[ToolProvider]
    _tool_manager: ToolManager
    _middleware_factories: list['SessionMiddlewareFactory']
    _middlewares: list['SessionMiddleware']
    _round_counter: int
    _theme_color: str
    _user: str

    def __init__(self, 
                 sid: str | SessionID,
                 agent: Agent, 
                 tool_providers: list[ToolProvider],
                 middleware_factories: list['SessionMiddlewareFactory'] = [],
                 user="You",
                 theme_color="") -> None:

        self._sid = SessionID.wrap(sid)
        self._agent = agent
        self._tool_providers = tool_providers
        self._middleware_factories = middleware_factories
        self._user = user
        self._theme_color = theme_color

        self._round_counter = 0
        self._history = History()

        self._tool_manager = ToolManager(initial_providers=tool_providers + [self])

        # create middlewares
        self._middlewares = []
        for factory in middleware_factories:
            obj = factory.create(self)
            if obj:
                self._middlewares.append(obj)

        self._post_init()

    def _post_init(self):
        for middleware in self._middlewares:
            middleware.post_session_init(self)

    def spawn(self, allow_sub_spawn: bool = False) -> Self:
        return self.__class__(
            sid=self._sid.spawn(),
            agent=self._agent.spawn(),
            tool_providers=self._tool_providers,
            middleware_factories=self._middleware_factories,
            user=self._sid.name
        )

    def stream(self, prompt: str) -> Generator[Event, None, str]:
        self._round_counter += 1

        # Append user turn
        self.append_message("user", prompt)
        yield UserPromptEvent(
            source_name=self._user,
            paths=self.extend_paths(), 
            prompt=prompt, 
            extra=self.event_extra
        )

        # Run the agent loop until it stops
        return (yield from self._agent.run(_AgentRunContext(self)))

    def get_tools(self) -> list[Tool]:

        @FunctionTool.wrapper(required_capabilities="subagent.spawn")
        def spawn_subagent(prompt: str):
            """Spawn a subagent to run task.
            
            Subagent will only return the final result instead of details 
            which will can make the context of main agent clean.
            If you need to run a task with many details, 
            using this tool to delegate to a subagent is a better way.

            Args:
                prompt (int): tell subagent what to do including background and details necessary

            Returns:
                str: final result
            """
            console = ConsoleEventRenderer()
            gen = self.spawn().stream(prompt)

            while True:
                try:
                    console.print(next(gen))
                except StopIteration as e:
                    return e.value

        return [spawn_subagent]
            

    def _post_step(self, resp: Message) -> Generator[Event, None, Optional[bool]]:
        loop_completed = None
        for middleware in self._middlewares:
            completed = yield from middleware.post_agent_step(self, resp)
            if completed is False:
                # the loop will continue if any middleware want it to continue
                loop_completed = False
        return loop_completed

    def add_tool_provider(self, provider: ToolProvider):
        self._tool_manager.add_provider(provider)

    def append_message(self, role: Literal["user", "assistant"], content):
        self._history.append(role, content)

    def extend_paths(self, *parts: str) -> list[str]:
        return [f"session:{self.sid.id}", f"round:{self.rounds}"] + list(parts)

    @property
    def rounds(self) -> int:
        return self._round_counter

    @property
    def history(self) -> Sequence[dict]:
        return self._history.messages

    @property
    def latest_message(self) -> Optional[dict]:
        """
        get latest message

        Returns:
            Optional[dict]: 
                None if no message or the latest one
        """
        if self._history.messages:
            return self._history.messages[-1]
        return None

    @property
    def sid(self) -> SessionID:
        return self._sid;

    @property
    def theme_color(self) -> str:
        return self._theme_color

    @property
    def event_extra(self) -> dict[str, Any]:
        return {
            "theme_color": self._theme_color,
            "session_id": self.sid
        }

    def close(self):
        print("Exiting.")
        self._agent.close()


class SessionMiddleware(Protocol):

    def post_session_init(self, session: 'Session') -> None: ...

    def post_agent_step(self, session: 'Session', resp: Message) -> Generator[Event, None, Optional[bool]]:
        yield from []
        return None


class SessionMiddlewareFactory(Protocol):

    def create(self, session: Session) -> SessionMiddleware:...