"""
Session management for myagents.
"""


from mailbox import Message
from typing import Generator, Literal, Optional, Protocol, Sequence

from anthropic.types import Message
from typing_extensions import override

from myagents.events import Any

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
    def post_step(self, resp: Message) -> Generator[Event, None, Optional[bool]]:
        return (yield from self._session._post_step(resp))

    @override
    def extend_paths(self, *parts: str) -> list[str]:
        return self._session.extend_paths(*parts)

    @override
    def get_evnet_extra(self) -> dict[str, Any]:
        return self._session.event_extra


class SessionID:

    _name: str
    _parent: Optional['SessionID']
    _created_at: datetime
    _sub_counter: int

    def __init__(self, name, parent: Optional['SessionID'] = None):
        self._name = name
        self._parent = parent
        self._created_at = datetime.now()
        self._sub_counter = 0
        self._id = self._gen_id()

    def _gen_id(self) -> str:
        paths = [self._name]
        parent = self._parent
        while parent:
            paths.append(parent.id)
            parent = parent._parent
        return ":".join(reversed(paths))

    def spawn(self) -> 'SessionID':
        self._sub_counter += 1
        return SessionID(f"sub{self._sub_counter}", self)

    @property
    def id(self):
        return self._id

    @property
    def parent(self):
        return self._parent

    @property
    def created_at(self):
        return self._created_at

    @property
    def is_root(self) -> bool:
        return self._parent is None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.id}/{self.created_at})"

    __repr__ = __str__


class Session:

    _sid: SessionID
    _history: History
    _agent: Agent
    _tool_manager: ToolManager
    _middleware_factories: list['SessionMiddlewareFactory']
    _middlewares: list['SessionMiddleware']
    _round_counter: int
    _theme_color: str
    _user: str

    def __init__(self, 
                 name: str,
                 agent: Agent, 
                 tool_providers: list[ToolProvider],
                 middleware_factories: list['SessionMiddlewareFactory'] = [],
                 user="You",
                 theme_color="") -> None:

        self._sid = SessionID(name)
        self._history = History()
        self._round_counter = 0
        self._theme_color = theme_color
        self._user = user

        self._agent = agent

        self._tool_manager = ToolManager(initial_providers=tool_providers)

        # create middlewares
        self._middleware_factories = middleware_factories
        self._middlewares = []
        for factory in middleware_factories:
            obj = factory.create(self)
            if obj:
                self._middlewares.append(obj)

        self._post_init()

    def _post_init(self):
        for middleware in self._middlewares:
            middleware.post_session_init(self)

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

    def stream(self, prompt: str):
        self._round_counter += 1
        # Append user turn
        self._history.append("user", prompt)
        yield UserPromptEvent(
            source_name=self._user,
            paths=self.extend_paths(), 
            prompt=prompt, 
            extra=self.event_extra
        )

        # Run the agent loop until it stops
        yield from self._agent.run(_AgentRunContext(self))

    def extend_paths(self, *parts: str) -> list[str]:
        return [f"session:{self.sid.id}", f"round:{self.rounds}"] + list(parts)

    @property
    def rounds(self) -> int:
        return self._round_counter

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