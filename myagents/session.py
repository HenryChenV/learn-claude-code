"""
Session management for myagents.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Optional, Protocol
from anthropic.types import ContentBlock
from typing_extensions import override
import traceback

from myagents.tools.core.provider import ToolProvider
from myagents.tools.core.registry import ToolRegistry

from .agent import Agent
from .tools.core import Tool, ToolManager
from .utils import truncate


class EventType(Enum):
    THINKING = "thinking"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    FINAL = "final"
    UNKNOWN = "unknown"


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass(frozen=True, kw_only=True)
class Event(ABC):
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not isinstance(self.role, Role):
            raise ValueError(f"role is required for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def role(self) -> Role:
        raise NotImplementedError("Subclasses must implement the role property.")


@dataclass(frozen=True, kw_only=True)
class AssistantEvent(Event, ABC):

    @property
    @override
    def role(self) -> Role:
        return Role.ASSISTANT


@dataclass(frozen=True, kw_only=True)
class UserEvent(Event, ABC):

    @property
    @override
    def role(self) -> Role:
        return Role.USER


@dataclass(frozen=True, kw_only=True)
class SystemEvent(Event, ABC):

    @property
    @override
    def role(self) -> Role:
        return Role.SYSTEM


@dataclass(frozen=True, kw_only=True)
class UnknownEvent(Event):

    data: Any

    @property
    @override
    def role(self) -> Role:
        return Role.UNKNOWN


@dataclass(frozen=True)
class ThinkingEvent(AssistantEvent):
    thinking: str


@dataclass(frozen=True)
class ToolUseEvent(AssistantEvent):
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]


@dataclass(frozen=True)
class AssistantErrorEvent(AssistantEvent):
    error: Any


@dataclass(frozen=True)
class ToolResultEvent(SystemEvent):
    tool_name: str
    tool_use_id: str
    tool_output: str


@dataclass(frozen=True)
class AssitantTextEvent(AssistantEvent):
    content: str


@dataclass(frozen=True)
class UserPromptEvent(UserEvent):
    prompt: str


class EventFactory:

    @classmethod
    def generate(cls, *blocks: ContentBlock) -> Iterable[Event]:
        for block in blocks:
            yield cls.create(block)

    @classmethod
    def create(cls, block: ContentBlock) -> Event:
        if block.type == "thinking":
            return ThinkingEvent(thinking=block.thinking)
        elif block.type == "tool_use":
            return ToolUseEvent(
                tool_name=block.name, 
                tool_use_id=block.id, 
                tool_input=block.input
            )
        elif block.type == "tool_result":
            return ToolResultEvent(
                tool_name=block.data["name"], 
                tool_use_id=block.data["id"], 
                tool_output=block.data["output"]
            )
        elif block.type == "text":
            return AssitantTextEvent(content=block.text)
        else:
            return UnknownEvent(data=block)


class BuildinToolProvider:

    _tools: list[Tool]

    def __init__(self, tools: list[Tool]):
        self._tools = tools

    def get_tools(self) -> list[Tool]:
        return self._tools


class Session:

    _main_agent: Agent
    _tool_manager: ToolManager

    def __init__(self, 
                 agent: Agent, 
                 tools: list[Tool] = [], 
                 middlewares: list['SessionMiddleware'] = []) -> None:

        self._history = []
        self._main_agent = agent
        self._tool_manager = ToolManager()

        self._tool_manager.add_provider(BuildinToolProvider(tools))

        for middleware in middlewares:
            middleware.post_init(self)

    def add_tool_provider(self, provider: ToolProvider):
        self._tool_manager.add_provider(provider)

    @property
    def tool_manager(self):
        return self._tool_manager

    def stream(self, prompt: str):
        # Append user turn
        self._history.append({"role": "user", "content": prompt})
        yield UserPromptEvent(prompt=prompt)

        # Run the agent loop until it stops
        yield from self._agent_loop()

    def _agent_loop(self):
        while True:
            # Agent takes a step
            try:
                response = self._main_agent.step(self._history, self._resolve_tools_for(self._main_agent))
            except Exception as e:
                yield AssistantErrorEvent(error=f"Error during agent step: {e}:\n{traceback.format_exc()}")
                return

            # Append assistant turn
            self._history.append({"role": "assistant", "content": response.content})

            # If the model didn't call a tool, we're done
            if response.stop_reason != "tool_use":
                yield from EventFactory.generate(*response.content)
                return

            # Execute each tool call, collect results, or call sub-agents as needed, and append results to history for next step
            results = []

            for block in response.content:
                yield EventFactory.create(block)

                # yield extra tool result for tool_use block
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_kwargs = block.input

                    # Tool call
                    output = self._tool_manager.execute(
                        allowed_tools=self._main_agent.allowed_tools, 
                        target_tool=tool_name, 
                        **tool_kwargs
                    )

                    # print(truncate(output))
                    yield ToolResultEvent(tool_name=tool_name, tool_use_id=block.id, tool_output=output)

                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

            self._history.append({"role": "user", "content": results})

    def _resolve_tools_for(self, agent: Agent) -> list[Tool]:
        return self._tool_manager.get_tools_by_names(
            agent.allowed_tools, 
            raise_if_nonexits=True
        )

    def close(self):
        print("Exiting.")
        self._main_agent.close()


class SessionMiddleware(Protocol):

    def post_init(self, session: Session) -> None: ...
