"""Events
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable
from anthropic.types import ContentBlock
from typing_extensions import override


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class Event(ABC):
    source_name: str
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not isinstance(self.source_role, Role):
            raise ValueError(f"role is required for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def source_role(self) -> Role:
        raise NotImplementedError("Subclasses must implement the role property.")


@dataclass(frozen=True, kw_only=True)
class AssistantEvent(Event, ABC):

    @property
    @override
    def source_role(self) -> Role:
        return Role.ASSISTANT


@dataclass(frozen=True, kw_only=True)
class UserEvent(Event, ABC):

    source_name: str = "You"

    @property
    @override
    def source_role(self) -> Role:
        return Role.USER


@dataclass(frozen=True, kw_only=True)
class SystemEvent(Event, ABC):

    @property
    @override
    def source_role(self) -> Role:
        return Role.SYSTEM


@dataclass(frozen=True, kw_only=True)
class UnknownEvent(Event):

    source_name:str = "unknown"

    data: Any

    @property
    @override
    def source_role(self) -> Role:
        return Role.UNKNOWN


@dataclass(frozen=True, kw_only=True)
class ThinkingEvent(AssistantEvent):
    thinking: str


@dataclass(frozen=True, kw_only=True)
class ToolUseEvent(AssistantEvent):
    tool_name: str
    tool_use_id: str
    tool_input: dict[str, Any]


@dataclass(frozen=True, kw_only=True)
class AssistantErrorEvent(AssistantEvent):
    error: Any


@dataclass(frozen=True, kw_only=True)
class SystemWarnEvent(SystemEvent):
    content: Any


@dataclass(frozen=True, kw_only=True)
class ToolResultEvent(SystemEvent):

    source_name: str = "ToolManager"

    tool_name: str
    tool_use_id: str
    tool_output: str


@dataclass(frozen=True, kw_only=True)
class AssitantOutputEvent(AssistantEvent):
    content: str


@dataclass(frozen=True, kw_only=True)
class UserPromptEvent(UserEvent):
    prompt: str


class EventFactory:

    @classmethod
    def generate(cls, agent_name: str, *blocks: ContentBlock) -> Iterable[Event]:
        for block in blocks:
            yield cls.create(agent_name, block)

    @classmethod
    def create(cls, agent_name: str, block: ContentBlock) -> Event:
        if block.type == "thinking":
            return ThinkingEvent(source_name=agent_name, thinking=block.thinking)
        elif block.type == "tool_use":
            return ToolUseEvent(
                source_name=agent_name,
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
            return AssitantOutputEvent(source_name=agent_name, content=block.text)
        else:
            return UnknownEvent(data=block)
