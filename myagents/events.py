"""Events
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable
from anthropic.types import ContentBlock
from typing_extensions import override


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
class SystemWarnEvent(SystemEvent):
    source: str
    content: Any


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
