"""Events
"""


from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from re import sub
from typing import Any, Callable, Iterable, Optional, Protocol, overload
from anthropic.types import ContentBlock
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text
from typing_extensions import override

from myagents.utils import truncate


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass(frozen=True)
class Event(ABC):
    source_name: str
    paths: list[str]
    extra: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        if not isinstance(self.source_role, Role):
            raise ValueError(f"role is required for {self.__class__.__name__}.")

    @property
    @abstractmethod
    def source_role(self) -> Role:
        raise NotImplementedError("Subclasses must implement the role property.")

    def update_extra(self, **kwargs):
        self.extra.update(kwargs)


@dataclass(frozen=True, kw_only=True)
class AssistantEvent(Event, ABC):

    @property
    @override
    def source_role(self) -> Role:
        return Role.ASSISTANT


@dataclass(frozen=True, kw_only=True)
class UserEvent(Event, ABC):

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
    output: str


@dataclass(frozen=True, kw_only=True)
class SkillResultEvent(SystemEvent):

    source_name: str = "SkillManager"

    tool_name: str
    tool_use_id: str
    output: str


@dataclass(frozen=True, kw_only=True)
class AssitantOutputEvent(AssistantEvent):
    content: str


@dataclass(frozen=True, kw_only=True)
class UserPromptEvent(UserEvent):
    prompt: str


class EventFactory:

    @classmethod
    def generate(cls, paths:list[str], agent_name: str, blocks: list[ContentBlock], extra={}) -> Iterable[Event]:
        for block in blocks:
            yield cls.create(paths, agent_name, block, extra=extra)

    @classmethod
    def create(cls, paths:list[str], agent_name: str, block: ContentBlock, extra={}) -> Event:
        if block.type == "thinking":
            return ThinkingEvent(
                paths=paths,
                source_name=agent_name, 
                thinking=block.thinking,
                extra=extra,
            )

        elif block.type == "tool_use":
            return ToolUseEvent(
                paths=paths,
                source_name=agent_name,
                tool_name=block.name, 
                tool_use_id=block.id, 
                tool_input=block.input,
                extra=extra,
            )

        elif block.type == "tool_result":
            return ToolResultEvent(
                paths=paths,
                tool_name=block.data["name"], 
                tool_use_id=block.data["id"], 
                tool_output=block.data["output"],
                extra=extra,
            )

        elif block.type == "text":
            return AssitantOutputEvent(
                paths=paths,
                source_name=agent_name, 
                content=block.text,
                extra=extra,
            )

        else:
            return UnknownEvent(
                paths=paths, 
                data=block,
                extra=extra,
            )


class EventSubscriber(Protocol):

    def get_interested_events(self) -> set[type[Event]]: 
        """return interested events
        """
        ...

    def handle_event(self, event: Event) -> None: 
        """handle events
        """
        ...


class EventBus:

    def __init__(self, subscribers: set[EventSubscriber] = set()):
        self._subcribers: set[EventSubscriber] = set()
        self._subscriptions: dict[type[Event], set[EventSubscriber]] = defaultdict(set)

        for sub in subscribers:
            self.add_subscriber(sub)

    def add_subscriber(self, subscriber: EventSubscriber) -> None:
        event_types = subscriber.get_interested_events()
        if not event_types:
            return
        
        self._subcribers.add(subscriber)
        for et in event_types:
            self._subscriptions[et].add(subscriber)

    def publish(self, event: Event) -> None:
        handlers: set[EventSubscriber] = set()

        for et in type(event).__mro__:
            if et not in self._subscriptions:
                continue
            handlers.update(self._subscriptions[et])
        
        for h in handlers:
            h.handle_event(event)
