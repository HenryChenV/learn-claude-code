"""Events
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from typing import Any, Iterable, Optional, Protocol, overload
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
    tool_output: str


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


class EventRenderer(Protocol):

    def print(self, event: Event) -> None: 
        """reader and print event
        """
        ...

    @overload
    def render_header(self, *, event: Optional[Event]) -> Text: ...

    @overload
    def render_header(self, *, paths: list[str] = [], color: str = "") -> Text: ...

    def render_header(self, *, event: Optional[Event] = None, paths: list[str] = [], color: str = "") -> Text: ...

    def render_title(self, 
                     source_role: Role, source_name: str, 
                     action: str = "",
                     color: str = "", 
                     signature="") -> Text: ...


class ConsoleEventRenderer:

    _console: Console

    def __init__(self, console: Optional[Console] = None):
        self._console = console or Console()

    def print(self, event: Event) -> None:
        match event:
            case UserPromptEvent(prompt=prompt):
                if event.source_name.lower() == "you":
                    # The prompt is just typed by user and is diaplayed in tui.
                    return None
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="UserPrompt", color="cyan"
                    ),
                    body=prompt,
                )

            case ThinkingEvent(thinking=thinking):
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="Thinking", color="magenta"
                    ),
                    body=Markdown(thinking),
                )

            case AssitantOutputEvent(content=content):
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="Output", color="green"
                    ),
                    body=Markdown(content)
                )

            case AssistantErrorEvent(error=error):
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="Error", color="red"
                    ),
                    body=Markdown(str(error))
                )

            case ToolUseEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                              tool_input=tool_input):
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="ToolUse", color="yellow",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=f"{tool_name}({json.dumps(tool_input, indent=2, ensure_ascii=False)})"
                )

            case ToolResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                                 tool_output=tool_output):
                if tool_name == "bash":
                    body = Syntax(truncate(tool_output, 100), "bash", theme="monokai", line_numbers=False)
                else:
                    body = f"{tool_name} -> {tool_output}"

                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="ToolResult", color="blue",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=body
                )

            case SystemWarnEvent(content=content):
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="SystemWarn", color="orange3",
                    ),
                    body=str(content)
                )

            case UnknownEvent(data=data):
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="SystemWarn", color="red",
                    ),
                    body=str(data)
                )

            case _:
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name,
                        color="red",
                    ),
                    body=str(event)
                )

    @overload
    def render_header(self, *, event: Optional[Event]) -> Text: ...

    @overload
    def render_header(self, *, paths: list[str] = [], color: str = "") -> Text: ...

    def render_header(self, *, event: Optional[Event] = None, paths: list[str] = [], color: str = "") -> Text:
        if event:
            final_paths = event.paths
            final_color = event.extra.get("theme_color", "")
        else:
            final_paths = paths
            final_color = color
        return Text(" > ".join([p.capitalize() for p in final_paths if p]), style=f"reverse {final_color}")

    def render_title(self, 
                     source_role: Role, source_name: str, 
                     action: str = "",
                     color: str = "", 
                     signature="") -> Text:
        return Text.assemble(
            (f"{source_role.name}/{source_name}", f"bold {color}"), 
            (f" {action}" if action else "", f"{color}"),
            (f" ({signature})" if signature else "", f"italic {color}")
        )

    def _print(self, header: Any, title: Any, body: Any) -> None:
        self._console.print(header)
        self._console.print(title)
        self._console.print(Padding(body, (0, 0, 0, 4)))