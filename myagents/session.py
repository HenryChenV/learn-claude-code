"""
Session management for myagents.
"""


from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Optional
from anthropic.types import ContentBlock
from typing_extensions import override

from .agent import AnthropicAgent
from .tools.core import ToolManager
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
    HARNESS = "harness"
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
class HarnessEvent(Event, ABC):

    @property
    @override
    def role(self) -> Role:
        return Role.HARNESS


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


class AssistantErrorEvent(AssistantEvent):
    error: Any


@dataclass(frozen=True)
class ToolResultEvent(HarnessEvent):
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


class Session:

    _main_agent: AnthropicAgent
    _sub_agents: dict[str, AnthropicAgent]
    _tool_manager: ToolManager

    def __init__(self, agent: AnthropicAgent) -> None:
        self._history = []
        self._main_agent = agent
        self._tool_manager = ToolManager()

        # register main agent's tools
        self._tool_manager.register_tools(*agent.tools)

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
                response = self._main_agent.step(self._history)
            except Exception as e:
                yield UnknownEvent(data=f"Error during agent step: {e}")
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
                        tool_name=tool_name, 
                        **tool_kwargs
                    )

                    # print(truncate(output))
                    yield ToolResultEvent(tool_name=tool_name, tool_use_id=block.id, tool_output=output)

                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

            self._history.append({"role": "user", "content": results})

    def close(self):
        print("Exiting.")
        self._main_agent.close()

