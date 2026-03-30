from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import MappingProxyType

import jsonschema


@dataclass(frozen=True)
class Tool(ABC):

    name: str
    description: str
    input_schema: dict

    def __post_init__(self):
        if not self.name:
            raise ValueError("Tool name cannot be empty.")
        if not self.description:
            raise ValueError("Tool description cannot be empty.")
        if not self.input_schema:
            raise ValueError("Tool input_schema cannot be empty.")

    def __call__(self, **kwargs) -> str:
        self._validate(kwargs)
        return self._run(**kwargs)

    def _validate(self, kwargs):
        try:
            jsonschema.validate(instance=kwargs, schema=self.input_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid input for tool '{self.name}': {e.message}")

    @abstractmethod
    def _run(self, **kwargs) -> str:
        pass

    def to_anthropic_tool(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:

    _tools: dict[str, Tool]

    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            print(f"Tool with name '{tool.name}' is already registered and will be overwritten.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise ValueError(f"Tool with name '{name}' is not supported.")
        return self._tools[name]

    def list_tools(self) -> set[str]:
        return set(self._tools.keys())

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def __getitem__(self, tool_name: str) -> Tool:
        return self._tools[tool_name]


class ToolManager:

    _registry: ToolRegistry

    def __init__(self):
        self._registry = ToolRegistry()

    def register_tools(self, *tools: Tool) -> None:
        for tool in tools:
            self._registry.register(tool)

    def execute(self, allowed_tools: list[str], tool_name: str, **kwargs) -> str:
        if tool_name not in allowed_tools:
            raise ValueError(f"Tool '{tool_name}' is not in the list of allowed tools.")

        if tool_name not in self._registry:
            raise ValueError(f"Tool '{tool_name}' is not supported.")

        return self._registry[tool_name](**kwargs)