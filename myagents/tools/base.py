from abc import ABC, abstractmethod
from dataclasses import dataclass
import inspect
from typing import Callable, Dict, List, get_args, get_origin, overload
from typing_extensions import override

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


class FunctionTool(Tool):

    _func: Callable[..., str]

    def __init__(self, func: Callable[..., str], name: str, description: str, input_schema: dict) -> None:
        super().__init__(name=name, description=description, input_schema=input_schema)
        self._func = func

    @override
    def _run(self, **kwargs) -> str:
        return self._func(**kwargs)

    @classmethod
    @overload
    def wrapper(
        cls,
        func: Callable[..., str],
    ) -> 'FunctionTool': ...

    @classmethod
    @overload
    def wrapper(
        cls,
        *,
        name: str | None = None, 
        description: str | None = None, 
        input_schema: dict | None = None
    ) -> Callable[..., 'FunctionTool']: ...

    @classmethod
    def wrapper(
        cls, 
        func: Callable[..., str] | None = None,
        *,
        name: str | None = None, 
        description: str | None = None, 
        input_schema: dict | None = None) -> Callable[..., 'FunctionTool'] | 'FunctionTool':

        def decorator(func: Callable[..., str]) -> 'FunctionTool':
            tool_name = name or func.__name__
            tool_description = description or func.__doc__
            tool_input_schema = input_schema or cls._infer_schema(func)

            if not tool_name:
                raise ValueError(f"Failed to infer tool name from function {func}. Please provide a name.")
            if not tool_description:
                raise ValueError(f"Failed to infer tool description from function {func}. Please provide a description.")
            if not tool_input_schema:
                raise ValueError(f"Failed to infer tool input schema from function {func}. Please provide an input_schema.")

            return cls(func=func, name=tool_name, description=tool_description, input_schema=tool_input_schema)

        if func is None:
            return decorator

        return decorator(func)

    @classmethod
    def _infer_schema(cls, func: Callable[..., str]) -> dict:
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            schema = cls.py_type_to_json_schema(param.annotation)

            if param.default is param.empty:
                required.append(name)
            else:
                schema["default"] = param.default

            properties[name] = schema

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @classmethod
    def py_type_to_json_schema(cls, py_type) -> dict:
        # basic types
        if py_type == str:
            return {"type": "string"}
        if py_type == int:
            return {"type": "integer"}
        if py_type == float:
            return {"type": "number"}
        if py_type == bool:
            return {"type": "boolean"} 

        if py_type is inspect._empty:
            return {"type": "string"}

        origin = get_origin(py_type)
        args = get_args(py_type)

        # List[T]
        if origin in (list, List):
            item_type = args[0] if args else str
            return {"type": "array", "items": cls.py_type_to_json_schema(item_type)}

        # Dict[str, T]
        if origin in (dict, Dict):
            value_type = args[1] if len(args) == 2 else str
            return {"type": "object", "additionalProperties": cls.py_type_to_json_schema(value_type)}

        raise ValueError(f"Unsupported parameter type: {py_type}")


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