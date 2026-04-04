"""Defines the base Tool class and a FunctionTool subclass that wraps Python functions as tools.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import inspect
from types import UnionType
from typing import Dict, List, Union, get_args, get_origin, overload
from typing_extensions import Callable, override

import jsonschema


@dataclass(frozen=True)
class ToolDesc:
    """Description of the Tool
    """
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


@dataclass(frozen=True)
class Tool(ABC):

    desc: ToolDesc
    required_capabilities: list[str] = field(default_factory=list)

    def __post_init__(self):
        # make sure required_capabilities is not empty
        required_capabilities = self.required_capabilities or ["default"]
        # make required_capablities immuatable
        object.__setattr__(
            self, 
            'required_capabilities', 
            tuple(required_capabilities)
        )

    def __call__(self, **kwargs) -> str:
        self._validate(kwargs)
        return self._run(**kwargs)

    def _validate(self, kwargs):
        try:
            jsonschema.validate(instance=kwargs, schema=self.desc.input_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid input for tool '{self.desc.name}': {e.message}")

    @abstractmethod
    def _run(self, **kwargs) -> str:
        pass


@dataclass(frozen=True, kw_only=True)
class FunctionTool(Tool):

    func: Callable[..., str]

    @override
    def _run(self, **kwargs) -> str:
        return self.func(**kwargs)

    @overload
    @classmethod
    def wrapper(
        cls,
        func: Callable[..., str],
    ) -> 'FunctionTool': ...

    @overload
    @classmethod
    def wrapper(
        cls,
        *,
        name: str | None = None, 
        description: str | None = None, 
        input_schema: dict | None = None,
        required_capabilities: str | list[str] = [],
    ) -> Callable[..., 'FunctionTool']: ...

    @classmethod
    def wrapper(
        cls, 
        func: Callable[..., str] | None = None,
        *,
        name: str | None = None, 
        description: str | None = None, 
        input_schema: dict | None = None,
        required_capabilities: str | list[str] = [],
    ) -> Callable[..., 'FunctionTool'] | 'FunctionTool':

        def decorator(func: Callable[..., str]) -> 'FunctionTool':
            tool_name = name or func.__name__
            tool_description = description or func.__doc__
            tool_input_schema = input_schema or cls._infer_schema(func)

            tool_required_capabilities = required_capabilities or []
            if isinstance(tool_required_capabilities, str):
                tool_required_capabilities = [tool_required_capabilities]

            if not tool_name:
                raise ValueError(f"Failed to infer tool name from function {func}. Please provide a name.")
            if not tool_description:
                raise ValueError(f"Failed to infer tool description from function {func}. Please provide a description.")
            if not tool_input_schema:
                raise ValueError(f"Failed to infer tool input schema from function {func}. Please provide an input_schema.")

            return cls(
                desc=ToolDesc(
                    name=tool_name, 
                    description=tool_description, 
                    input_schema=tool_input_schema,
                ),
                func=func, 
                required_capabilities=tool_required_capabilities
            )

        if func is None:
            return decorator

        return decorator(func)

    @classmethod
    def _infer_schema(cls, func: Callable[..., str]) -> dict:
        sig = inspect.signature(func)
        properties = {}
        required = []

        for name, param in sig.parameters.items():
            try:
                schema = cls.py_type_to_json_schema(param.annotation)
            except ValueError as e:
                raise ValueError(f"Failed to infer JSON schema for function '{func}': {e}")

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

        # None
        if py_type is type(None):
            return {"type": "null"}

        # Empty
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

        # Union
        if origin in (Union, UnionType):
            return {"oneOf": [cls.py_type_to_json_schema(arg) for arg in args]}

        # Enum
        if isinstance(py_type, type) and issubclass(py_type, Enum):
            values = [e.value for e in py_type]

            value_types = {type(v) for v in values}
            if len(value_types) == 1:
                t = value_types.pop()
                if t is int:
                    json_type = "integer"
                elif t is float:
                    json_type = "number"
                elif t is bool:
                    json_type = "boolean"
                else:
                    json_type = "string"
            else:
                # mixed type
                json_type = "string"

            enum_desc = ",".join([f"{e.name}={e.value}" for e in py_type])

            return {"type": json_type, "enum": values, "description": f"Enum values: {enum_desc}"}

        raise ValueError(f"Unsupported parameter type: {py_type}")
