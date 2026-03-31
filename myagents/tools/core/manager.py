"""Tool manager for executing tools based on agent requests.
"""


from .registry import ToolRegistry
from .tool import Tool


class ToolManager:

    _registry: ToolRegistry

    def __init__(self):
        self._registry = ToolRegistry()

    def register_tools(self, *tools: Tool) -> None:
        for tool in tools:
            self._registry.register(tool)

    def execute(self, allowed_tools: list[str], tool_name: str, **kwargs) -> str:
        # TODO distinguish between different errors 
        # to facilitate better error handling by the agent .
        # e.g. tool not allowed / permisson denied vs tool execution error
        try:
            return self._execute(allowed_tools, tool_name, **kwargs)
        except Exception as e:
            return f"Error executing tool '{tool_name}': {e}"

    def _execute(self, allowed_tools: list[str], tool_name: str, **kwargs) -> str:
        if tool_name not in allowed_tools:
            raise ValueError(f"Tool '{tool_name}' is not in the list of allowed tools.")

        if tool_name not in self._registry:
            raise ValueError(f"Tool '{tool_name}' is not supported.")

        return self._registry[tool_name](**kwargs)
