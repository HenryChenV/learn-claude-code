"""Tool registry for managing available tools in the system.
"""


from .tool import Tool


class ToolRegistry:

    _tools: dict[str, Tool]

    def __init__(self):
        self._tools = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool with name '{tool.name}' is already registered")
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
