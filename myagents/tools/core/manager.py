"""Tool manager for executing tools based on agent requests.
"""


from .registry import ToolRegistry
from .tool import Tool
from .provider import ToolProvider


class ToolManager:

    _provider: list[ToolProvider]
    _tool_cache: dict[str, Tool] | None

    def __init__(self):
        self._provider = []
        self._tool_cache = None

    def add_provider(self, provider: ToolProvider):
        self._provider.append(provider)
        self._invalid_tools_cache()

    def remove_provider(self, provider: ToolProvider):
        self._provider.remove(provider)
        self._invalid_tools_cache()

    def get_tools_by_names(self, names: list[str], raise_if_nonexits=False):
        tools = []
        for name in names:
            if name not in self._tools:
                raise ValueError(f"tool {name} doesn't exists")
            tools.append(self._tools[name])
        return tools

    @property
    def _tools(self) -> dict[str, Tool]:
        if self._tool_cache is None:
            # build tools cache
            self._tool_cache = self._resolve_tools()
        return self._tool_cache

    def _invalid_tools_cache(self):
        self._tool_cache = None

    def _resolve_tools(self) -> dict[str, Tool]:
        tool_to_provider = {}
        tool_cache = {}
        for provider in self._provider:
            for name, tool in provider.get_tools().items():
                if name in tool_to_provider:
                    raise ValueError(f"tool {name} was provided by both {tool_to_provider[name]} and {provider}")
                tool_cache[name] = tool
                tool_to_provider[name] = provider
        return tool_cache

    def execute(self, 
                allowed_tools: list[str], 
                target_tool: str, 
                **kwargs) -> str:

        # TODO distinguish between different errors 
        # to facilitate better error handling by the agent .
        # e.g. tool not allowed / permisson denied vs tool execution error
        try:
            return self._execute(allowed_tools, target_tool, **kwargs)
        except Exception as e:
            return f"Error executing tool '{target_tool}': {e}"

    def _execute(self, allowed_tools: list[str], target_tool: str, **kwargs) -> str:
        if target_tool not in allowed_tools:
            raise ValueError(f"Tool '{target_tool}' is not in the list of allowed tools.")

        if target_tool not in self._tools:
            raise ValueError(f"Tool '{target_tool}' is not supported.")

        return self._tools[target_tool](**kwargs)
