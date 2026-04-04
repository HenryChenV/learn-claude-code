"""Tool manager for executing tools based on agent requests.
"""


from collections import defaultdict
from json import tool
from math import log
from typing import Sequence

from .capability import CapabilityEvaluator, CapabilityRule

from .tool import Tool
from .provider import ToolProvider

from myagents.log import get_logger


logger = get_logger(__name__)



class ResolvedToolsCacheValue:

    _tools: tuple[Tool, ...]
    _tool_by_name: dict[str, Tool]

    def __init__(self, tools: tuple[Tool, ...]):
        self._tools = tools
        self._tool_by_name = {t.desc.name: t for t in tools}

    def get_all(self) -> tuple[Tool, ...]:
        return self._tools

    def get_by_name(self, name) -> Tool | None:
        return self._tool_by_name.get(name)


class ToolManager:
    """Tool Manager

    Attributes:
        _providers: tool providers
        _resolved_tool_descs: 
            cache for resolved tool descs, key is frozenset of capacilities, value is tools
    """

    _providers: list[ToolProvider]
    _resolved_tools_cache: dict[frozenset, ResolvedToolsCacheValue] | None

    def __init__(self, initial_providers: list[ToolProvider] = []):
        self._providers = []

        self._resolved_tools_cache = None

        if initial_providers:
            self.add_providers(*initial_providers)

    def add_providers(self, *providers: ToolProvider):
        """add providers in order

        if the provider is already in list, 
        it will be removed and append to the end
        """
        for p in providers:
            self.add_provider(p)

    def add_provider(self, provider: ToolProvider):
        """add provider

        if the provider is already in list, 
        it will be removed and append to the end
        """
        if provider in self._providers:
            self._providers.remove(provider)
        self._providers.append(provider)
        self._invalid_tools_cache()

    def remove_provider(self, provider: ToolProvider):
        """remove provider
        """
        if provider not in self._providers:
            return
        self._providers.remove(provider)
        self._invalid_tools_cache()

    def resolve_tools_by_capabilities(self, capabilities: list[CapabilityRule]) -> tuple[Tool, ...]:
        """resolve tools by capabilities

        Iterate providers from newest to oldest.
        Pick the newer one if the name is duplicated
        """
        return self._get_or_resolved_tools_by_capabilities(capabilities).get_all()

    def _get_or_resolved_tools_by_capabilities(self, capabilities: list[CapabilityRule]) -> ResolvedToolsCacheValue:
        if self._resolved_tools_cache is None:
            self._resolved_tools_cache = {}

        cache_key = frozenset(capabilities)

        if not cache_key in self._resolved_tools_cache:
            tools = self._find_tools_by_capabilities(capabilities)
            self._resolved_tools_cache[cache_key] = ResolvedToolsCacheValue(tools)
            logger.debug(f"resolve tools by {capabilities} -> {tools}")

        return self._resolved_tools_cache[cache_key]

    def _find_tools_by_capabilities(self, capabilities: list[CapabilityRule]) -> tuple[Tool, ...]:
        evaluator = CapabilityEvaluator(capabilities)
        allowed_tools: dict[str, Tool] = {}

        # traverse provders in reverse order
        for provider in reversed(self._providers):
            tools = provider.get_tools()
            for tool in tools:
                if not evaluator.is_allowed(tool.required_capabilities):
                    # capabilities not matched
                    continue
                if tool.desc.name in allowed_tools:
                    # newer tool selected
                    continue
                allowed_tools[tool.desc.name] = tool
        return tuple(allowed_tools.values())

    def _invalid_tools_cache(self):
        self._resolved_tools_cache = None

    def _get_tools_as_map(self) -> dict[str, Tool]:
        stats: dict[str, list[tuple[ToolProvider, Tool]]] = defaultdict(list)
        map: dict[str, Tool] = {}
        for p in self._providers:
            for t in p.get_tools():
                map[t.desc.name] = t
                stats[t.desc.name].append((p, t))

        for tool_name, provided in stats.items():
            if len(provided) > 1:
                logger.warning(f"Tool {tool_name} has multiple providers: {provided}")

        return map

    def execute(self, 
                allowed_capabilities: list[CapabilityRule], 
                target_tool: str, 
                **kwargs) -> str:
        """execute the tools with allowed_capabilities

        Note: 
        If agent doesn't have enough capabilities to access the tool,
        it should not expose if the tool exists or not.
        Becuase in the agent's view, it cannot see the tool.
        """

        # TODO distinguish between different errors 
        # to facilitate better error handling by the agent .
        # e.g. tool not allowed / permisson denied vs tool execution error
        try:
            return self._execute(allowed_capabilities, target_tool, **kwargs)
        except Exception as e:
            return f"Error executing tool '{target_tool}': {e}"

    def _execute(self, allowed_capabilities: list[CapabilityRule], target_tool: str, **kwargs) -> str:
        cached = self._get_or_resolved_tools_by_capabilities(allowed_capabilities)

        tool = cached.get_by_name(target_tool)
        if not tool:
            raise RuntimeError(
                f"Tool {target_tool} is not supported yet. Please check if the tool exists or capabilities are allowed")

        return tool(**kwargs)
