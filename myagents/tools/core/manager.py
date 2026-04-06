"""Tool manager for executing tools based on agent requests.
"""


from typing import Iterable, Optional

from myagents.capability import CapabilityEvaluator, CapabilityRule

from .tool import Tool, ToolMeta
from .provider import ToolProvider

from myagents.log import get_logger


logger = get_logger(__name__)



class ResolvedToolsCacheValue:

    def __init__(self, tools: tuple[Tool, ...]):
        self._tools: tuple[Tool, ...] = tools
        self._tool_metas: dict[str, ToolMeta] = {t.meta.name: t.meta for t in tools}
        self._tool_by_name: dict[str, Tool] = {t.meta.name: t for t in tools}

    def get_metas(self) -> dict[str, ToolMeta]:
        return self._tool_metas

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
    _resolved_tools_cache: dict[frozenset[CapabilityRule], ResolvedToolsCacheValue]

    def __init__(self, initial_providers: list[ToolProvider] = []):
        self._providers = []
        self._resolved_tools_cache = {}

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

    def resolve_tools(self, 
                      allowed_capabilities: Iterable[CapabilityRule],
                      extra_providers: Iterable[ToolProvider]) -> Iterable[ToolMeta]:
        """resolve tools by capabilities

        The new one will overwrite the old one if the name of tool is duplicated.
        Tools provided by extra_providers will overwrite existing one.
        Tools resoved from extra_providers will not be cached.
        """
        tools = self._get_or_resolve_tools(allowed_capabilities).get_metas()

        if extra_providers:
            extra_tools = self._find_tools(
                allowed_capabilities=allowed_capabilities,
                providers=extra_providers
            )
            if extra_tools:
                # overwrite tools by name
                tools.update({name: tool.meta for name, tool in extra_tools.items()})

        return tools.values()

    def _get_or_resolve_tools(self, allowed_capabilities: Iterable[CapabilityRule]) -> ResolvedToolsCacheValue:
        """get or resolve tools by capabilities

        If the capabilities hint cache, return it from cache.
        Or resolve tools, update cache and return

        Args:
            capabilities (list[CapabilityRule]): allowed capabilities

        Returns:
            ResolvedToolsCacheValue: cached value
        """
        if self._resolved_tools_cache is None:
            self._resolved_tools_cache = {}

        cache_key = frozenset(allowed_capabilities)

        if cache_key not in self._resolved_tools_cache:
            tools = self._find_tools(
                allowed_capabilities=allowed_capabilities,
                providers=self._providers
            )
            cache_value = ResolvedToolsCacheValue(tuple(tools.values()))
            self._resolved_tools_cache[cache_key] = cache_value
            logger.debug(f"resolve tools by {allowed_capabilities} -> {tools}")

        return self._resolved_tools_cache[cache_key]

    def _find_tools(self, 
                    allowed_capabilities: Iterable[CapabilityRule], 
                    providers: Iterable[ToolProvider]) -> dict[str, Tool]:
        evaluator = CapabilityEvaluator(allowed_capabilities)
        allowed_tools: dict[str, Tool] = {}

        # the older one will be skipped
        for provider in reversed(list(providers)):
            provided_tools = provider.get_tools()
            for tool in provided_tools:
                if not evaluator.is_allowed(tool.meta.required_capabilities):
                    # capabilities not matched
                    continue
                if tool.meta.name in allowed_tools:
                    # skip if name is duplicated
                    continue
                # select the first allowed one
                allowed_tools[tool.meta.name] = tool
        return allowed_tools

    def _find_first_tool(self, 
                         tool_name: str, 
                         allowed_capabilities: Iterable[CapabilityRule], 
                         providers: Iterable[ToolProvider]) -> Optional[Tool]:
        evaluator = CapabilityEvaluator(allowed_capabilities)

        # the older one will be skipped
        for provider in reversed(list(providers)):
            provided_tools = provider.get_tools()
            for tool in provided_tools:
                if tool_name != tool.meta.name:
                    continue
                if not evaluator.is_allowed(tool.meta.required_capabilities):
                    # capabilities not matched
                    continue
                # return the first caplibities matched tool
                return tool

        return None

    def _invalid_tools_cache(self):
        self._resolved_tools_cache.clear()

    def execute(self, 
                allowed_capabilities: Iterable[CapabilityRule], 
                tool_name: str, 
                tool_kwargs: dict,
                extra_providers: Iterable[ToolProvider] = []) -> str:
        """execute the tools with allowed_capabilities

        Note: 
        If agent doesn't have enough capabilities to access the tool,
        it should not expose if the tool exists or not.
        Becuase in the agent's view, it cannot see the tool.
        """

        # TODO distinguish between different errors 
        # to facilitate better error handling by the agent .
        # e.g. tool not allowed / permisson denied vs tool execution error

        tool = None
        # try to find the tool in extra_providers first
        if extra_providers:
            tool = self._find_first_tool(
                tool_name=tool_name,
                allowed_capabilities=allowed_capabilities,
                providers=extra_providers,
            )

        # If tool doesn't exist in extra_providers,
        # try to find the tool in holded providers
        if not tool:
            # this method is the same one used in resolve_tools,
            # which provides the same view as the resolution phase.
            cached = self._get_or_resolve_tools(allowed_capabilities)
            tool = cached.get_by_name(tool_name)

        if not tool:
            raise RuntimeError(
                f"Tool {tool_name} is not supported yet. Please check if the tool exists or capabilities are allowed")

        return tool(**tool_kwargs)
