"""
Agent class for the myagents package.
"""


import traceback
from typing import Iterable, Union

from anthropic import Anthropic, Omit, omit
from anthropic.types import Message, TextBlockParam

from myagents.tools.core.manager import ToolManager

from .tools.core import Tool
from .events import *
from .history import History


@dataclass(frozen=True)
class AgentRunContext:
    tool_manager: ToolManager
    history: History


class Agent:

    _name: str
    _client: Anthropic
    _model_id: str | None
    _system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit
    _allowed_tools: list[str]
    _max_tokens: int

    _tool_descs_cache: list[dict] | None

    def __init__(
            self, 
            name: str, 
            base_url: str | None, 
            model_id: str | None, 
            allowed_tools: list[str] = [],
            system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit,
            max_tokens: int = 8000) -> None:
        self._name = name
        self._client = Anthropic(base_url=base_url)
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._allowed_tools = allowed_tools
        self._max_tokens = max_tokens
        self._tool_descs_cache = None

    def allow_tools(self, *tools: str) -> None:
        """Allow new tools to use
        """
        self._allowed_tools.extend(tools)
        self._invalid_allowed_tool_descs_cache()

    def deny_tools(self, *tools: str) -> None:
        """ Deny tools to use
        """
        for tool in tools:
            self._allowed_tools.remove(tool)
        self._invalid_allowed_tool_descs_cache()

    def run(self, ctx: AgentRunContext):
        try:
            tool_descs = self._resolve_tools(ctx)
            yield from self._loop(ctx, tool_descs)
        except Exception as e:
            yield AssistantErrorEvent(error=f"Error during agent loop: {e}:\n{traceback.format_exc()}")

    def _loop(self, ctx: AgentRunContext, tool_descs: list[dict]):
        while True:
            # Agent takes a step
            try:
                response = self._step(ctx.history.messages, tool_descs)
            except Exception as e:
                yield AssistantErrorEvent(error=f"Error during agent step: {e}:\n{traceback.format_exc()}")
                return

            # Append assistant turn
            ctx.history.append("assistant", response.content)

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

                    # Tool call
                    output = self._use_tool(ctx, tool_name, **block.input) 
                    # print(truncate(output))
                    yield ToolResultEvent(tool_name=tool_name, tool_use_id=block.id, tool_output=output)

                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

            ctx.history.append("user", results)

    def _step(self, inputs: Iterable[dict], tool_descs: list[dict]) -> Message:
        return self._client.messages.create(
            model=self._model_id, # type: ignore
            system=self._system_prompt,
            messages=inputs, # type: ignore
            tools=tool_descs, # type: ignore
            max_tokens=self._max_tokens,
        )

    def _resolve_tools(self, ctx: AgentRunContext) -> list[dict]:
        if self._tool_descs_cache is None:
            tools = ctx.tool_manager.resolve_tools(self._allowed_tools, raise_if_nonexits=True)
            self._tool_descs_cache = [self._resolve_tool_desc(t) for t in tools]
        return self._tool_descs_cache

    def _invalid_allowed_tool_descs_cache(self):
        self._tool_descs_cache = None

    def _use_tool(self, ctx: AgentRunContext, tool_name, **tool_input) -> str:
        if tool_name not in self._allowed_tools:
            raise RuntimeError(f"Tool '{tool_name}' is not in not allowed.")

        return ctx.tool_manager.execute(
            allowed_tools=self._allowed_tools, 
            target_tool=tool_name, 
            **tool_input
        )

    def _resolve_tool_desc(self, tool: Tool) -> dict:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }


    def close(self):
        self._client.close()
