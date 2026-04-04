"""
Agent class for the myagents package.
"""


import traceback
from typing import Iterable, Literal, Sequence, Union

from anthropic import Anthropic, Omit, omit
from anthropic.types import Message, TextBlockParam

from .tools.core.capability import CapabilityRule

from .tools.core.manager import ToolManager

from .tools.core import Tool
from .events import *
from .history import History



class AgentHooks:

    def post_step(self, resp: Message, loop_completed: bool) -> tuple[list[Event], bool]:
        """hook before loop end

        If some messages are appended, you may want the loop continues.

        Returns:
            bool: completed or not
        """
        return [], True

class AgentRunContext(AgentHooks, ABC):

    @abstractmethod
    def get_inputs(self) -> Sequence[dict]:
        """get inputs
        """
        pass

    @abstractmethod
    def append_message(self, role: Literal["user", "assistant"], content):
        """append message to history
        """
        pass

    @abstractmethod
    def resolve_tools(self, allowed_capabilities: list[CapabilityRule]) -> tuple[Tool, ...]:
        """resolve tools by allowed capabilities
        """
        pass

    @abstractmethod
    def use_tool(self, allowed_capabilities: list[CapabilityRule], tool_to_use: str, **tool_params) -> str:
        """use tools subjected to allowed capabilities
        """
        pass


class Agent:
    """
    Agent

    Agents are like persons with a specific role, 
    having their own preference, capabilities, etc.

    Attributes:
        _name (str): name
        _system_prompt (str): system prompt, define its preference in nature languange.
        _allowed_capabilities (list[str]): capabilities
    """

    _name: str
    _client: Anthropic
    _model_id: str | None
    _system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit
    _allowed_capabilities: list[CapabilityRule]
    _max_tokens: int

    def __init__(
            self, 
            name: str, 
            base_url: str | None, 
            model_id: str | None, 
            allowed_capabilities: list[str] = [],
            system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit,
            max_tokens: int = 8000) -> None:
        self._name = name
        self._client = Anthropic(base_url=base_url)
        self._model_id = model_id
        self._system_prompt = system_prompt
        self._allowed_capabilities = [CapabilityRule.wrap(c) for c in allowed_capabilities]
        self._max_tokens = max_tokens

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
                response = self._chat(ctx.get_inputs(), tool_descs)
            except Exception as e:
                yield AssistantErrorEvent(error=f"Error during agent step: {e}:\n{traceback.format_exc()}")
                return

            # Append assistant turn
            ctx.append_message("assistant", response.content)

            loop_completed = None
            if response.stop_reason == "tool_use":
                # Tool Use
                # Execute each tool call, collect results, 
                # or call sub-agents as needed, 
                # and append results to history for next step

                loop_completed = False

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

                ctx.append_message("user", results)

            else:
                # Final Message
                # If the model didn't call a tool, we're done
                loop_completed = True
                yield from EventFactory.generate(*response.content)

            events, loop_completed = ctx.post_step(response, loop_completed)

            if events:
                for event in events:
                    yield event
            
            if loop_completed:
                return

    def _chat(self, inputs: Iterable[dict], tool_descs: list[dict]) -> Message:
        return self._client.messages.create(
            model=self._model_id, # type: ignore
            system=self._system_prompt,
            messages=inputs, # type: ignore
            tools=tool_descs, # type: ignore
            max_tokens=self._max_tokens,
        )

    def _resolve_tools(self, ctx: AgentRunContext) -> list[dict]:
        tools = ctx.resolve_tools(self._allowed_capabilities)
        return [self._build_tool_desc(t) for t in tools]

    def _build_tool_desc(self, tool: Tool) -> dict:
        return {
            "name": tool.desc.name,
            "description": tool.desc.description,
            "input_schema": tool.desc.input_schema,
        }

    def _use_tool(self, ctx: AgentRunContext, tool_name, **tool_input) -> str:
        return ctx.use_tool(
            allowed_capabilities=self._allowed_capabilities, 
            tool_to_use=tool_name, 
            **tool_input
        )

    def close(self):
        self._client.close()
