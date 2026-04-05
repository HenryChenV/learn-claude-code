"""
Agent class for the myagents package.
"""


import traceback
from typing import Generator, Iterable, Literal, Optional, Sequence, Union

from anthropic import Omit, omit
from anthropic.types import Message, TextBlockParam

from .tools.core.capability import CapabilityRule

from .tools.core import Tool
from .events import *
from .models import Model


class AgentRunHooks:

    def post_step(self, resp: Message) -> Generator[Event, None, Optional[bool]]:
        """hook before loop end

        If some messages are appended, you may want the loop continues.

        Returns:
            bool: 
                True: completed
                False: continue
                None: determined by caller
        """
        yield from []
        return None

class AgentRunContext(AgentRunHooks, ABC):

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

    @abstractmethod
    def extend_paths(self, *parts: str) -> list[str]:
        """extend paths to the paths in context and return full path
        """
        pass

    def get_evnet_extra(self) -> dict[str, Any]:
        """get extra info as map for event
        """
        return {}


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
    _system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit
    _model: Model
    _allowed_capabilities: list[CapabilityRule]
    _max_tokens: int

    def __init__(
            self, 
            name: str, 
            model: Model,
            allowed_capabilities: list[str] = [],
            system_prompt: Union[str, Iterable[TextBlockParam]] | Omit = omit,
            max_tokens: int = 8000) -> None:
        self._name = name
        self._model = model
        self._system_prompt = system_prompt
        self._allowed_capabilities = [CapabilityRule.wrap(c) for c in allowed_capabilities]
        self._max_tokens = max_tokens

    def run(self, ctx: AgentRunContext):
        try:
            tool_descs = self._resolve_tools(ctx)
            yield from self._loop(ctx, tool_descs)
        except Exception as e:
            yield AssistantErrorEvent(
                paths=self._build_paths(ctx, "loop", "error"),
                source_name=self._name, 
                error=f"Error during agent loop: {e}:\n{traceback.format_exc()}",
                extra=self._build_event_extra(ctx),
            )

    def _loop(self, ctx: AgentRunContext, tool_descs: list[dict]):
        step_counter = 0
        while True:
            step_counter += 1

            # Agent takes a step
            try:
                response = self._chat(ctx.get_inputs(), tool_descs)
            except Exception as e:
                yield AssistantErrorEvent(
                    paths=self._build_paths(ctx, f"step:{step_counter}"),
                    source_name=self._name,
                    error=f"Error during agent step: {e}:\n{traceback.format_exc()}",
                    extra=self._build_event_extra(ctx),
                )
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
                    yield EventFactory.create(
                        self._build_paths(ctx, f"step:{step_counter}"),
                        self._name, 
                        block,
                        extra=self._build_event_extra(ctx),
                    )

                    # yield extra tool result for tool_use block
                    if block.type == "tool_use":
                        tool_name = block.name

                        # Tool call
                        output = self._use_tool(ctx, tool_name, **block.input) 
                        # print(truncate(output))
                        yield ToolResultEvent(
                            paths=self._build_paths(ctx, f"step:{step_counter}"),
                            tool_name=tool_name, 
                            tool_use_id=block.id, 
                            tool_output=output,
                            extra=self._build_event_extra(ctx),
                        )

                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

                ctx.append_message("user", results)

            else:
                # Final Message
                # If the model didn't call a tool, we're done
                loop_completed = True
                yield from EventFactory.generate(
                    self._build_paths(ctx, f"step:{step_counter}"),
                    self._name, 
                    response.content,
                    extra=self._build_event_extra(ctx),
                )

            # If the completed is None, it will be ignored.
            # If anyone need the loop to continue, it must respoend an explicit False.
            completed = yield from ctx.post_step(response)
            if completed is False:
                loop_completed = False
            
            if loop_completed:
                return

    def _chat(self, inputs: Iterable[dict], tool_descs: list[dict]) -> Message:
        return self._model.chat(
            max_tokens=self._max_tokens,
            messages=inputs, # type: ignore
            system_prompt=self._system_prompt,
            tools=tool_descs, # type: ignore
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

    def _build_paths(self, ctx: AgentRunContext, *parts):
        return ctx.extend_paths(f"agent:{self._name}", "loop", *parts)

    def _build_event_extra(self, ctx: AgentRunContext):
        return ctx.get_evnet_extra()

    def close(self):
        pass
