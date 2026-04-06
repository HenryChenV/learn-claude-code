"""
Agent class for the myagents package.
"""


import traceback
from typing import Generator, Iterable, Literal, Optional, Self, Union

from anthropic import Omit, omit
from anthropic.types import Message, MessageParam, TextBlockParam, ToolUnionParam

from myagents.tools.core.provider import ToolProvider

from .skill import SkillMeta, SkillProvider

from .ids import HierarchicalID

from .capability import CapabilityRule

from .tools.core import ToolMeta
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
    def get_inputs(self) -> Iterable[MessageParam]:
        """get inputs
        """
        pass

    @abstractmethod
    def append_message(self, role: Literal["user", "assistant"], content):
        """append message to history
        """
        pass

    @abstractmethod
    def resolve_skills(self, 
                       allowed_capabilities: list[CapabilityRule],
                       extra_providers: Iterable[SkillProvider] = []) -> Iterable[SkillMeta]:
        """resolve skills by capabilities

        Args:
            allowed_capabilities (list[CapabilityRule]): capabilities

        Returns:
            tuple[str, ...]: (name, description) of skills
        """
        pass

    @abstractmethod
    def is_skill_use(self, tool_name: str) -> bool:
        """if the given tool is to use skill
        """
        pass

    @abstractmethod
    def use_skill(self, 
                  allowed_capabilities: list[CapabilityRule], 
                  skill_kwargs: dict, 
                  extra_providers: Iterable[SkillProvider] = []) -> str:
        """use skill

        Args:
            allowed_capabilities (list[CapabilityRule]): allowed capabilities
            skill_name (str): name of the skill
            extra_providers (Iterable[ToolProvider], optional): extra providers

        Returns:
            _type_: content of the skill
        """
        pass

    @abstractmethod
    def resolve_tools(self, 
                      allowed_capabilities: list[CapabilityRule], 
                      extra_providers: list[ToolProvider] = []) -> Iterable[ToolMeta]:
        """resolve tools by allowed capabilities
        """
        pass

    @abstractmethod
    def use_tool(self, 
                 allowed_capabilities: list[CapabilityRule], 
                 tool_name: str, 
                 tool_kwargs: dict, 
                 extra_providers: Iterable[ToolProvider] = []) -> str:
        """use tools subjected to allowed capabilities
        """
        pass

    @abstractmethod
    def extend_paths(self, *parts: str) -> list[str]:
        """extend paths to the paths in context and return full path
        """
        pass

    @abstractmethod
    def get_final_message(self) -> str:
        """get final message
        """
        pass

    def get_evnet_extra(self) -> dict[str, Any]:
        """get extra info as map for event
        """
        return {}


class AgentID(HierarchicalID):
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

    _aid: AgentID
    _agent_sys_prompt: Iterable[TextBlockParam]
    _model: Model
    _allowed_capabilities: list[CapabilityRule]
    _max_tokens: int

    def __init__(
            self, 
            aid: Union[str, AgentID], 
            model: Model,
            allowed_capabilities: list[str] = [],
            sys_prompt: Optional[str] = None,
            max_tokens: int = 8000) -> None:
        self._aid = AgentID.wrap(aid)
        self._model = model
        self._agent_sys_prompt = [{"type": "text", "text": sys_prompt}] if sys_prompt else []
        self._allowed_capabilities = [CapabilityRule.wrap(c) for c in allowed_capabilities]
        self._max_tokens = max_tokens

    def spawn(self, allow_sub_spawn: bool = False) -> Self:
        if allow_sub_spawn:
            capabilities = [c.raw for c in self._allowed_capabilities]
        else:
            capabilities = [c.raw for c in self._allowed_capabilities] + ["!subagent.spawn"]
        return self.__class__(
            self._aid.spawn(),
            model=self._model,
            allowed_capabilities=capabilities,
            max_tokens=self._max_tokens
        )

    @property
    def name(self):
        return self._aid.name

    def run(self, ctx: AgentRunContext) -> Generator[Event, None, str]:
        tool_metas = self._resolve_tools(ctx)
        system_prompt = self._build_system_prompt(ctx)
        try:
            return (yield from self._loop(ctx, tool_metas, system_prompt))

        except Exception as e:
            error=f"Error during agent loop: {e}"
            yield AssistantErrorEvent(
                paths=self._build_paths(ctx, "loop", "error"),
                source_name=self.name, 
                error=f"{error}:\n{traceback.format_exc()}",
                extra=self._build_event_extra(ctx),
            )
            return error

    def _loop(self, 
              ctx: AgentRunContext, 
              tool_metas: list[ToolUnionParam],
              system_prompt: Iterable[TextBlockParam],
        ) -> Generator[Event, None, str]:
        step_counter = 0
        while True:
            step_counter += 1

            # Agent takes a step
            try:
                response = self._chat(
                    messages=ctx.get_inputs(), 
                    tools=tool_metas,
                    system_prompt=system_prompt,
                )
            except Exception as e:
                error=f"Error during agent step: {e}"
                yield AssistantErrorEvent(
                    paths=self._build_paths(ctx, f"step:{step_counter}"),
                    source_name=self.name,
                    error=f"{error}: \n{traceback.format_exc()}",
                    extra=self._build_event_extra(ctx),
                )
                return error

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
                        self.name, 
                        block,
                        extra=self._build_event_extra(ctx),
                    )

                    # yield extra tool result for tool_use block
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_kwargs = block.input

                        # Tool call
                        # skill use is a special tool use:
                        # - For other tool uses, only the required capaibilites of tool should be evaluated,
                        # - For skill use, not only the requried capabilities of the tool should be evaluated, 
                        #   but also the required capabilities of skill to use should be evaluated. 
                        #   The allowed capabilities should be given.
                        if ctx.is_skill_use(tool_name):
                            output = self._use_skill(ctx, tool_kwargs)
                            yield SkillResultEvent(
                                paths=self._build_paths(ctx, f"step:{step_counter}"),
                                tool_name=tool_name, 
                                tool_use_id=block.id, 
                                output=output,
                                extra=self._build_event_extra(ctx),
                            )
                        else:
                            output = self._use_tool(ctx, tool_name, tool_kwargs) 
                            # print(truncate(output))
                            yield ToolResultEvent(
                                paths=self._build_paths(ctx, f"step:{step_counter}"),
                                tool_name=tool_name, 
                                tool_use_id=block.id, 
                                output=output,
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
                    self.name, 
                    response.content,
                    extra=self._build_event_extra(ctx),
                )

            # If the completed is None, it will be ignored.
            # If anyone need the loop to continue, it must respoend an explicit False.
            completed = yield from ctx.post_step(response)
            if completed is False:
                loop_completed = False
            
            if loop_completed:
                return ctx.get_final_message()

    def _chat(self, 
              messages: Iterable[MessageParam], 
              tools: Iterable[ToolUnionParam],
              system_prompt: Iterable[TextBlockParam] = []) -> Message:
        return self._model.chat(
            max_tokens=self._max_tokens,
            messages=messages,
            system_prompt=system_prompt,
            tools=tools,
        )

    def _resolve_tools(self, ctx: AgentRunContext) -> list[ToolUnionParam]:
        tools = ctx.resolve_tools(
            allowed_capabilities=self._allowed_capabilities,
            extra_providers=[]
        )
        return [self._build_tool_desc(t) for t in tools]

    def _build_system_prompt(self, ctx: AgentRunContext) -> Iterable[TextBlockParam]:
        skill_prompt = self._resolve_skills_as_prompt(ctx)

        if skill_prompt:
            return list(self._agent_sys_prompt) + [skill_prompt]
        return self._agent_sys_prompt

    def _resolve_skills_as_prompt(self, ctx: AgentRunContext) -> Optional[TextBlockParam]:
        skills = ctx.resolve_skills(self._allowed_capabilities)

        if not skills:
            return None
        else:
            skill_prompt = (
                "If you plan to use the skill, please use use_skill tool to get more details. "
                "The names and brief descriptions of available skill are below: "
                "\n".join(f"  - {str(s)}" for s in skills)
            )
        return {"type": "text", "text": skill_prompt}

    def _build_tool_desc(self, meta: ToolMeta) -> ToolUnionParam:
        return {
            "name": meta.name,
            "description": meta.description,
            "input_schema": meta.input_schema,
        }

    def _use_skill(self,
                   ctx: AgentRunContext,
                   tool_kwargs: dict) -> str:
        return ctx.use_skill(
            allowed_capabilities=self._allowed_capabilities,
            skill_kwargs=tool_kwargs,
            extra_providers=[]
        )

    def _use_tool(self, 
                  ctx: AgentRunContext, 
                  tool_name: str, 
                  tool_kwargs: dict) -> str:
        return ctx.use_tool(
            allowed_capabilities=self._allowed_capabilities, 
            tool_name=tool_name, 
            tool_kwargs=tool_kwargs,
            extra_providers=[]
        )

    def _build_paths(self, ctx: AgentRunContext, *parts):
        return ctx.extend_paths(f"agent:{self.name}", "loop", *parts)

    def _build_event_extra(self, ctx: AgentRunContext):
        return ctx.get_evnet_extra()

    def close(self):
        pass
