"""Execution Engine
"""


import traceback
from typing import Any, Iterable, Optional

from anthropic.types import Message, TextBlockParam

from .chat_model import ChatModel
from .utils import estimate_next_context

from .events import *

from .tools.core.tool import ToolMeta

from .agent import Agent
from .session import Session


class ExecutionEngine:
    """Runtime Execution Engine
    """

    def run(self, session: Session, agent: Agent) -> None:
        """run loop

        Args:
            session (Session): session
            agent (Agent): agent
        """
        try:
            self._loop(session, agent)

        except Exception as e:
            error=f"Error during agent loop: {e}"
            session.publish(AssistantErrorEvent(
                paths=self._build_paths_for_loop(session, agent, "error"),
                source_name=agent.name, 
                error=f"{error}:\n{traceback.format_exc()}",
                extra=self._merge_event_extra(session, agent),
            ))

    def _loop(self, session: Session, agent: Agent) -> None:
        tools: Iterable[ToolMeta] = session.resolve_tools(agent.allowed_capabilities)
        system_prompt = self._build_system_prompt(session, agent)

        model = session.resolve_model(agent.models)
        if not model:
            raise RuntimeError(f"no models available, declared are {agent.models}")

        step_counter = 0
        while True:
            step_counter += 1

            continue_loop = self._step(
                session=session,
                agent=agent,
                model=model,
                step_counter=step_counter,
                tools=tools,
                system_prompt=system_prompt
            )
            
            if not continue_loop:
                return

    def _step(self, 
              session: Session, 
              agent: Agent,
              model: ChatModel,
              step_counter: int,
              tools: Iterable[ToolMeta],
              system_prompt: Iterable[TextBlockParam]) -> bool:
        event_paths_of_step = self._build_paths_for_loop(session, agent, f"step:{step_counter}")
        event_extra = self._merge_event_extra(session, agent)

        session.publish(StepStartEvent(
            paths=event_paths_of_step,
            source_name=agent.name,
            extra=event_extra
        ))

        # Agent takes a step
        try:
            response: Message = model.chat(
                max_tokens=agent.max_tokens,
                messages=session.messages, 
                tools=tools,
                system_prompt=system_prompt,
            )
        except Exception as e:
            error=f"Error during agent step: {e}"
            session.publish(AssistantErrorEvent(
                paths=event_paths_of_step,
                source_name=agent.name,
                error=f"{error}: \n{traceback.format_exc()}",
                extra=event_extra,
            ))
            return False

        # Append assistant turn
        session.append_message("assistant", response.content)

        continue_loop = False
        tool_use_results = []
        if response.stop_reason == "tool_use":
            # Tool Use
            # Execute each tool call, collect results, 
            # or call sub-agents as needed, 
            # and append results to conversation for next step

            continue_loop = True


            for block in response.content:
                session.publish(EventFactory.create(
                    event_paths_of_step,
                    agent.name, 
                    block,
                    extra=event_extra,
                ))

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
                    if session.is_skill_use(tool_name):
                        output = session.use_skill(
                            allowed_capabilities=agent.allowed_capabilities,
                            skill_kwargs=tool_kwargs
                        )
                        session.publish(SkillResultEvent(
                            paths=event_paths_of_step,
                            tool_name=tool_name, 
                            tool_use_id=block.id, 
                            output=output,
                            extra=event_extra,
                        ))
                    else:
                        output = session.use_tool(
                            allowed_capabilities=agent.allowed_capabilities,
                            tool_name=tool_name, 
                            tool_kwargs=tool_kwargs
                        ) 
                        # print(truncate(output))
                        session.publish(ToolResultEvent(
                            paths=event_paths_of_step,
                            tool_name=tool_name, 
                            tool_use_id=block.id, 
                            output=output,
                            extra=event_extra,
                        ))

                    tool_use_results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

            session.append_message("user", tool_use_results)

        else:
            # Final Message
            # If the model didn't call a tool, we're done
            continue_loop = False
            for event in EventFactory.generate(
                event_paths_of_step,
                agent.name, 
                response.content,
                extra=event_extra,
            ):
                session.publish(event)

        # If the completed is None, it will be ignored.
        # If anyone need the loop to continue, it must respoend an explicit False.
        if session.post_step(response):
            continue_loop = True

        session.publish(StepEndEvent(
            paths=event_paths_of_step,
            source_name=agent.name,
            extra=event_extra,
            model=response.model,
            usage=self._evaluate_usage(
                model,
                response, 
                agent.max_tokens,
                [str(tool_use_results)]
            )
        ))

        return continue_loop

    def _evaluate_usage(self, 
                        model: ChatModel,
                        resp: Message, 
                        max_output_tokens: int,
                        new_inputs: list[str] = []):
        if not resp or not resp.usage:
            return {}

        input_tokens = resp.usage.input_tokens
        output_tokens = resp.usage.output_tokens
        cache_read_input_tokens = resp.usage.cache_read_input_tokens
        cur_context = sum([input_tokens, output_tokens, cache_read_input_tokens or 0])
        next_context_estimate = estimate_next_context(
            model=model.model,
            prev_usage=resp.usage,
            new_inputs=new_inputs
        )
        return {
            "input_tokens": input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": resp.usage.cache_creation_input_tokens,
            "cur_context": cur_context,
            "cur_context_percentage": cur_context/model.context_window,
            "next_context_estimate": next_context_estimate,
            "next_context_estimate_percentage": next_context_estimate/model.context_window,
        }

    def _build_system_prompt(self, session: Session, agent: Agent) -> Iterable[TextBlockParam]:
        skill_prompt = self._resolve_skills_as_prompt(session, agent)

        if skill_prompt:
            return list(agent.sys_prompt) + [skill_prompt]
        return agent.sys_prompt

    def _resolve_skills_as_prompt(self, session: Session, agent: Agent) -> Optional[TextBlockParam]:
        skills = session.resolve_skills(agent.allowed_capabilities)

        if not skills:
            return None
        else:
            skill_prompt = (
                "在执行用户的任务前，先看下可用技能。"
                "技能列表如下(仅包含name和简单描述, 更多细节使用工具use_skill获取):"
                "\n".join(f"  - {str(s)}" for s in skills)
            )
        return {"type": "text", "text": skill_prompt}

    def _build_paths_for_loop(self, session: Session, agent: Agent, *parts):
        return session.extend_paths(f"agent:{agent.name}", *parts)

    def _merge_event_extra(self, session: Session, agent: Agent) -> dict[str, Any]:
        extra = {}
        extra.update(session.event_extra)
        extra.update(agent.event_extra)
        return extra