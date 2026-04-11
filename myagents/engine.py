"""Execution Engine
"""


import traceback
from typing import Any, Iterable, Optional

from anthropic.types import Message, MessageParam, TextBlockParam, ToolUnionParam

from myagents.chat_model import ChatModel

from .events import AssistantErrorEvent, EventFactory, Role, SkillResultEvent, ToolResultEvent

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
                    paths=self._build_paths_for_loop(session, agent, f"step:{step_counter}"),
                    source_name=agent.name,
                    error=f"{error}: \n{traceback.format_exc()}",
                    extra=self._merge_event_extra(session, agent),
                ))
                return

            # Append assistant turn
            session.append_message("assistant", response.content)

            loop_completed = None
            continue_loop = False
            if response.stop_reason == "tool_use":
                # Tool Use
                # Execute each tool call, collect results, 
                # or call sub-agents as needed, 
                # and append results to conversation for next step

                continue_loop = True

                results = []

                for block in response.content:
                    session.publish(EventFactory.create(
                        self._build_paths_for_loop(session, agent, f"step:{step_counter}"),
                        agent.name, 
                        block,
                        extra=self._merge_event_extra(session, agent),
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
                                paths=self._build_paths_for_loop(session, agent, f"step:{step_counter}"),
                                tool_name=tool_name, 
                                tool_use_id=block.id, 
                                output=output,
                                extra=self._merge_event_extra(session, agent),
                            ))
                        else:
                            output = session.use_tool(
                                allowed_capabilities=agent.allowed_capabilities,
                                tool_name=tool_name, 
                                tool_kwargs=tool_kwargs
                            ) 
                            # print(truncate(output))
                            session.publish(ToolResultEvent(
                                paths=self._build_paths_for_loop(session, agent, f"step:{step_counter}"),
                                tool_name=tool_name, 
                                tool_use_id=block.id, 
                                output=output,
                                extra=self._merge_event_extra(session, agent),
                            ))

                        results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

                session.append_message("user", results)

            else:
                # Final Message
                # If the model didn't call a tool, we're done
                continue_loop = False
                session.publish(*EventFactory.generate(
                    self._build_paths_for_loop(session, agent, f"step:{step_counter}"),
                    agent.name, 
                    response.content,
                    extra=self._merge_event_extra(session, agent),
                ))

            # If the completed is None, it will be ignored.
            # If anyone need the loop to continue, it must respoend an explicit False.
            if session.post_step(response):
                continue_loop = True
            
            if not continue_loop:
                return

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
                "If you plan to use the skill, please use use_skill tool to get more details. "
                "The names and brief descriptions of available skill are below: "
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