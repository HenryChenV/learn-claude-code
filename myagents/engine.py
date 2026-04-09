"""Execution Engine
"""


import traceback
from typing import Any, Iterable, Optional

from anthropic.types import TextBlockParam, ToolUnionParam

from .events import AssistantErrorEvent

from .tools.core.tool import ToolMeta

from .agent import Agent
from .session import Session


class ExecutionEngine:
    """Runtime Execution Engine
    """

    def run(self, session: Session, agent: Agent) -> str:
        """run loop

        Args:
            session (Session): session
            agent (Agent): agent
        """
        tool_metas = self._resolve_tools(session, agent)
        system_prompt = self._build_system_prompt(session, agent)
        try:
            # TODO
            return (yield from self._loop(ctx, tool_metas, system_prompt))

        except Exception as e:
            error=f"Error during agent loop: {e}"
            session.publish(AssistantErrorEvent(
                paths=self._build_paths(session, agent, "error"),
                source_name=agent.name, 
                error=f"{error}:\n{traceback.format_exc()}",
                extra=self._build_event_extra(session, agent),
            ))
            return error

    def _resolve_tools(self, session: Session, agent: Agent) -> list[ToolUnionParam]:
        tools = session.resolve_tools(
            allowed_capabilities=agent.allowed_capabilities,
            extra_providers=[]
        )
        return [self._build_tool_desc(t) for t in tools]

    def _build_tool_desc(self, meta: ToolMeta) -> ToolUnionParam:
        return {
            "name": meta.name,
            "description": meta.description,
            "input_schema": meta.input_schema,
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
                "If you plan to use the skill, please use use_skill tool to get more details. "
                "The names and brief descriptions of available skill are below: "
                "\n".join(f"  - {str(s)}" for s in skills)
            )
        return {"type": "text", "text": skill_prompt}

    def _build_paths(self, session: Session, agent: Agent, *parts):
        return session.extend_paths(f"agent:{agent.name}", "loop", *parts)

    def _build_event_extra(self, session: Session, agent: Agent) -> dict[str, Any]:
        extra = {}
        extra.update(session.event_extra)
        extra.update(agent.event_extra)
        return extra