"""subagent
"""


from functools import cached_property
from typing import Iterable


from .capability import Capability

from .tools.core.tool import FunctionTool, Tool

from .agent import Agent
from .session import Session


class SubagentToolProvider:

    def __init__(self, parent_session: Session, parent_agent: Agent):
        self._psession: Session = parent_session
        self._pagent: Agent = parent_agent

    def get_tools(self) -> Iterable[Tool]:
        return [self.subagent_spawn]

    @cached_property
    def subagent_spawn(self):

        @FunctionTool.wrapper(required_capabilities=Capability.SUBAGENT_SPAWN.value)
        def spawn_subagent(prompt: str) -> str:
            """Spawn a subagent to run task.
            
            Subagent will only return the final result instead of details 
            which will can make the context of main agent clean.
            If you need to run a task with many details, 
            using this tool to delegate to a subagent is a better way.

            Args:
                prompt (int): tell subagent what to do including background and details necessary

            Returns:
                str: final result
            """
            from .runner import AgentRunner
            from .engine import ExecutionEngine

            runner = AgentRunner(ExecutionEngine())

            return runner.run(self._psession.spawn(), self._pagent.spawn(), prompt)

        return spawn_subagent
        