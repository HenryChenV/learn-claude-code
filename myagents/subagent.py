"""subagent
"""


from functools import cached_property
from re import sub
from typing import Iterable, Optional

from myagents.events import EventSubscriber


from .capability import Capability

from .tools.core.tool import FunctionTool, Tool

from .agent import Agent
from .session import Session
from .events import Event


class FinalEventHolder:

    def __init__(self) -> None:
        self._latest: Optional[Event] = None

    def get_interested_events(self) -> set[type[Event]]: 
        """return interested events
        """
        return set([Event])

    def handle_event(self, event: Event) -> None: 
        """handle events
        """
        if event:
            self._latest = event

    def get(self) -> Optional[Event]:
        return self._latest


class SubagentToolProvider:

    def __init__(self, parent_session: Session, parent_agent: Agent, subs: Iterable[EventSubscriber] = []):
        self._psession: Session = parent_session
        self._pagent: Agent = parent_agent
        self._subs: Iterable[EventSubscriber] = subs

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

            final_event_holder = FinalEventHolder()

            runner = AgentRunner(ExecutionEngine())

            subsession = self._psession.spawn()
            if self._subs:
                for sub in self._subs:
                    subsession.subscribe(sub)
            subsession.subscribe(final_event_holder)

            subagent = self._pagent.spawn()

            runner.run(subsession, subagent, prompt)

            final_event = final_event_holder.get()

            return str(final_event) if final_event else "<no output>"


        return spawn_subagent
        