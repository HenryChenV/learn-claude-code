"""Agent Runner

reveive input and call engine to run
"""


from myagents.agent import Agent

from .session import Session

from .events import EventSubscriber

from .engine import ExecutionEngine


class AgentRunner:

    def __init__(self, 
                 engine: ExecutionEngine, 
                 observers: list[EventSubscriber] = []):
        self._engine: ExecutionEngine = engine

    def run(self, session: Session, agent: Agent, user_input: str) -> str:
        # append user input to session
        session.append_user_input(user_input)
        # run loop
        return self._engine.run(session, agent)