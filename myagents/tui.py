import os

from dotenv import load_dotenv
from rich.console import Console
from prompt_toolkit import prompt

from myagents.engine import ExecutionEngine
from myagents.subagent import SubagentToolProvider

from .capability import Capability
from .common import WORKDIR
from .runner import AgentRunner
from .skill import StaticSkillsLoader
from .tools import BuildinToolProvider

from .chat_model import ChatModelManager

from .events import (
    EventRenderer, 
    ConsoleEventRenderer,
)
from .task_tracker import TaskTrackerFactory
from .agent import Agent
from .session import Session
from .events import Event
from myagents import session


# init env
load_dotenv(override=True)
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
if BASE_URL:
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
MODEL = os.environ["MODEL_ID"]


SYSTEM_PROMPT = f"""
You are a coding agent at {os.getcwd()}. 
Use bash to solve tasks. Act, don't explain.
Before creating task, you MUST ask the user for confirmation.
"""


class TUI:

    def __init__(self, mainsession: Session, mainagent: Agent, runner: AgentRunner):
        self._console: Console = Console()
        self._renderer: ConsoleEventRenderer = ConsoleEventRenderer(self._console)
        self._mainsession: Session = mainsession
        self._mainagent: Agent = mainagent
        self._runner = runner

        # subscribe all events published
        self._mainsession.subscribe(self)

    def run(self) -> None:
        while True:

            # Get user input
            try:
                prompt = self.input(self._mainsession)
            except (EOFError, KeyboardInterrupt):
                return self._mainsession.close()

            prompt = prompt.strip()
            if prompt.strip().lower() in {"exit", "quit", "q"}:
                return self._mainsession.close()
            if not prompt:
                continue

            # Handle the prompt
            self._runner.run(
                session=self._mainsession, 
                agent=self._mainagent, 
                user_input=prompt
            )

            print()

    def input(self, session: Session) -> str:
        header = self._renderer.render_header(
            paths=[f"session:{session.sid.id}", f"round:{session.rounds + 1}"],
            color=session.theme_color
        )
        self._console.print(header)
        self._console.print(f"[bold cyan]Input:[/bold cyan] ", end="")
        return prompt()

    def handle_event(self, event: Event) -> None:
        self._renderer.print(event)

    def get_interested_events(self) -> set[type[Event]]:
        return set([Event])


if __name__ == "__main__":
    mainagent = Agent(
        aid="main", 
        model=ChatModelManager.get_default().get_model("MiniMax", "MiniMax-M2.7"),
        allowed_capabilities=[
            Capability.BASH.value,
            Capability.FILE_READ.value,
            "task.*", 
            Capability.SUBAGENT_SPAWN.value,
            Capability.SKILL_USE.value,
        ],
        sys_prompt=SYSTEM_PROMPT
    )

    mainsession = Session(
        sid="tui",
        agent=mainagent,
        tool_providers=[BuildinToolProvider.get_instance()],
        skill_proviers=[StaticSkillsLoader(WORKDIR / "skills")], 
        middleware_factories=[TaskTrackerFactory(3)],
        theme_color="dodger_blue2"
    )
    mainsession.add_tool_provider(SubagentToolProvider(mainsession, mainagent))

    runner = AgentRunner(ExecutionEngine())
    TUI(mainsession, mainagent, runner).run()