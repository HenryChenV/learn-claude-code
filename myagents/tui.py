import os

from dotenv import load_dotenv
from rich.console import Console
from prompt_toolkit import prompt

from myagents.capability import Capability
from myagents.common import WORKDIR
from myagents.skill import StaticSkillsLoader
from myagents.tools import BuildinToolProvider

from .chat_model import ChatModelManager

from .events import (
    EventRenderer, 
    ConsoleEventRenderer,
)
from .task_tracker import TaskTrackerFactory
from .agent import Agent
from .session import Session


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

    _console: Console
    _renderer: EventRenderer

    def __init__(self):
        self._console = Console()
        self._renderer = ConsoleEventRenderer(self._console)

    def run(self, session: Session) -> None:
        while True:

            # Get user input
            try:
                prompt = self.input(session)
            except (EOFError, KeyboardInterrupt):
                return session.close()

            prompt = prompt.strip()
            if prompt.strip().lower() in {"exit", "quit", "q"}:
                return session.close()
            if not prompt:
                continue

            # Handle the prompt and get the final content
            for event in session.stream(prompt):
                # Display the final output
                self._renderer.print(event)

            print()

    def input(self, session: Session) -> str:
        header = self._renderer.render_header(
            paths=[f"session:{session.sid.id}", f"round:{session.rounds + 1}"],
            color=session.theme_color
        )
        self._console.print(header)
        self._console.print(f"[bold cyan]Input:[/bold cyan] ", end="")
        return prompt()


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
    TUI().run(
        Session(
            sid="tui",
            agent=mainagent,
            tool_providers=[BuildinToolProvider.get_instance()],
            skill_proviers=[StaticSkillsLoader(WORKDIR / "skills")], 
            middleware_factories=[TaskTrackerFactory(3)],
            theme_color="dodger_blue2"
        ),
    )