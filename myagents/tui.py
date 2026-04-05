import json
import os
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from .models import ModelManager

from .events import (
    Role, 
    Event,
    SystemWarnEvent,
    AssistantErrorEvent, 
    AssitantOutputEvent, 
    ThinkingEvent, 
    ToolResultEvent, 
    ToolUseEvent, 
    UnknownEvent, 
    UserPromptEvent
)
from .task_tracker import TaskTracker
from .utils import truncate
from .agent import Agent
from .tools.impls import (
    run_bash, 
    read_file, 
    write_file, 
    edit_file
)
from .session import Session, SessionBuildinToolProvider


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

    def __init__(self):
        self._console = Console()

    def run(self, session: Session) -> None:
        round = 1
        while True:

            # Get user input
            try:
                self._console.print(f"[bold cyan][{round}] Input:[/bold cyan] ", end="")
                prompt = input()
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
                self.print(round, event)

            print()

            round += 1

    def print(self, round: int, event: Event) -> str | None:
        match event:
            case UserPromptEvent(prompt=prompt):
                return None

            case ThinkingEvent(source_role=source_role, source_name=source_name, 
                               thinking=thinking):
                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="Thinking", color="magenta"
                    ),
                    Markdown(thinking)
                )

            case AssitantOutputEvent(source_role=source_role, source_name=source_name, 
                                     content=content):
                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="Output", color="green"
                    ),
                    Markdown(content)
                )

            case AssistantErrorEvent(source_role=source_role, source_name=source_name, 
                                     error=error):
                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="Error", color="red"
                    ),
                    Markdown(str(error))
                )

            case ToolUseEvent(source_role=source_role, source_name=source_name, 
                              tool_name=tool_name, tool_use_id=tool_use_id, 
                              tool_input=tool_input):
                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="ToolUse", color="yellow",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    f"{tool_name}({json.dumps(tool_input, indent=2, ensure_ascii=False)})"
                )

            case ToolResultEvent(source_role=source_role, source_name=source_name, 
                                 tool_name=tool_name, tool_use_id=tool_use_id, 
                                 tool_output=tool_output):
                if tool_name == "bash":
                    body = Syntax(truncate(tool_output, 100), "bash", theme="monokai", line_numbers=False)
                else:
                    body = f"{tool_name} -> {tool_output}"

                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="TolResult", color="blue",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body
                )

            case SystemWarnEvent(source_role=source_role, source_name=source_name, 
                                 content=content):
                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="SystemWarn", color="orange3",
                    ),
                    str(content)
                )

            case UnknownEvent(source_role=source_role, source_name=source_name, data=data):
                self._print(
                    self._format_title(
                        round, source_role, source_name, 
                        action="SystemWarn", color="red",
                    ),
                    str(data)
                )

            case _:
                self._print(
                    self._format_title(
                        round, Role.UNKNOWN, "unknwon",
                        color="red",
                    ),
                    str(event)
                )

    def _format_title(self, 
                      round: int, 
                      source_role: Role, source_name: str, 
                      action: str = "",
                      color: str = "", 
                      signature="") -> Text:
        return Text.assemble(
            (f"[{round}]", f"{color}"), 
            (f" {source_role.name}/{source_name}", f"bold {color}"), 
            (f" {action}" if action else "", f"{color}"),
            (f" ({signature})" if signature else "", f"italic {color}")
        )

    def _print(self, title: Any, body: Any) -> None:
        self._console.print(title)
        self._console.print(Padding(body, (0, 0, 0, 6)))


if __name__ == "__main__":
    TUI().run(
        Session(
            agent=Agent(
                name="main", 
                model=ModelManager.get_default().get_model("MiniMax", "MiniMax-M2.7"),
                allowed_capabilities=["bash", "file.*", "task.*"],
                system_prompt=SYSTEM_PROMPT
            ),
            tool_providers=[SessionBuildinToolProvider([run_bash, read_file, write_file, edit_file])],
            middlewares=[TaskTracker()],
        ),
    )