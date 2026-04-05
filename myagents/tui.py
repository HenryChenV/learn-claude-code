import json
import os
from typing import Any, Optional, overload

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
from .task_tracker import TaskTracker, TaskTrackerFactory
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
                self.print(event)

            print()

    def input(self, session: Session) -> str:
        header = self._format_header(
            paths=[f"session:{session.sid.id}", f"round:{session.rounds + 1}"],
            color=session.theme_color
        )
        self._console.print(header)
        self._console.print(f"[bold cyan]Input:[/bold cyan] ", end="")
        return input()

    def print(self, event: Event) -> str | None:
        match event:
            case UserPromptEvent(prompt=prompt):
                if event.source_name.lower() == "you":
                    # The prompt is just typed by user and is diaplayed in tui.
                    return None
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="UserPrompt", color="cyan"
                    ),
                    body=prompt,
                )

            case ThinkingEvent(thinking=thinking):
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="Thinking", color="magenta"
                    ),
                    body=Markdown(thinking),
                )

            case AssitantOutputEvent(content=content):
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="Output", color="green"
                    ),
                    body=Markdown(content)
                )

            case AssistantErrorEvent(error=error):
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="Error", color="red"
                    ),
                    body=Markdown(str(error))
                )

            case ToolUseEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                              tool_input=tool_input):
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="ToolUse", color="yellow",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=f"{tool_name}({json.dumps(tool_input, indent=2, ensure_ascii=False)})"
                )

            case ToolResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                                 tool_output=tool_output):
                if tool_name == "bash":
                    body = Syntax(truncate(tool_output, 100), "bash", theme="monokai", line_numbers=False)
                else:
                    body = f"{tool_name} -> {tool_output}"

                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="TolResult", color="blue",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=body
                )

            case SystemWarnEvent(content=content):
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="SystemWarn", color="orange3",
                    ),
                    body=str(content)
                )

            case UnknownEvent(data=data):
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name, 
                        action="SystemWarn", color="red",
                    ),
                    body=str(data)
                )

            case _:
                self._print(
                    header=self._format_header(event=event),
                    title=self._format_title(
                        event.source_role, event.source_name,
                        color="red",
                    ),
                    body=str(event)
                )

    @overload
    def _format_header(self, *, event: Optional[Event]) -> Text: ...

    @overload
    def _format_header(self, *, paths: list[str] = [], color: str = "") -> Text: ...

    def _format_header(self, *, event: Optional[Event] = None, paths: list[str] = [], color: str = "") -> Text:
        if event:
            final_paths = [p.capitalize() for p in event.paths if p]
            final_color = color or event.extra.get("theme_color", "")
        else:
            final_paths = paths
            final_color = color
        return Text(" > ".join(final_paths), style=f"reverse {final_color}")

    def _format_title(self, 
                      source_role: Role, source_name: str, 
                      action: str = "",
                      color: str = "", 
                      signature="") -> Text:
        return Text.assemble(
            (f"{source_role.name}/{source_name}", f"bold {color}"), 
            (f" {action}" if action else "", f"{color}"),
            (f" ({signature})" if signature else "", f"italic {color}")
        )

    def _print(self, header: Any, title: Any, body: Any) -> None:
        self._console.print(header)
        self._console.print(title)
        self._console.print(Padding(body, (0, 0, 0, 4)))


if __name__ == "__main__":
    TUI().run(
        Session(
            name="tui",
            agent=Agent(
                name="main", 
                model=ModelManager.get_default().get_model("MiniMax", "MiniMax-M2.7"),
                allowed_capabilities=["bash", "file.*", "task.*"],
                system_prompt=SYSTEM_PROMPT
            ),
            tool_providers=[SessionBuildinToolProvider([run_bash, read_file, write_file, edit_file])],
            middleware_factories=[TaskTrackerFactory(3)],
            theme_color="dodger_blue2"
        ),
    )