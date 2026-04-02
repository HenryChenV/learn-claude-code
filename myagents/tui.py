import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.syntax import Syntax

from .agent import Agent
from .tools.core import Tool
from .tools.impls import (
    run_bash, 
    read_file, 
    write_file, 
    edit_file
)
from .session import (
    AssistantErrorEvent, 
    AssitantTextEvent, 
    Session, Event, 
    ThinkingEvent, 
    ToolResultEvent, 
    ToolUseEvent, 
    UnknownEvent, 
    UserPromptEvent
)


# init env
load_dotenv(override=True)
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
if BASE_URL:
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
MODEL = os.environ["MODEL_ID"]


SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


class TUI:

    _console: Console

    def __init__(self):
        self._console = Console()

    def run(self, session: Session) -> None:
        i = 0
        while True:
            i = i + 1

            # Get user input
            try:
                self._console.print(f"[bold cyan]Input[{i}]:[/bold cyan] ", end="")
                prompt = input()
            except (EOFError, KeyboardInterrupt):
                return session.close()
            if prompt.strip().lower() in {"exit", "quit", "q", ""}:
                return session.close()

            # Handle the prompt and get the final content
            for event in session.stream(prompt):
                # Display the final output
                self.render(event)

            print()

    def render(self, event: Event) -> str | None:
        match event:
            case UserPromptEvent(prompt=prompt):
                return None

            case ThinkingEvent(thinking=thinking):
                self._console.print("[bold magenta]Thinking:[/bold magenta]")
                md = Markdown(thinking)
                self._console.print(Padding(md, (0, 0, 0, 4)))

            case AssitantTextEvent(content=content):
                self._console.print("[bold green]Output:[/bold green]")
                self._console.print(Padding(Markdown(content), (0, 0, 0, 4)))

            case AssistantErrorEvent(error=error):
                self._console.print("[bold red]Assistant Error:[/bold red]")
                self._console.print(Padding(str(error), (0, 0, 0, 4)))

            case ToolUseEvent(tool_name=tool_name, tool_use_id=tool_use_id, tool_input=tool_input):
                full_content = f"[bold yellow]ToolUse([dim]{tool_use_id}[/dim]):[/bold yellow] {tool_name}({tool_input})"
                self._console.print(Padding(full_content, (0, 0, 0, 0)))

            case ToolResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, tool_output=tool_output):
                self._console.print(f"[bold blue]ToolResult([dim]{tool_use_id}[/dim]):[/bold blue]")
                
                if tool_name == "bash":
                    syntax = Syntax(tool_output, "bash", theme="monokai", line_numbers=False)
                    self._console.print(Padding(syntax, (0, 0, 0, 4)))
                else:
                    content = f"{tool_name} -> {tool_output}"
                    self._console.print(Padding(content, (0, 0, 0, 4)))

            case UnknownEvent(data=data):
                self._console.print("[bold red]Unknown event:[/bold red]")
                self._console.print(Padding(str(data), (0, 0, 0, 4)))

            case _:
                self._console.print("[bold red]Unhandled event:[/bold red]")
                self._console.print(Padding(str(event), (0, 0, 0, 4)))


if __name__ == "__main__":
    buildin_tools: list[Tool] = [run_bash, read_file, write_file, edit_file]
    TUI().run(
        Session(
            agent=Agent(
                name="main", 
                base_url=BASE_URL, 
                model_id=MODEL,
                allowed_tools=[t.name for t in buildin_tools],
                system_prompt=SYSTEM_PROMPT
            ),
            tools=buildin_tools,
            middlewares=[]
        ),
    )