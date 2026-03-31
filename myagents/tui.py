import os

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel

from .agent import AnthropicAgent
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
                prompt = input(f"\033[36mInput[{i}]: \033[0m")
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
                #print(f"\033[35mThinking: {thinking}\033[0m")
                self._console.print("[bold magenta]Thinking:[/bold magenta]")
                md = Markdown(thinking)
                self._console.print(Padding(md, (0, 0, 0, 4)))

            case AssitantTextEvent(content=content):
                self._console.print("[bold green]Output:[/bold green]")
                self._console.print(Padding(Markdown(content), (0, 0, 0, 4)))

            case AssistantErrorEvent(error=error):
                # print(f"\033[31mAssistant Error: {error}\033[0m")
                self._console.print("[bold red]Assistant Error:[/bold red]")
                self._console.print(Padding(str(error), (0, 0, 0, 4)))

            case ToolUseEvent(tool_name=tool_name, tool_use_id=tool_use_id, tool_input=tool_input):
                print(f"\033[33mToolUse[{tool_use_id}]: {tool_name}({tool_input})\033[0m")
                self._console.print(f"[bold yellow]ToolUse[{tool_use_id}]:[/bold yellow]")
                content = f"{tool_name}({tool_input})"
                self._console.print(Padding(content, (0, 0, 0, 4)))

            case ToolResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, tool_output=tool_output):
                # print(f"\033[34mToolResult[{tool_use_id}]: {tool_name} -> {tool_output}\033[0m")
                self._console.print(f"[bold blue]ToolResult[{tool_use_id}]:[/bold blue]")
                content = f"{tool_name} -> {tool_output}"
                self._console.print(Padding(content, (0, 0, 0, 4)))

            case UnknownEvent(data=data):
                # print(f"\033[31mUnknown event: {data}\033[0m")
                self._console.print("[bold red]Unknown event:[/bold red]")
                self._console.print(Padding(str(data), (0, 0, 0, 4)))

            case _:
                # print(f"\033[31mUnhandled event: {event}\033[0m")
                self._console.print("[bold red]Unhandled event:[/bold red]")
                self._console.print(Padding(str(event), (0, 0, 0, 4)))


if __name__ == "__main__":
    TUI().run(Session(agent=AnthropicAgent(
        name="main", 
        base_url=BASE_URL, 
        model_id=MODEL,
        tools=[run_bash, read_file, write_file, edit_file],
        system_prompt=SYSTEM_PROMPT
    )))