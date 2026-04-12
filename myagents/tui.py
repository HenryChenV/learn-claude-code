from abc import ABC
import json
import os
from re import sub
from typing import Optional

from dotenv import load_dotenv
from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text
from rich.console import Console
from prompt_toolkit import prompt

from myagents.engine import ExecutionEngine
from myagents.subagent import SubagentToolProvider
from myagents.utils import truncate

from .capability import Capability
from .common import WORKDIR
from .runner import AgentRunner
from .skill import StaticSkillsLoader
from .tools import BuildinToolProvider

from .chat_model import ChatModelManager, ModelSpec

from .events import *

from .task_tracker import TaskTrackerFactory
from .agent import Agent
from .session import Session
from .events import Event


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


class ConsoleEventRenderer:

    _console: Console

    def __init__(self, console: Optional[Console] = None):
        self._console = console or Console()

    def print(self, event: Event) -> None:
        match event:
            case StepStartEvent():
                self._print(
                    header=self.render_header(event=event),
                )

            case UserPromptEvent(prompt=prompt):
                if event.source_name.lower() == "you":
                    # The prompt is just typed by user and is diaplayed in tui.
                    return None
                self._print(
                    header=self.render_header(event=event),
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="UserPrompt", color="cyan"
                    ),
                    body=prompt,
                )

            case ThinkingEvent(thinking=thinking):
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="Thinking", color="magenta"
                    ),
                    body=Markdown(thinking),
                )

            case AssitantOutputEvent(content=content):
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="Output", color="green"
                    ),
                    body=Markdown(content)
                )

            case AssistantErrorEvent(error=error):
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="Error", color="red"
                    ),
                    body=Markdown(str(error))
                )

            case ToolUseEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                              tool_input=tool_input):
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="ToolUse", color="yellow",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=f"{tool_name}({json.dumps(tool_input, indent=2, ensure_ascii=False)})"
                )

            case ToolResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                                 output=output):
                if tool_name == "bash":
                    body = Syntax(truncate(output, 150), "bash", theme="monokai", line_numbers=False)
                else:
                    body = f"{tool_name} -> {truncate(output, 150)}"

                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="ToolResult", color="blue",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=body
                )

            case SkillResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, 
                                 output=output):
                body = f"{tool_name} -> {truncate(output, 150)}"

                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="ToolResult", color="blue",
                        signature=f"{tool_name}/{tool_use_id}"
                    ),
                    body=body
                )

            case SystemWarnEvent(content=content):
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="SystemWarn", color="orange3",
                    ),
                    body=str(content)
                )

            case UnknownEvent(data=data):
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name, 
                        action="SystemWarn", color="red",
                    ),
                    body=str(data)
                )

            case StepEndEvent(model=model, usage=usage):
                blocks = [f"{model}"]
                if usage:
                    blocks.append(
                        "[Usage] " +
                        " ".join([f"{k}={format(v, ',') if isinstance(v, int) else v}" for k, v in usage.items() if v])
                    )
                self._print(
                    footer=self.render_footer(
                        event=event,
                        blocks=blocks
                    )
                )

            case _:
                self._print(
                    title=self.render_title(
                        event.source_role, event.source_name,
                        color="red",
                    ),
                    body=str(event)
                )

    @overload
    def render_header(self, *, event: Optional[Event]) -> Text: ...

    @overload
    def render_header(self, *, paths: list[str] = [], color: str = "") -> Text: ...

    def render_header(self, *, event: Optional[Event] = None, paths: list[str] = [], color: str = "") -> Text:
        if event:
            final_paths = event.paths
            final_color = event.extra.get("theme_color", "")
        else:
            final_paths = paths
            final_color = color
        return Text(" > ".join([p.capitalize() for p in final_paths if p]), style=f"reverse bold {final_color}")

    def render_title(self, 
                     source_role: Role, source_name: str, 
                     action: str = "",
                     color: str = "", 
                     signature="") -> Text:
        return Text.assemble(
            (f"{source_role.name}/{source_name}", f"bold {color}"), 
            (f" {action}" if action else "", f"{color}"),
            (f" ({signature})" if signature else "", f"italic {color}")
        )

    def render_footer(self, *, event: Optional[Event] = None, blocks: list[str] = [], color: str = "") -> Text:
        if event:
            final_color = event.extra.get("theme_color", "")
        else:
            final_color = color
        return Text("▸ " + " · ".join(blocks), style=f"reverse {final_color}")

    def _print(self, *, 
               header: Any = None, 
               title: Any = None, 
               body: Any = None,
               footer: Any = None) -> None:
        if header:
            self._console.print(header)
        if title:
            self._console.print(title)
        if body:
            self._console.print(Padding(body, (0, 0, 0, 4)))
        if footer:
            self._console.print(Padding(footer, (0, 0, 1, 0)))


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
                return

            if not prompt:
                continue

            prompt = prompt.strip()
            if prompt.strip().lower() in {"exit", "quit", "q"}:
                return

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
        models=[ModelSpec("MiniMax", "MiniMax-M2.7")],
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
        model_manager=ChatModelManager.get_default(),
        tool_providers=[BuildinToolProvider.get_instance()],
        skill_providers=[StaticSkillsLoader(WORKDIR / "skills")], 
        middleware_factories=[TaskTrackerFactory(3)],
        theme_color="dodger_blue2"
    )

    runner = AgentRunner(ExecutionEngine())

    tui = TUI(mainsession, mainagent, runner)

    # support subagent which will also publish events to tui
    mainsession.add_tool_provider(SubagentToolProvider(mainsession, mainagent, subs=[tui]))

    tui.run()