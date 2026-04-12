import json
import os
from typing import Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from rich.markdown import Markdown
from rich.padding import Padding
from rich.syntax import Syntax
from rich.text import Text
from rich.console import Console

from myagents.engine import ExecutionEngine
from myagents.subagent import SubagentToolProvider
from myagents.utils import truncate

from .capability import Capability
from .common import WORKDIR
from .runner import AgentRunner
from .skill import StaticSkillsLoader
from .tools import BuildinToolProvider

from .chat_model import ChatModelManager

from .events import *

from .task_tracker import TaskTrackerFactory
from .agent import Agent
from .session import Session, SessionID
from .events import Event


# init env
load_dotenv(override=True)


SYSTEM_PROMPT = f"""
You are a coding agent at {os.getcwd()}. 
尽量使用tools或skills完成任务。
在执行任务前，如果你觉得任务不够清晰，可以和用户确认细节。
得到不要的细节后，你先规划如何做，规划必须和用户确认, 得到用户允许后再执行。
收到用户任务后, 如果任务较复杂或者执行时间较长, 使用工具跟踪进度, 使用subagent帮你做每个步骤, 你负责汇总.
Act, don't explain.
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
                    items: list[str] = []
                    if usage.get("input_tokens"):
                        items.append(f"input={usage['input_tokens']:,}")
                    if usage.get("cache_read_input_tokens"):
                        items.append(f"cache_read={usage['cache_read_input_tokens']:,}")
                    if usage.get("output_tokens"):
                        items.append(f"output={usage['output_tokens']:,}")
                    if usage.get("cache_creation_input_tokens"):
                        items.append(f"cache_creation={usage['cache_creation_input_tokens']:,}")
                    if usage.get("cur_context"):
                        items.append(
                            f"Context={usage['cur_context']:,}"
                            f"/{usage['cur_context_percentage']:.2%}"
                        )
                    if usage.get("next_context_estimate"):
                        items.append(
                            f"Estimate={usage['next_context_estimate']:,}"
                            f"/{usage['next_context_estimate_percentage']:.2%}"
                        )
                    blocks.append(f"[Usage] {'|'.join(items)}")
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

    def __init__(self):
        self._console: Console = Console()
        self._renderer: ConsoleEventRenderer = ConsoleEventRenderer(self._console)
        self._promptkit: PromptSession = PromptSession(multiline=True)

    def run(self) -> None:
        mainagent = Agent(
            aid="main", 
            models=["MiniMax/MiniMax-M2.7"],
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
            middleware_factories=[TaskTrackerFactory(self, max_idle_steps=3)],
            theme_color="dodger_blue2"
        )
        # subscribe all events published
        mainsession.subscribe(self)

        runner = AgentRunner(ExecutionEngine())

        # support subagent which will also publish events to tui
        mainsession.add_tool_provider(SubagentToolProvider(mainsession, mainagent, subs=[tui]))

        while True:

            # Get user input
            try:
                prompt = self._input(mainsession.sid, 
                                     mainsession.rounds, 
                                     mainsession.theme_color)
            except (EOFError, KeyboardInterrupt):
                return

            if not prompt:
                continue

            prompt = prompt.strip()
            if prompt.strip().lower() in {"exit", "quit", "q"}:
                return

            # Handle the prompt
            runner.run(
                session=mainsession, 
                agent=mainagent, 
                user_input=prompt
            )

            print()

    def input(self, prompt: str = "") -> str:
        """Implementation of HumanInput
        """
        return self._promptkit.prompt(prompt) 

    def handle_event(self, event: Event) -> None:
        self._renderer.print(event)

    def get_interested_events(self) -> set[type[Event]]:
        return set([Event])

    def _input(self, 
              session_id: SessionID, 
              session_rounds: int,
              theme_color: str) -> str:
        header = self._renderer.render_header(
            paths=[f"session:{session_id.id}", f"round:{session_rounds + 1}"],
            color=theme_color
        )
        self._console.print(header)
        self._console.print(f"[bold cyan]Input:[/bold cyan] ", end="")
        return self.input()


if __name__ == "__main__":

    tui = TUI()

    tui.run()