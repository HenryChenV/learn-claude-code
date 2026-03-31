import os

from dotenv import load_dotenv

from .agent import AnthropicAgent
from .tools.impls import run_bash, read_file, write_file, edit_file
from .session import AssistantErrorEvent, AssitantTextEvent, Session, Event, ThinkingEvent, ToolResultEvent, ToolUseEvent, UnknownEvent, UserPromptEvent


# init env
load_dotenv(override=True)
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
if BASE_URL:
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
MODEL = os.environ["MODEL_ID"]


SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


class TUI:

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
                self.output(event)

            print()

    def output(self, event: Event) -> str | None:
        formatted = self.format(event)
        if formatted is not None:
            print(formatted)

    def format(self, event: Event) -> str | None:
        match event:
            case UserPromptEvent(prompt=prompt):
                return None

            case ThinkingEvent(thinking=thinking):
                return f"\033[35mThinking: {thinking}\033[0m"

            case AssitantTextEvent(content=content):
                return f"\033[32mOutput: {content}\033[0m"

            case AssistantErrorEvent(error=error):
                return f"\033[31mAssistant Error: {error}\033[0m"

            case ToolUseEvent(tool_name=tool_name, tool_use_id=tool_use_id, tool_input=tool_input):
                return f"\033[33mToolUse[{tool_use_id}]: {tool_name}({tool_input})\033[0m"

            case ToolResultEvent(tool_name=tool_name, tool_use_id=tool_use_id, tool_output=tool_output):
                return f"\033[34mToolResult[{tool_use_id}]: {tool_name} -> {tool_output}\033[0m"

            case UnknownEvent(data=data):
                return f"\033[31mUnknown event: {data}\033[0m"

            case _:
                return f"\033[31mUnhandled event: {event}\033[0m"       


if __name__ == "__main__":
    TUI().run(Session(agent=AnthropicAgent(
        name="main", 
        base_url=BASE_URL, 
        model_id=MODEL,
        tools=[run_bash, read_file, write_file, edit_file],
        system_prompt=SYSTEM_PROMPT
    )))