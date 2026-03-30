import os

from dotenv import load_dotenv

from agent import AnthropicAgent
from tools import ToolManager, bash
from session import Session


# init env
load_dotenv(override=True)
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
if BASE_URL:
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
MODEL = os.environ["MODEL_ID"]


SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


class TUI:

    @staticmethod
    def run(session: Session) -> None:
        i = 0
        while True:
            i = i + 1

            try:
                prompt = input(f"\033[36mInput {i}: \033[0m")
            except (EOFError, KeyboardInterrupt):
                return session.close()
            if prompt.strip().lower() in {"exit", "quit", "q", ""}:
                return session.close()

            session.handle(prompt)


if __name__ == "__main__":
    TUI.run(
        Session(
            agent=AnthropicAgent(name="main", 
                base_url=BASE_URL, 
                model_id=MODEL,
                tools=[bash],
                system_prompt=SYSTEM_PROMPT
            )
        )
    )