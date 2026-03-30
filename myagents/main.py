import os

from dotenv import load_dotenv

from agent import AnthropicAgent
from tools import run_bash, read_file, write_file, edit_file
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

            # Get user input
            try:
                prompt = input(f"\033[36mInput {i}: \033[0m")
            except (EOFError, KeyboardInterrupt):
                return session.close()
            if prompt.strip().lower() in {"exit", "quit", "q", ""}:
                return session.close()

            # Handle the prompt and get the final content
            content = session.handle(prompt)

            # Display the final output
            print("\033[32mOutput:\033[0m")
            if isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        print(f"Text: {block.text}")
                    elif hasattr(block, "thinking"):
                        print(f"Thinking: {block.thinking}")
                    else:
                        print(f"Unknown: {block}")
            else:
                print(content)

            print()


if __name__ == "__main__":
    TUI.run(
        Session(
            agent=AnthropicAgent(name="main", 
                base_url=BASE_URL, 
                model_id=MODEL,
                tools=[run_bash, read_file, write_file, edit_file],
                system_prompt=SYSTEM_PROMPT
            )
        )
    )