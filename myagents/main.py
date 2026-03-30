import os
import subprocess

from anthropic import Anthropic
from anthropic.types import Message, TextBlock, ThinkingBlock
from dotenv import load_dotenv


# init env
load_dotenv(override=True)
BASE_URL = os.getenv("ANTHROPIC_BASE_URL")
if BASE_URL:
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
MODEL = os.environ["MODEL_ID"]


SYSTEM_PROMPT = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."


DANGEROUS_COMMANDS = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]


def run_bash(command: str) -> str:
    if any(d in command for d in DANGEROUS_COMMANDS):
        return "Error: Dangerous command blocked"
    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },  
    }
]


AVAILABLE_TOOLS: dict[str, dict] = {
    "bash": {
        "name": "bash",
        "description": "Run a shell command.",
        "func": run_bash,
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}


def truncate(output: str, max_length: int = 50000) -> str:
    if len(output) > max_length:
        return f"{output[:max_length]}..."
    else:
        return output


class Agent:

    _name: str
    _client: Anthropic
    _model_id: str | None
    _tools: list[dict]

    def __init__(self, name: str, base_url: str | None, model_id: str | None, tools: list[str] = []) -> None:
        self._name = name
        self._client = Anthropic(base_url=base_url)
        self._model_id = model_id
        self._tools = self._build_tools(tools)

    def name(self) -> str:
        return self._name

    def step(self, inputs: list[dict]) -> Message:
        return self._client.messages.create(
            model=self._model_id, # type: ignore
            system=SYSTEM_PROMPT,
            messages=inputs, # type: ignore
            tools=self._tools, # type: ignore
            max_tokens=8000,
        )

    def _build_tools(self, tool_names):
        tools = []
        for name in tool_names:
            if name in AVAILABLE_TOOLS:
                t = AVAILABLE_TOOLS[name]
                tools.append({
                    "name": t["name"],
                    "description": t["description"],
                    "input_schema": t["input_schema"]
                })
            else:
                raise ValueError(f"Tool '{name}' not found in AVAILABLE_TOOLS.")
        return tools

    def close(self):
        self._client.close()


class Session:

    _main_agent: Agent
    _sub_agents: dict[str, Agent]

    def __init__(self, agent: Agent) -> None:
        self._history = []
        self._main_agent = agent

    def handle(self, prompt: str) -> None:
        # Append user turn
        input_message = {"role": "user", "content": prompt}
        self._history.append(input_message)

        # Run the agent loop until it stops
        self._agent_loop()

        # Display the final output
        print("\033[32mOutput:\033[0m")
        final_message = self._history[-1]
        content = final_message["content"]
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

    def _agent_loop(self):
        while True:
            # Agent takes a step
            response = self._main_agent.step(self._history)

            # Append assistant turn
            self._history.append({"role": "assistant", "content": response.content})

            # If the model didn't call a tool, we're done
            if response.stop_reason != "tool_use":
                return

            # Execute each tool call, collect results, or call sub-agents as needed, and append results to history for next step
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    # block.name
                    print(f"\033[33mTool$ {block.input['command']}\033[0m")
                    # only one tool --- run_bash
                    output = run_bash(block.input["command"]) # type: ignore
                    print(truncate(output))
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            self._history.append({"role": "user", "content": results})

    def exit(self):
        print("Exiting.")

        self._main_agent.close()

        exit(0)



class TUI:

    @staticmethod
    def run(session: Session) -> None:
        while True:

            try:
                prompt = input("\033[36mInput : \033[0m")
            except (EOFError, KeyboardInterrupt):
                return session.exit()
            if prompt.strip().lower() in {"exit", "quit", "q", ""}:
                return session.exit()

            session.handle(prompt)

if __name__ == "__main__":
    TUI.run(
        Session(
            agent=Agent(name="main", 
                base_url=BASE_URL, 
                model_id=MODEL,
                tools=["bash"],
            )
        )
    )