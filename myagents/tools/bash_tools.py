import os
import subprocess

from .base import Tool


class BashTool(Tool):

    DANGEROUS_COMMANDS = frozenset({"rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"})

    def __init__(self) -> None:
        super().__init__(
            name="bash", 
            description="Run a shell command.", 
            input_schema={
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            }
        )

    def _run(self, **kwargs) -> str:
        command = kwargs.get("command", "")
        if not command:
            return "Error: Command cannot be empty."

        if any(d in command for d in self.DANGEROUS_COMMANDS):
            return "Error: Dangerous command blocked"
        try:
            r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                               capture_output=True, text=True, timeout=120)
            out = (r.stdout + r.stderr).strip()
            return out[:50000] if out else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Timeout (120s)"


bash = BashTool()
