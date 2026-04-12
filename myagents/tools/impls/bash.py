import os
import subprocess

from myagents.capability import Capability

from ..core import FunctionTool


DANGEROUS_COMMANDS = frozenset({"rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"})


@FunctionTool.wrapper(name="bash", required_capabilities=Capability.BASH.value)
def run_bash(command: str) -> str:
    """ Run a shell command
    """
    if not command:
        return "Error: Command cannot be empty."

    for d in DANGEROUS_COMMANDS:
        if d in command:
            return f"Error: Dangerous command blocked. Your command contained {d}"

    try:
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
