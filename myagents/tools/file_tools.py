from pathlib import Path

from .base import FunctionTool


WORKDIR = Path.cwd()


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not str(path).startswith(str(WORKDIR)):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@FunctionTool.wrapper
def read_file(path: str, limit: int | None = None) -> str:
    """Read file contents with optional line limit. 
    """
    try:
        with open(safe_path(path), "r") as f:
            lines = []
            for line in f:
                lines.append(line)
                if limit is not None and len(lines) >= limit:
                    lines.append(f"... ({len(lines) - limit} more lines)")
                    break
            return "\n".join(lines)
    except Exception as e:
        return f"Error: failed to read {path}: {e}"


@FunctionTool.wrapper
def write_file(path: str, content: str) -> str:
    """Write content to a file.
    """
    try:
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error: failed to write to {path}: {e}"


@FunctionTool.wrapper
def edit_file(path: str, new_content: str) -> str:
    """Edit a file by replacing exact text in file.
    """
    try:
        p = safe_path(path)
        old_content = ""
        if p.exists():
            with open(p, "r") as f:
                old_content = f.read()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            f.write(new_content)
        return f"Successfully edited {path}."
    except Exception as e:
        return f"Error: failed to edit {path}: {e}"