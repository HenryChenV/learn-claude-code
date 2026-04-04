from pathlib import Path

from ..core import FunctionTool


WORKDIR = Path.cwd()


def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not str(path).startswith(str(WORKDIR)):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


@FunctionTool.wrapper(required_capabilities="file.read")
def read_file(path: str, limit: int = 50000) -> str:
    """Read file contents with optional line limit (default is 50000). 
    """
    try:
        lines: list[str] = []
        with safe_path(path).open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= limit:
                    return "".join(lines) + f"... (truncated, showing first {limit} lines)\n"
                lines.append(line)

        return "".join(lines)

    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except UnicodeDecodeError:
        return f"Error: file is not valid UTF-8 text: {path}"
    except Exception as e:
        return f"Error: failed to read {path}: {e}"


@FunctionTool.wrapper(required_capabilities="file.write")
def write_file(path: str, content: str) -> str:
    """Write content to a file.
    """
    try:
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote {len(content)} characters to {path}"
    except Exception as e:
        return f"Error: failed to write to {path}: {e}"


@FunctionTool.wrapper(required_capabilities="file.edit")
def edit_file(path: str, old_content, new_content: str) -> str:
    """Edit a file by replacing exact text in file.
    """
    try:
        fp = safe_path(path)
        content = fp.read_text()
        if old_content not in content:
            return f"Error: Text not found in {path}"
        fp.write_text(content.replace(old_content, new_content, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: failed to edit {path}: {e}"