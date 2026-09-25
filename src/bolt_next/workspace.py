from __future__ import annotations

import asyncio
import shlex
import subprocess
from pathlib import Path

from agents import function_tool


class WorkspaceError(ValueError):
    """An attempted workspace access was invalid."""


def resolve_workspace(path: str | Path | None = None) -> Path:
    workspace = Path(path or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise WorkspaceError(f"Workspace is not a directory: {workspace}")
    return workspace


def resolve_workspace_path(workspace: Path, path: str) -> Path:
    if not path or "\x00" in path:
        raise WorkspaceError("Path must be a non-empty relative path")
    candidate = (workspace / path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise WorkspaceError("Path is outside the workspace") from exc
    return candidate


def make_read_file_tool(workspace: Path):
    @function_tool
    async def read_file(path: str) -> str:
        """Read a UTF-8 text file inside the workspace.

        Args:
            path: A relative path from the workspace root.
        """
        try:
            target = resolve_workspace_path(workspace, path)
            if not target.is_file():
                return f"Error: file does not exist: {path}"
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"Error reading {path!r}: {exc}"

    return read_file


def make_write_file_tool(workspace: Path):
    @function_tool
    async def write_file(path: str, content: str) -> str:
        """Create or replace a UTF-8 text file inside the workspace.

        Args:
            path: A relative path from the workspace root.
            content: The full file contents to write.
        """
        try:
            target = resolve_workspace_path(workspace, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.parent.resolve().is_relative_to(workspace):
                return f"Error writing {path!r}: Path is outside the workspace"
            target.write_text(content, encoding="utf-8")
            return f"Wrote {path} ({len(content.encode('utf-8'))} bytes)"
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"Error writing {path!r}: {exc}"

    return write_file


def make_run_command_tool(workspace: Path):
    @function_tool
    async def run_command(command: str) -> str:
        """Run a command in the workspace and return its exit code and output.

        Args:
            command: Command and arguments, for example `go run main.go`.
        """
        if not command or not command.strip() or "\x00" in command:
            return "Error: command must be a non-empty string"
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return f"Error: could not parse command: {exc}"
        if not args:
            return "Error: command must be a non-empty string"

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                args,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=120,
            )

        try:
            completed = await asyncio.to_thread(execute)
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120 seconds"
        except FileNotFoundError:
            return f"Error: command not found: {args[0]}"
        except OSError as exc:
            return f"Error running command: {exc}"
        return (
            f"exit_code={completed.returncode}\n"
            f"stdout:\n{completed.stdout}"
            f"stderr:\n{completed.stderr}"
        )

    return run_command
