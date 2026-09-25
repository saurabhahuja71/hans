from __future__ import annotations

import asyncio
import os
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


# Characters that only have meaning in a shell. run_command never invokes one.
_SHELL_META = set("|;&<>$`(){}[]*?~!#\\")
_SHELL_PROGRAMS = {"sh", "bash", "dash", "zsh", "fish", "ksh", "csh", "tcsh"}
# Developer tools still need a few variables. Secrets and the rest of the process
# environment are not copied into the command.
_COMMAND_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "GO111MODULE",
    "GOROOT",
    "GOPATH",
    "GOPROXY",
    "GOSUMDB",
    "GOPRIVATE",
    "GOCACHE",
    "GOMODCACHE",
    "GOTOOLCHAIN",
    "GOFLAGS",
    "CGO_ENABLED",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
)


def command_environment(workspace: Path) -> dict[str, str]:
    """Environment passed to run_command. HOME and TMPDIR stay inside the workspace."""
    env = {key: os.environ[key] for key in _COMMAND_ENV_KEYS if os.environ.get(key)}
    tmp = workspace / ".hans-tmp"
    tmp.mkdir(exist_ok=True)
    env["HOME"] = str(workspace)
    env["TMPDIR"] = str(tmp)
    env["PWD"] = str(workspace)
    return env


def reject_shell_syntax(command: str) -> str | None:
    if any(char in _SHELL_META or char in "\n\r" for char in command):
        return (
            "Error: run_command executes a direct argv command, not a shell. "
            "Pipes, redirects, &&, ||, globs, and other shell syntax are not supported."
        )
    return None


def reject_workspace_escape(workspace: Path, args: list[str]) -> str | None:
    program = Path(args[0]).name
    if program in _SHELL_PROGRAMS:
        return "Error: run_command does not run a shell. Pass the program and its arguments directly."
    for arg in args:
        if arg.startswith("-"):
            continue
        path = Path(arg)
        if ".." in path.parts:
            return f"Error: command argument escapes the workspace: {arg}"
        if path.is_absolute():
            try:
                path.resolve().relative_to(workspace)
            except ValueError:
                return f"Error: command argument is outside the workspace: {arg}"
    return None


def make_run_command_tool(workspace: Path):
    @function_tool
    async def run_command(command: str) -> str:
        """Run one direct command in the workspace and return its exit code and output.

        The command is split into argv and executed without a shell. Pipes, redirects,
        &&, ||, globs, and substitution are rejected. The working directory is the
        workspace. The command does not receive API keys or the rest of the process
        environment.

        Args:
            command: Program and arguments, for example `go test ./...`.
        """
        if not command or not command.strip() or "\x00" in command:
            return "Error: command must be a non-empty string"
        shell_error = reject_shell_syntax(command)
        if shell_error:
            return shell_error
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return f"Error: could not parse command: {exc}"
        if not args:
            return "Error: command must be a non-empty string"
        escape_error = reject_workspace_escape(workspace, args)
        if escape_error:
            return escape_error

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                args,
                cwd=workspace,
                env=command_environment(workspace),
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
