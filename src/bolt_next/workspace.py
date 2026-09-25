from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
from pathlib import Path

from agents import function_tool

from bolt_next.context_budget import estimate_tokens, tool_result_token_budget


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


def _range_result(path: str, lines: list[str], start_line: int, end_line: int) -> str:
    total = len(lines)
    body = "\n".join(lines[start_line - 1 : end_line])
    shown_end = start_line + body.count("\n") if body else start_line - 1
    if body:
        shown_end = start_line + body.count("\n")
    header = (
        f"path: {path}\n"
        f"total_lines: {total}\n"
        f"returned_range: {start_line}-{shown_end}\n"
    )
    if shown_end < total:
        header += (
            f"remaining_ranges: {shown_end + 1}-{total}\n"
            f"request_next: read_file path={path} start_line={shown_end + 1}\n"
        )
    else:
        header += "remaining_ranges: none\n"
    header += (
        "note: the lines below are the authoritative file text for returned_range. "
        "Lines outside that range were not included.\n"
        "---\n"
    )
    return header + body


def _fitting_end(path: str, lines: list[str], start_line: int, end_line: int) -> int | None:
    budget = tool_result_token_budget()
    if estimate_tokens(_range_result(path, lines, start_line, end_line)) <= budget:
        return end_line
    low = start_line
    high = end_line
    best = None
    while low <= high:
        mid = (low + high) // 2
        if estimate_tokens(_range_result(path, lines, start_line, mid)) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


def make_read_file_tool(workspace: Path):
    @function_tool
    async def read_file(path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a UTF-8 text file inside the workspace.

        Small files are returned in full. A large file is returned as an explicit
        line range, never as a summary. Use start_line and end_line to inspect
        another range. end_line 0 means "as far as the context budget allows".

        Args:
            path: A relative path from the workspace root.
            start_line: First line to return, starting at 1.
            end_line: Last line to return, inclusive. 0 selects a budget-sized range.
        """
        try:
            target = resolve_workspace_path(workspace, path)
            if not target.is_file():
                return f"Error: file does not exist: {path}"
            text = target.read_text(encoding="utf-8")
            lines = text.splitlines()
            total = len(lines)
            if total == 0:
                return ""
            if start_line < 1 or start_line > total:
                return (
                    f"Error reading {path!r}: start_line {start_line} is outside 1-{total}. "
                    f"total_lines: {total}"
                )
            explicit = end_line > 0
            if not explicit:
                end_line = total
            end_line = min(end_line, total)
            if end_line < start_line:
                return f"Error reading {path!r}: end_line must be >= start_line"
            if (
                not explicit
                and start_line == 1
                and end_line == total
                and estimate_tokens(text) <= tool_result_token_budget()
            ):
                return text
            fitted = _fitting_end(path, lines, start_line, end_line)
            if fitted is None:
                return (
                    f"Error reading {path!r}: requested range {start_line}-{end_line} "
                    f"does not fit in the tool result budget of {tool_result_token_budget()} tokens. "
                    f"total_lines: {total}. Request a smaller end_line. "
                    "No partial source was returned."
                )
            if explicit and fitted < end_line:
                return (
                    f"Error reading {path!r}: requested range {start_line}-{end_line} "
                    f"does not fit in the tool result budget of {tool_result_token_budget()} tokens. "
                    f"total_lines: {total}. A range ending at {fitted} fits. "
                    "No partial source was returned."
                )
            if fitted == total and start_line == 1 and estimate_tokens(text) <= tool_result_token_budget():
                return text
            return _range_result(path, lines, start_line, fitted)
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
        result = (
            f"exit_code={completed.returncode}\n"
            f"stdout:\n{completed.stdout}"
            f"stderr:\n{completed.stderr}"
        )
        budget = tool_result_token_budget()
        if estimate_tokens(result) <= budget:
            return result
        keep = budget * 4
        return (
            result[:keep]
            + "\n---\n"
            + "command output truncated to fit the context budget. "
            + f"original_tokens≈{estimate_tokens(result)} budget_tokens={budget}. "
            + "This is not a summary. Re-run a narrower command for the omitted output.\n"
        )

    return run_command
