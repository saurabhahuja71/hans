from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
import time
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
            if target.is_dir():
                return f"Error reading {path!r}: path is a directory; use list_directory instead"
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


def _bounded_lines(lines: list[str], *, max_results: int | None = None) -> str:
    budget = tool_result_token_budget()
    limited = lines if max_results is None else lines[:max_results]
    if not limited:
        return ""
    shown: list[str] = []
    for index, line in enumerate(limited):
        candidate = "\n".join([*shown, line])
        remaining = len(limited) - index - 1
        if not remaining and estimate_tokens(candidate) <= budget:
            return candidate
        notice = f"... {remaining} result(s) omitted to fit the tool result budget."
        rendered = f"{candidate}\n{notice}" if candidate else notice
        if estimate_tokens(rendered) > budget:
            break
        shown.append(line)
    omitted = len(limited) - len(shown)
    notice = f"... {omitted} result(s) omitted to fit the tool result budget."
    result = "\n".join(shown)
    return f"{result}\n{notice}" if result else notice


_SEARCH_IGNORED_DIRECTORIES = {
    ".eggs",
    ".git",
    ".hans-tmp",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}


def _search_match_line(path: Path, line_number: int, line: str) -> str:
    prefix = f"{path}:{line_number}: "
    suffix = " ... [matching line truncated]"
    maximum_line_length = max(0, tool_result_token_budget() * 3 - len(prefix) - len(suffix))
    if len(line) <= maximum_line_length:
        return prefix + line
    return prefix + line[:maximum_line_length] + suffix


def _search_candidates(workspace: Path, target: Path):
    if target.is_file():
        yield target
        return
    try:
        if any(part in _SEARCH_IGNORED_DIRECTORIES for part in target.relative_to(workspace).parts):
            return
    except ValueError:
        return
    for candidate in sorted(target.iterdir(), key=lambda entry: entry.name):
        try:
            candidate.resolve().relative_to(workspace)
        except (OSError, ValueError):
            continue
        if candidate.is_symlink() and candidate.is_dir():
            continue
        if candidate.is_dir():
            if candidate.name not in _SEARCH_IGNORED_DIRECTORIES:
                yield from _search_candidates(workspace, candidate)
        elif candidate.is_file():
            yield candidate


def make_list_directory_tool(workspace: Path):
    @function_tool
    async def list_directory(path: str = ".") -> str:
        """List direct workspace-directory entries in sorted order.

        Args:
            path: A relative directory path from the workspace root.
        """
        try:
            target = resolve_workspace_path(workspace, path)
            if target.is_file():
                return f"Error listing {path!r}: path is a file; use read_file instead"
            if not target.is_dir():
                return f"Error listing {path!r}: directory does not exist"
            entries = []
            for entry in sorted(target.iterdir(), key=lambda candidate: candidate.name):
                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir():
                    kind = "directory"
                elif entry.is_file():
                    kind = "file"
                else:
                    kind = "other"
                entries.append(f"{kind}: {entry.name}")
            return _bounded_lines(entries)
        except (OSError, WorkspaceError) as exc:
            return f"Error listing {path!r}: {exc}"

    return list_directory


def make_search_files_tool(workspace: Path):
    @function_tool
    async def search_files(query: str, path: str = ".", max_results: int = 50) -> str:
        """Search UTF-8 text files in the workspace and return matching path:line text.

        Binary and unreadable files are skipped. Results are sorted and constrained by both
        max_results and the tool result context budget.

        Args:
            query: Literal text to find. It must not be empty.
            path: A relative file or directory path from the workspace root.
            max_results: Maximum matching lines to return, from 1 through 100.
        """
        if not query:
            return "Error searching: query must be a non-empty string"
        if max_results < 1 or max_results > 100:
            return "Error searching: max_results must be between 1 and 100"
        try:
            target = resolve_workspace_path(workspace, path)
            if not target.exists():
                return f"Error searching {path!r}: path does not exist"
            matches: list[str] = []
            for candidate in _search_candidates(workspace, target):
                try:
                    relative = candidate.relative_to(workspace)
                    file_matches: list[str] = []
                    binary = False
                    with candidate.open(encoding="utf-8") as source:
                        for line_number, line in enumerate(source, start=1):
                            if "\x00" in line:
                                binary = True
                                break
                            line = line.rstrip("\r\n")
                            if query in line and len(matches) + len(file_matches) < max_results:
                                file_matches.append(_search_match_line(relative, line_number, line))
                    if binary:
                        continue
                    matches.extend(file_matches)
                except (OSError, UnicodeError, ValueError):
                    continue
                if len(matches) >= max_results:
                    return _bounded_lines(matches, max_results=max_results)
            return _bounded_lines(matches, max_results=max_results)
        except (OSError, WorkspaceError) as exc:
            return f"Error searching {path!r}: {exc}"

    return search_files


def make_replace_in_file_tool(workspace: Path):
    @function_tool
    async def replace_in_file(path: str, old_text: str, new_text: str) -> str:
        """Replace exactly one literal text occurrence in an existing UTF-8 workspace file.

        Args:
            path: A relative file path from the workspace root.
            old_text: Existing text that must occur exactly once.
            new_text: Replacement text.
        """
        if not old_text:
            return "Error replacing: old_text must be a non-empty string"
        try:
            target = resolve_workspace_path(workspace, path)
            if not target.is_file():
                return f"Error replacing {path!r}: file does not exist"
            source_stat = target.stat()
            text = target.read_text(encoding="utf-8")
            occurrences = text.count(old_text)
            if occurrences != 1:
                return f"Error replacing {path!r}: old_text must occur exactly once (found {occurrences})"
            replacement = text.replace(old_text, new_text, 1)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
                ) as temporary:
                    temporary.write(replacement)
                    temporary_path = Path(temporary.name)
                os.chmod(temporary_path, source_stat.st_mode)
                os.utime(
                    temporary_path,
                    ns=(
                        temporary_path.stat().st_atime_ns,
                        max(time.time_ns(), source_stat.st_mtime_ns + 1_000_000_000),
                    ),
                )
                os.replace(temporary_path, target)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return f"Replaced text in {path}"
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"Error replacing {path!r}: {exc}"

    return replace_in_file


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
    for argument in command.split():
        if argument == "./...":
            continue
        if any(char in _SHELL_META for char in argument):
            return (
                "Error: run_command executes a direct argv command, not a shell. "
                "Pipes, redirects, &&, ||, globs, and other shell syntax are not supported."
            )
    if "\n" in command or "\r" in command:
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
    async def run_command(command: str, purpose: str = "inspect") -> str:
        """Run one direct command in the workspace and return its exit code and output.

        The command is split into argv and executed without a shell. Pipes, redirects,
        &&, ||, globs, and substitution are rejected. The working directory is the
        workspace. The command does not receive API keys or the rest of the process
        environment. purpose must be `inspect` for investigation or `verify` for a
        command intended to validate requested behavior.

        Args:
            command: Program and arguments, for example `go test ./...`.
            purpose: `inspect` or `verify`; defaults to `inspect`.
        """
        if purpose not in {"inspect", "verify"}:
            return "Error: purpose must be either 'inspect' or 'verify'"
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
        notice = (
            "\n---\n"
            "command output truncated to fit the context budget. "
            f"original_tokens≈{estimate_tokens(result)} budget_tokens={budget}. "
            "This is not a summary. Re-run a narrower command for the omitted output.\n"
        )
        return result[: max(0, budget * 3 - len(notice))] + notice

    return run_command
