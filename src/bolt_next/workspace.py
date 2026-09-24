from __future__ import annotations

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
    def read_file(path: str) -> str:
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
