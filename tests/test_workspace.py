from pathlib import Path

import pytest

from bolt_next.workspace import WorkspaceError, make_read_file_tool, resolve_workspace_path
import json


def invoke(tool, arguments: str):
    # The SDK invokes this wrapped function after parsing the structured JSON arguments.
    return tool.__wrapped__(**json.loads(arguments))


def test_workspace_path_validation(tmp_path: Path) -> None:
    assert resolve_workspace_path(tmp_path, "main.go") == tmp_path / "main.go"


def test_read_file_success(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tool = make_read_file_tool(tmp_path)
    assert invoke(tool, '{"path":"note.txt"}') == "hello"


def test_read_file_missing_file(tmp_path: Path) -> None:
    tool = make_read_file_tool(tmp_path)
    result = invoke(tool, '{"path":"missing.txt"}')
    assert "does not exist" in result


def test_path_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        resolve_workspace_path(tmp_path, "../secret.txt")


def test_symlink_outside_workspace_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        resolve_workspace_path(tmp_path, "link.txt")
