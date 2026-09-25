from pathlib import Path

import pytest

from bolt_next.workspace import (
    WorkspaceError,
    make_read_file_tool,
    make_run_command_tool,
    make_write_file_tool,
    resolve_workspace_path,
)
import json


def invoke(tool, arguments: str):
    # The SDK invokes this wrapped function after parsing the structured JSON arguments.
    return __import__("asyncio").run(tool.__wrapped__(**json.loads(arguments)))


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


def test_write_file_creates_file(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    result = invoke(tool, '{"path":"main.go","content":"package main\\n"}')
    assert "Wrote main.go" in result
    assert (tmp_path / "main.go").read_text(encoding="utf-8") == "package main\n"


def test_write_file_rejects_traversal(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    result = invoke(tool, '{"path":"../secret.txt","content":"nope"}')
    assert "outside the workspace" in result
    assert not (tmp_path.parent / "secret.txt").exists()


def test_run_command_returns_stdout(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"printf HELLO_HANS"}')
    assert "exit_code=0" in result
    assert "HELLO_HANS" in result


@pytest.mark.parametrize(
    "command",
    [
        "go test ./... 2>&1",
        "go test ./... && gofmt",
        "cat a | head",
        "echo $(pwd)",
        "ls *.go",
    ],
)
def test_run_command_rejects_shell_syntax(tmp_path: Path, command: str) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, json.dumps({"command": command}))
    assert "not a shell" in result
    assert "exit_code" not in result


def test_run_command_rejects_parent_path(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"ls .."}')
    assert "escapes the workspace" in result
    assert "exit_code" not in result


def test_run_command_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"cat /etc/passwd"}')
    assert "outside the workspace" in result
    assert "root:" not in result


def test_run_command_hides_process_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "super-secret-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"env"}')
    assert "exit_code=0" in result
    assert "super-secret-key" not in result
    assert "openai-secret" not in result
    assert f"HOME={tmp_path}" in result


def test_symlink_outside_workspace_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        resolve_workspace_path(tmp_path, "link.txt")
