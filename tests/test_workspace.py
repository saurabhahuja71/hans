from pathlib import Path

import pytest

from bolt_next.context_budget import estimate_tokens, tool_result_token_budget
from bolt_next.workspace import (
    WorkspaceError,
    _bounded_lines,
    make_list_directory_tool,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_run_command_tool,
    make_search_files_tool,
    make_write_file_tool,
    reject_shell_syntax,
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


def test_list_directory_is_nonrecursive_sorted_and_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "zeta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "nested.txt").write_text("nested", encoding="utf-8")
    for index in range(100):
        (tmp_path / f"long-entry-{index:03d}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")

    result = invoke(make_list_directory_tool(tmp_path), '{"path":"."}')

    lines = result.splitlines()
    entries = [line for line in lines if not line.startswith("...")]
    assert entries == sorted(entries)
    assert "alpha" in entries
    assert any(line.startswith("long-entry-") for line in entries)
    assert "nested.txt" not in result
    assert estimate_tokens(result) <= tool_result_token_budget()


def test_list_directory_rejects_workspace_escape(tmp_path: Path) -> None:
    result = invoke(make_list_directory_tool(tmp_path), '{"path":"../"}')

    assert "outside the workspace" in result


def test_search_files_returns_sorted_matches_and_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("needle beta\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("needle alpha\nother needle\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"needle\x00hidden")

    result = invoke(make_search_files_tool(tmp_path), '{"query":"needle","max_results":2}')

    assert result.splitlines() == ["a.txt:1: needle alpha", "a.txt:2: other needle"]
    assert "binary.bin" not in result


def test_search_files_rejects_escape_and_honors_context_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(100):
        (tmp_path / f"file-{index:03d}.txt").write_text("needle " + "x" * 30, encoding="utf-8")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")
    tool = make_search_files_tool(tmp_path)

    result = invoke(tool, '{"query":"needle","max_results":100}')
    escaped = invoke(tool, '{"query":"needle","path":"../"}')

    assert estimate_tokens(result) <= tool_result_token_budget()
    assert "outside the workspace" in escaped


def test_search_files_skips_vcs_and_cache_directories_without_reading_whole_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "visible.txt").write_text("needle visible\n", encoding="utf-8")
    for directory in (".git", ".hans-tmp", "__pycache__"):
        ignored = tmp_path / directory
        ignored.mkdir()
        (ignored / "ignored.txt").write_text("needle ignored\n", encoding="utf-8")

    def read_bytes(*_args, **_kwargs):
        raise AssertionError("search_files must read files line by line")

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    result = invoke(make_search_files_tool(tmp_path), '{"query":"needle"}')

    assert result == "visible.txt:1: needle visible"


def test_search_files_truncates_a_huge_matching_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "huge.txt").write_text("needle " + "x" * 10_000, encoding="utf-8")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")

    result = invoke(make_search_files_tool(tmp_path), '{"query":"needle"}')

    assert result.startswith("huge.txt:1: needle ")
    assert "[matching line truncated]" in result
    assert estimate_tokens(result) <= tool_result_token_budget()


def test_bounded_lines_reports_an_oversized_omitted_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")

    result = _bounded_lines(["x" * 10_000])

    assert result == "... 1 result(s) omitted to fit the tool result budget."
    assert estimate_tokens(result) <= tool_result_token_budget()


def test_replace_in_file_requires_one_match_and_replaces_atomically(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("before\nold value\nafter\n", encoding="utf-8")
    target.chmod(0o755)
    original_stat = target.stat()
    original_inode = original_stat.st_ino
    original_mode = original_stat.st_mode & 0o777
    original_mtime_ns = original_stat.st_mtime_ns
    tool = make_replace_in_file_tool(tmp_path)

    result = invoke(tool, '{"path":"main.py","old_text":"old value","new_text":"new value"}')

    assert result == "Replaced text in main.py"
    assert target.read_text(encoding="utf-8") == "before\nnew value\nafter\n"
    assert target.stat().st_ino != original_inode
    assert target.stat().st_mode & 0o777 == original_mode
    assert target.stat().st_mtime_ns >= original_mtime_ns + 1_000_000_000
    (tmp_path / "duplicate.txt").write_text("old\nold\n", encoding="utf-8")
    duplicate = invoke(tool, '{"path":"duplicate.txt","old_text":"old","new_text":"other"}')
    missing = invoke(tool, '{"path":"main.py","old_text":"absent","new_text":"other"}')
    assert "exactly once (found 2)" in duplicate
    assert "exactly once (found 0)" in missing


def test_replace_in_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-replace.txt"
    outside.write_text("old", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)

    result = invoke(make_replace_in_file_tool(tmp_path), '{"path":"link.txt","old_text":"old","new_text":"new"}')

    assert "outside the workspace" in result
    assert outside.read_text(encoding="utf-8") == "old"


def test_run_command_returns_stdout(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"printf HELLO_HANS"}')
    assert "exit_code=0" in result
    assert "HELLO_HANS" in result


def test_run_command_allows_literal_go_package_pattern_without_running_go() -> None:
    assert reject_shell_syntax("go test ./...") is None


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
