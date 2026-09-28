from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path

import pytest
from agents import Agent, Runner, SQLiteSession, set_tracing_disabled
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from bolt_next.workspace import (
    make_list_directory_tool,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_run_command_tool,
    make_search_files_tool,
    make_write_file_tool,
)


@pytest.fixture(autouse=True)
def avoid_broken_executor_shutdown(monkeypatch):
    """Keep tests deterministic in this environment's broken asyncio executor shutdown.

    SDK behavior remains under test; only the stdlib worker dispatch is made inline so pytest
    can exit. Production code does not patch asyncio.
    """

    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    set_tracing_disabled(True)


def run(coro):
    return asyncio.run(coro)


def make_agent(model: ScriptedModel, workspace: Path) -> Agent:
    return Agent(
        name="HANS test agent",
        instructions="Use read_file when asked to inspect a file.",
        model=model,
        tools=[
            make_list_directory_tool(workspace),
            make_search_files_tool(workspace),
            make_read_file_tool(workspace),
            make_replace_in_file_tool(workspace),
            make_write_file_tool(workspace),
            make_run_command_tool(workspace),
        ],
    )


def test_structured_read_file_call_executes_and_reaches_next_model_turn(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("HANS_KAGGLE_READ_TEST_123", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("read_file", {"path": "sample.txt"}, call_id="read-1")]),
            ModelStep(output=[assistant_message("The file contains HANS_KAGGLE_READ_TEST_123.")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("sdk-tool-test")

    result = run(Runner.run(agent, "Read sample.txt and report the exact contents.", session=session))

    assert "HANS_KAGGLE_READ_TEST_123" in result.final_output
    assert len(model.calls) == 2
    assert "HANS_KAGGLE_READ_TEST_123" in repr(model.calls[1].input)
    session.close()


def test_discovery_search_targeted_replace_and_verification_workflow(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "greeting.py").write_text("MESSAGE = 'old'\n", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("list_directory", {"path": "src"}, call_id="list-1")]),
            ModelStep(output=[function_call("search_files", {"query": "old", "path": "src"}, call_id="search-1")]),
            ModelStep(
                output=[
                    function_call(
                        "replace_in_file",
                        {"path": "src/greeting.py", "old_text": "'old'", "new_text": "'new'"},
                        call_id="replace-1",
                    )
                ]
            ),
            ModelStep(output=[function_call("run_command", {"command": "printf VERIFIED"}, call_id="verify-1")]),
            ModelStep(output=[assistant_message("Targeted replacement verified.")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("sdk-targeted-workflow-test")

    result = run(Runner.run(agent, "Find and fix the greeting, then verify it.", session=session))

    assert (tmp_path / "src" / "greeting.py").read_text(encoding="utf-8") == "MESSAGE = 'new'\n"
    assert result.final_output == "Targeted replacement verified."
    assert len(model.calls) == 5
    assert "greeting.py:1: MESSAGE = 'old'" in repr(model.calls[2].input)
    assert "Replaced text in src/greeting.py" in repr(model.calls[3].input)
    assert "exit_code=0" in repr(model.calls[4].input)
    session.close()


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def make_python_repository(root: Path, *, user_changes: bool) -> dict[str, Path]:
    source = root / "src"
    tests = root / "tests"
    source.mkdir()
    tests.mkdir()
    (root / "pyproject.toml").write_text(
        "[project]\nname = 'sample-service'\nversion = '0.1.0'\n", encoding="utf-8"
    )
    (root / "README.md").write_text("# Sample service\nFeature: next_value\n", encoding="utf-8")
    (root / ".gitignore").write_text(".hans-tmp/\n__pycache__/\n", encoding="utf-8")
    math_ops = source / "math_ops.py"
    math_ops.write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    service = source / "service.py"
    service.write_text(
        "from math_ops import add\n\n\ndef next_value(value):\n    return add(value, -1)\n", encoding="utf-8"
    )
    (source / "formatting.py").write_text("def label(value):\n    return f'[{value}]'\n", encoding="utf-8")
    focused_test = tests / "test_service.py"
    focused_test.write_text(
        "import sys\n\nsys.path.insert(0, 'src')\n\nfrom service import next_value\n\n\ndef test_next_value():\n    assert next_value(2) == 3\n",
        encoding="utf-8",
    )
    docs = root / "docs"
    docs.mkdir()
    (docs / "large-unrelated.md").write_text("unrelated\n" * 30_000, encoding="utf-8")
    for number in range(20):
        (docs / f"note-{number}.md").write_text(f"note {number}\n", encoding="utf-8")
    notes = root / "notes.md"
    notes.write_text("tracked baseline\n", encoding="utf-8")
    git(root, "init")
    git(root, "config", "user.name", "HANS Test")
    git(root, "config", "user.email", "hans-test@example.invalid")
    git(root, "add", ".")
    git(root, "commit", "-m", "baseline")
    paths = {"math_ops": math_ops, "service": service, "focused_test": focused_test, "notes": notes}
    if user_changes:
        notes.write_text("tracked user edit\n", encoding="utf-8")
        scratch = root / "scratch.txt"
        scratch.write_text("untracked user work\n", encoding="utf-8")
        paths["scratch"] = scratch
    return paths


def test_scripted_agent_repairs_a_git_python_repository_without_touching_user_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"/tmp/hans-python312/bin:{os.environ['PATH']}")
    paths = make_python_repository(tmp_path, user_changes=True)
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("list_directory", {"path": "."}, call_id="list-1")]),
            ModelStep(output=[function_call("read_file", {"path": "pyproject.toml"}, call_id="read-metadata")]),
            ModelStep(output=[function_call("search_files", {"query": "next_value", "path": "src"}, call_id="search-1")]),
            ModelStep(output=[function_call("read_file", {"path": "src/math_ops.py"}, call_id="read-math")]),
            ModelStep(output=[function_call("read_file", {"path": "src/service.py"}, call_id="read-service")]),
            ModelStep(output=[function_call("run_command", {"command": "git status --short"}, call_id="status-before")]),
            ModelStep(output=[function_call("run_command", {"command": "git diff"}, call_id="diff-before")]),
            ModelStep(output=[function_call("replace_in_file", {"path": "src/math_ops.py", "old_text": "return left - right", "new_text": "return left + right"}, call_id="replace-math")]),
            ModelStep(output=[function_call("replace_in_file", {"path": "src/service.py", "old_text": "add(value, -1)", "new_text": "add(value, 1)"}, call_id="replace-callsite")]),
            ModelStep(output=[function_call("run_command", {"command": "python -m pytest -q tests/test_service.py"}, call_id="pytest")]),
            ModelStep(output=[function_call("run_command", {"command": "git diff --check"}, call_id="diff-check")]),
            ModelStep(output=[function_call("run_command", {"command": "git status --short"}, call_id="status-after")]),
            ModelStep(output=[assistant_message("The focused repair is verified without touching user work.")]),
        ]
    )
    session = SQLiteSession("sdk-git-python-repair-test")

    result = run(
        Runner.run(make_agent(model, tmp_path), "Repair next_value and verify it.", session=session, max_turns=20)
    )

    assert result.final_output == "The focused repair is verified without touching user work."
    assert paths["math_ops"].read_text(encoding="utf-8") == "def add(left, right):\n    return left + right\n"
    assert paths["service"].read_text(encoding="utf-8").endswith("return add(value, 1)\n")
    assert paths["notes"].read_text(encoding="utf-8") == "tracked user edit\n"
    assert paths["scratch"].read_text(encoding="utf-8") == "untracked user work\n"
    status = git(tmp_path, "status", "--short")
    assert " M notes.md" in status
    assert " M src/math_ops.py" in status
    assert " M src/service.py" in status
    assert "?? scratch.txt" in status
    assert "directory: src" in repr(model.calls[1].input)
    assert "src/service.py:4: def next_value" in repr(model.calls[3].input)
    assert "exit_code=0" in repr(model.calls[6].input) and "notes.md" in repr(model.calls[6].input)
    assert "diff --git a/notes.md" in repr(model.calls[7].input)
    assert "Replaced text in src/math_ops.py" in repr(model.calls[8].input)
    assert "Replaced text in src/service.py" in repr(model.calls[9].input)
    assert "1 passed" in repr(model.calls[10].input)
    assert "exit_code=0" in repr(model.calls[11].input)
    assert "scratch.txt" in repr(model.calls[12].input)
    session.close()


def test_scripted_agent_discovers_an_unfamiliar_repository_without_changes(tmp_path: Path) -> None:
    paths = make_python_repository(tmp_path, user_changes=False)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and ".git" not in path.parts}
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("list_directory", {"path": "."}, call_id="list-1")]),
            ModelStep(output=[function_call("read_file", {"path": "pyproject.toml"}, call_id="metadata")]),
            ModelStep(output=[function_call("search_files", {"query": "next_value", "path": "src"}, call_id="search")]),
            ModelStep(output=[function_call("read_file", {"path": "src/service.py"}, call_id="feature")]),
            ModelStep(output=[assistant_message("The sample-service next_value feature is implemented in src/service.py.")]),
        ]
    )
    session = SQLiteSession("sdk-read-only-discovery-test")

    result = run(Runner.run(make_agent(model, tmp_path), "Find the next_value feature without changing files.", session=session))

    after = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file() and ".git" not in path.parts}
    assert result.final_output.endswith("src/service.py.")
    assert before == after
    assert git(tmp_path, "status", "--short") == ""
    assert paths["service"].read_text(encoding="utf-8").endswith("add(value, -1)\n")
    assert "name = 'sample-service'" in repr(model.calls[2].input)
    assert "src/service.py:4: def next_value" in repr(model.calls[3].input)
    session.close()


def test_write_then_run_command_reaches_next_model_turn(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("write_file", {"path": "main.go", "content": "package main\n"}, call_id="write-1")]),
            ModelStep(output=[function_call("run_command", {"command": "printf HELLO_HANS"}, call_id="run-1")]),
            ModelStep(output=[assistant_message("Verified output HELLO_HANS.")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("sdk-write-run-test")

    result = run(
        Runner.run(agent, "Create a program that prints HELLO_HANS, run it, and verify.", session=session)
    )

    assert (tmp_path / "main.go").read_text(encoding="utf-8") == "package main\n"
    assert result.final_output == "Verified output HELLO_HANS."
    assert len(model.calls) == 3
    assert "HELLO_HANS" in repr(model.calls[2].input)
    assert "exit_code=0" in repr(model.calls[2].input)
    session.close()


def test_failed_verification_reaches_focused_source_and_callsite_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"/tmp/hans-python312/bin:{os.environ['PATH']}")
    paths = make_python_repository(tmp_path, user_changes=False)
    paths["service"].write_text(
        "from math_ops import add\n\n\ndef next_value(value):\n    return add(value, 2)\n", encoding="utf-8"
    )
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("run_command", {"command": "python -m pytest -q tests/test_service.py"}, call_id="failed-test")]),
            ModelStep(output=[function_call("search_files", {"query": "return left - right", "path": "src"}, call_id="search-source")]),
            ModelStep(output=[function_call("read_file", {"path": "src/math_ops.py"}, call_id="read-source")]),
            ModelStep(output=[function_call("read_file", {"path": "src/service.py"}, call_id="read-callsite")]),
            ModelStep(output=[function_call("replace_in_file", {"path": "src/math_ops.py", "old_text": "return left - right", "new_text": "return left + right"}, call_id="repair-source")]),
            ModelStep(output=[function_call("replace_in_file", {"path": "src/service.py", "old_text": "add(value, 2)", "new_text": "add(value, 1)"}, call_id="repair-callsite")]),
            ModelStep(output=[function_call("run_command", {"command": "python -m pytest -q tests/test_service.py"}, call_id="retest")]),
            ModelStep(output=[assistant_message("The failed focused test was repaired and now passes.")]),
        ]
    )
    session = SQLiteSession("sdk-verification-recovery-test")

    result = run(Runner.run(make_agent(model, tmp_path), "Repair the project and verify it.", session=session))

    assert paths["math_ops"].read_text(encoding="utf-8") == "def add(left, right):\n    return left + right\n"
    assert paths["service"].read_text(encoding="utf-8").endswith("add(value, 1)\n")
    assert result.final_output == "The failed focused test was repaired and now passes."
    assert "exit_code=1" in repr(model.calls[1].input)
    assert "AssertionError" in repr(model.calls[1].input)
    assert "src/math_ops.py:2:     return left - right" in repr(model.calls[2].input)
    assert "return add(value, 2)" in repr(model.calls[4].input)
    assert "Replaced text in src/math_ops.py" in repr(model.calls[5].input)
    assert "Replaced text in src/service.py" in repr(model.calls[6].input)
    assert "exit_code=0" in repr(model.calls[7].input)
    assert "1 passed" in repr(model.calls[7].input)
    session.close()


def test_streaming_run_reaches_sdk_completion(tmp_path: Path) -> None:
    model = ScriptedModel([ModelStep(output=[assistant_message("streamed answer")])])
    agent = make_agent(model, tmp_path)

    result, events = run(run_streaming_turn(agent))

    assert result.is_complete
    assert result.final_output == "streamed answer"
    assert model.calls[0].streamed is True
    assert any(event.type == "raw_response_event" for event in events)


async def run_streaming_turn(agent):
    result = Runner.run_streamed(agent, "Say hello.")
    events = await collect_events(result)
    return result, events


async def collect_events(result):
    return [event async for event in result.stream_events()]


def test_sqlite_session_preserves_multi_turn_context(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep(output=[assistant_message("This project uses Go 1.23.")]),
            ModelStep(output=[assistant_message("The project uses Go 1.23.")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("multi-turn-test")

    run(Runner.run(agent, "Remember that this project uses Go 1.23.", session=session))
    second = run(Runner.run(agent, "What Go version does this project use?", session=session))

    assert second.final_output == "The project uses Go 1.23."
    assert "Go 1.23" in repr(model.calls[1].input)
    session.close()


def test_session_can_continue_after_model_error(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep.raise_error(RuntimeError("temporary test failure")),
            ModelStep(output=[assistant_message("recovered")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("error-recovery-test")

    with pytest.raises(RuntimeError, match="temporary test failure"):
        run(Runner.run(agent, "first turn", session=session))
    result = run(Runner.run(agent, "second turn", session=session))

    assert result.final_output == "recovered"
    session.close()
