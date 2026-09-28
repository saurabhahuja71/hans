from __future__ import annotations

import asyncio
import os
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


def test_scripted_agent_repairs_a_multifile_python_repository_without_touching_unrelated_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"/tmp/hans-python312/bin:{os.environ['PATH']}")
    implementation = tmp_path / "calculator.py"
    regression_test = tmp_path / "tests" / "test_calculator.py"
    unrelated = tmp_path / "notes.py"
    implementation.write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    regression_test.parent.mkdir()
    regression_test.write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    unrelated_contents = "STATUS = 'pre-existing local change'\n"
    unrelated.write_text(unrelated_contents, encoding="utf-8")
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("list_directory", {"path": "."}, call_id="list-1")]),
            ModelStep(
                output=[function_call("search_files", {"query": "return left - right"}, call_id="search-1")]
            ),
            ModelStep(
                output=[
                    function_call(
                        "replace_in_file",
                        {
                            "path": "calculator.py",
                            "old_text": "return left - right",
                            "new_text": "return left + right",
                        },
                        call_id="replace-1",
                    )
                ]
            ),
            ModelStep(
                output=[
                    function_call(
                        "run_command",
                        {"command": "python -m pytest -q tests/test_calculator.py"},
                        call_id="verify-1",
                    )
                ]
            ),
            ModelStep(output=[assistant_message("The calculator regression test now passes.")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("sdk-multifile-python-repair-test")

    result = run(Runner.run(agent, "Fix the calculator implementation and verify its test.", session=session))

    assert result.final_output == "The calculator regression test now passes."
    assert implementation.read_text(encoding="utf-8") == "def add(left, right):\n    return left + right\n"
    assert regression_test.read_text(encoding="utf-8") == (
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    )
    assert unrelated.read_text(encoding="utf-8") == unrelated_contents
    assert len(model.calls) == 5
    assert "calculator.py:2:     return left - right" in repr(model.calls[2].input)
    verification_input = repr(model.calls[4].input)
    assert "exit_code=0" in verification_input
    assert "1 passed" in verification_input
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


def test_failed_verification_reaches_a_targeted_repair_and_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PATH", f"/tmp/hans-python312/bin:{os.environ['PATH']}")
    (tmp_path / "calculator.py").write_text("def add(left, right):\n    return left - right\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_calculator.py").write_text(
        "from calculator import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n", encoding="utf-8"
    )
    model = ScriptedModel(
        [
            ModelStep(
                output=[
                    function_call(
                        "run_command",
                        {"command": "python -m pytest -q tests/test_calculator.py"},
                        call_id="verify-1",
                    )
                ]
            ),
            ModelStep(
                output=[
                    function_call(
                        "replace_in_file",
                        {"path": "calculator.py", "old_text": "return left - right", "new_text": "return left + right"},
                        call_id="replace-1",
                    )
                ]
            ),
            ModelStep(
                output=[
                    function_call(
                        "run_command",
                        {"command": "python -m pytest -q tests/test_calculator.py"},
                        call_id="verify-2",
                    )
                ]
            ),
            ModelStep(output=[assistant_message("The repair is verified.")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("sdk-verification-recovery-test")

    result = run(Runner.run(agent, "Repair the project and verify it.", session=session))

    assert (tmp_path / "calculator.py").read_text(encoding="utf-8") == "def add(left, right):\n    return left + right\n"
    assert result.final_output == "The repair is verified."
    assert len(model.calls) == 4
    assert "exit_code=1" in repr(model.calls[1].input)
    assert "Replaced text in calculator.py" in repr(model.calls[2].input)
    final_verification_input = repr(model.calls[3].input)
    assert "exit_code=0" in final_verification_input
    assert "1 passed" in final_verification_input
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
