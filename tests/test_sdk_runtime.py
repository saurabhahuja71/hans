from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from agents import Agent, Runner, SQLiteSession, set_tracing_disabled
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from bolt_next.workspace import make_read_file_tool, make_run_command_tool, make_write_file_tool


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
            make_read_file_tool(workspace),
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
