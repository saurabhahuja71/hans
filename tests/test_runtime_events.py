from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import Agent, SQLiteSession, set_tracing_disabled
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from bolt_next.events import (
    AssistantMessageComplete,
    AssistantMessageDelta,
    ConnectionChanged,
    RequestCancelled,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
    ToolCompleted,
    ToolOutput,
    ToolStarted,
    UserMessageSubmitted,
    VerificationFailed,
    VerificationPassed,
)
from bolt_next.runtime import HansRuntime
from bolt_next.workspace import make_read_file_tool, make_run_command_tool, make_write_file_tool


@pytest.fixture(autouse=True)
def avoid_broken_executor_shutdown(monkeypatch):
    async def inline_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", inline_to_thread)
    set_tracing_disabled(True)


def run(coro):
    return asyncio.run(coro)


async def collect(runtime: HansRuntime, message: str):
    return [event async for event in runtime.submit(message)]


def make_agent(model: ScriptedModel, workspace: Path) -> Agent:
    return Agent(
        name="HANS runtime test agent",
        instructions="Use the provided tools.",
        model=model,
        tools=[
            make_read_file_tool(workspace),
            make_write_file_tool(workspace),
            make_run_command_tool(workspace),
        ],
    )


def test_scripted_model_submission_emits_semantic_lifecycle_and_connection(tmp_path: Path) -> None:
    model = ScriptedModel([ModelStep(output=[assistant_message("hello from HANS")])])
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-answer"))

    events = run(collect(runtime, "Say hello."))

    assert events[0] == UserMessageSubmitted("Say hello.")
    assert events[1] == RequestStarted("Say hello.")
    assert AssistantMessageDelta("hello from HANS") in events
    assert AssistantMessageComplete("hello from HANS") in events
    assert RequestCompleted(None) in events
    assert events[-1] == ConnectionChanged(True)
    runtime.close()


def test_scripted_tool_events_preserve_authoritative_output_and_verification(tmp_path: Path) -> None:
    original_output = "exit_code=0\nstdout:\nVERIFIEDstderr:\n"
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("run_command", {"command": "printf VERIFIED"}, call_id="verify-1")]),
            ModelStep(output=[assistant_message("Verification passed.")]),
        ]
    )
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-verify"))

    events = run(collect(runtime, "Verify the project."))

    tool_output = next(event for event in events if isinstance(event, ToolOutput))
    assert tool_output.call_id == "verify-1"
    assert tool_output.output == original_output
    assert ToolStarted("verify-1", "run_command", "printf VERIFIED") in events
    assert ToolCompleted("verify-1", "run_command", "printf VERIFIED", True, 0) in events
    passed = next(event for event in events if isinstance(event, VerificationPassed))
    assert passed.evidence.command == "printf VERIFIED"
    assert passed.evidence.success
    completed = next(event for event in events if isinstance(event, RequestCompleted))
    assert completed.evidence == passed.evidence
    assert "VERIFIED" in repr(model.calls[1].input)
    runtime.close()


def test_failed_verification_then_successful_write_invalidates_evidence(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("run_command", {"command": "false"}, call_id="verify-1")]),
            ModelStep(output=[function_call("write_file", {"path": "fixed.txt", "content": "fixed"}, call_id="write-1")]),
            ModelStep(output=[assistant_message("Fixed but not reverified.")]),
        ]
    )
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-invalidate"))

    events = run(collect(runtime, "Attempt a repair."))

    failed = next(event for event in events if isinstance(event, VerificationFailed))
    assert failed.evidence.command == "false"
    assert not failed.evidence.success
    assert ToolCompleted("write-1", "write_file", "fixed.txt", True) in events
    assert (tmp_path / "fixed.txt").read_text(encoding="utf-8") == "fixed"
    completed = next(event for event in events if isinstance(event, RequestCompleted))
    assert completed.evidence is None
    runtime.close()


def test_model_failure_becomes_generic_failure_and_disconnection(tmp_path: Path) -> None:
    model = ScriptedModel([ModelStep.raise_error(RuntimeError("connection refused by test"))])
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-failure"))

    events = run(collect(runtime, "Fail."))

    failure = next(event for event in events if isinstance(event, RequestFailed))
    assert failure.category == "connection"
    assert failure.message == "connection to the configured model endpoint failed"
    assert failure.debug_message == "connection refused by test"
    assert events[-1] == ConnectionChanged(False)
    runtime.close()


def test_cancel_active_settles_as_request_cancelled() -> None:
    class _Result:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        async def stream_events(self):
            while not self.cancelled:
                await asyncio.sleep(0.001)
            return
            yield None

    class _Runner:
        def __init__(self) -> None:
            self.result = _Result()

        def run_streamed(self, *_args, **_kwargs):
            return self.result

    async def scenario():
        runner = _Runner()
        runtime = HansRuntime(agent=object(), session=object(), runner=runner)
        task = asyncio.create_task(collect(runtime, "Wait."))
        await asyncio.sleep(0.01)
        runtime.cancel_active()
        events = await asyncio.wait_for(task, timeout=1)
        return runner, events

    runner, events = run(scenario())
    assert runner.result.cancelled
    assert isinstance(events[-1], RequestCancelled)
    assert not any(isinstance(event, RequestCompleted) for event in events)


def test_interleaved_tool_outputs_are_correlated_by_call_id_not_order() -> None:
    runtime = HansRuntime(agent=object(), session=object(), runner=object())

    def called(call_id: str, provider_id: str, command: str):
        return SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=SimpleNamespace(
                call_id=call_id,
                tool_name="run_command",
                raw_item={"id": provider_id, "call_id": call_id, "arguments": {"command": command}},
            ),
        )

    def output(call_id: str, provider_id: str, value: str):
        return SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=SimpleNamespace(call_id=call_id, output=value, raw_item={"id": provider_id, "call_id": call_id}),
        )

    assert ToolStarted("call-a", "run_command", "printf A") in runtime.translate_stream_event(
        called("call-a", "provider-a", "printf A")
    )
    assert ToolStarted("call-b", "run_command", "printf B") in runtime.translate_stream_event(
        called("call-b", "provider-b", "printf B")
    )
    b_events = tuple(runtime.translate_stream_event(output("call-b", "provider-output-b", "exit_code=0\nstdout:\nB\n")))
    a_events = tuple(runtime.translate_stream_event(output("call-a", "provider-output-a", "exit_code=1\nstdout:\nA\n")))

    assert b_events[0] == ToolOutput("call-b", "exit_code=0\nstdout:\nB\n")
    assert ToolCompleted("call-b", "run_command", "printf B", True, 0) in b_events
    b_verification = next(event for event in b_events if isinstance(event, VerificationPassed))
    assert b_verification.evidence.command == "printf B"
    assert a_events[0] == ToolOutput("call-a", "exit_code=1\nstdout:\nA\n")
    assert ToolCompleted("call-a", "run_command", "printf A", False, 1) in a_events
    a_verification = next(event for event in a_events if isinstance(event, VerificationFailed))
    assert a_verification.evidence.command == "printf A"


def test_tui_modules_do_not_depend_on_sdk_or_raw_wire_names() -> None:
    for relative_path in ("src/bolt_next/tui.py", "src/bolt_next/tui_screen.py"):
        tree = ast.parse(Path(relative_path).read_text(encoding="utf-8"), filename=relative_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(not alias.name.startswith(("agents", "openai")) for alias in node.names)
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("agents", "openai"))
            if isinstance(node, ast.Name):
                assert node.id != "raw_item"
            if isinstance(node, ast.Attribute):
                assert node.attr != "raw_item"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"exit_code=", "stdout:", "stderr:"}
