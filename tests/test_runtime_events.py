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
    TaskChangeSummary,
    TaskDiff,
    TaskUndoRefused,
    TaskUndoSucceeded,
    ToolCompleted,
    ToolOutput,
    ToolStarted,
    UserMessageSubmitted,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)
import bolt_next.runtime as runtime_module
from bolt_next.runtime import HansRuntime
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
            make_list_directory_tool(workspace),
            make_search_files_tool(workspace),
            make_read_file_tool(workspace),
            make_replace_in_file_tool(workspace),
            make_write_file_tool(workspace),
            make_run_command_tool(workspace),
        ],
    )


def test_missing_model_configuration_is_rendered_as_request_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("BOLT_MODEL_BASE_URL", raising=False)
    monkeypatch.delenv("BOLT_MODEL_API_KEY", raising=False)
    runtime = HansRuntime(session=SQLiteSession("runtime-missing-config"), workspace=str(tmp_path))

    events = run(collect(runtime, "Say hello."))

    failure = next(event for event in events if isinstance(event, RequestFailed))
    assert failure.category == "configuration"
    assert failure.message == "Set BOLT_MODEL_BASE_URL and BOLT_MODEL_API_KEY before starting Hans"
    assert events[-1] == ConnectionChanged(False)
    runtime.close()


def test_invalid_local_context_configuration_is_rendered_as_configuration_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BOLT_MODEL_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "invalid")
    runtime = HansRuntime(session=SQLiteSession("runtime-invalid-context"), workspace=str(tmp_path))

    events = run(collect(runtime, "Say hello."))

    failure = next(event for event in events if isinstance(event, RequestFailed))
    assert failure.category == "configuration"
    assert failure.message == "BOLT_MODEL_CONTEXT_TOKENS must be an integer"
    runtime.close()


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
            ModelStep(
                output=[function_call("run_command", {"command": "printf VERIFIED", "purpose": "verify"}, call_id="verify-1")]
            ),
            ModelStep(output=[assistant_message("Verification passed.")]),
        ]
    )
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-verify"))

    events = run(collect(runtime, "Verify the project."))

    tool_output = next(event for event in events if isinstance(event, ToolOutput))
    assert tool_output.call_id == "verify-1"
    assert tool_output.output == original_output
    assert ToolStarted("verify-1", "run_command", "printf VERIFIED", "verify") in events
    assert ToolCompleted("verify-1", "run_command", "printf VERIFIED", True, 0) in events
    passed = next(event for event in events if isinstance(event, VerificationPassed))
    assert passed.evidence.command == "printf VERIFIED"
    assert passed.evidence.success
    completed = next(event for event in events if isinstance(event, RequestCompleted))
    assert completed.evidence == passed.evidence
    assert "VERIFIED" in repr(model.calls[1].input)
    runtime.close()


def test_inspection_command_does_not_replace_verification_evidence(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep(
                output=[function_call("run_command", {"command": "printf VERIFIED", "purpose": "verify"}, call_id="verify-1")]
            ),
            ModelStep(output=[function_call("run_command", {"command": "printf INSPECTED"}, call_id="inspect-1")]),
            ModelStep(output=[assistant_message("Verified and inspected.")]),
        ]
    )
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-inspect"))

    events = run(collect(runtime, "Verify and inspect the project."))

    assert ToolStarted("verify-1", "run_command", "printf VERIFIED", "verify") in events
    assert ToolStarted("inspect-1", "run_command", "printf INSPECTED", "inspect") in events
    assert ToolCompleted("inspect-1", "run_command", "printf INSPECTED", True, 0) in events
    assert not any(
        isinstance(event, (VerificationStarted, VerificationPassed, VerificationFailed)) and event.call_id == "inspect-1"
        for event in events
    )
    completed = next(event for event in events if isinstance(event, RequestCompleted))
    assert completed.evidence is not None
    assert completed.evidence.command == "printf VERIFIED"
    assert completed.evidence.success
    runtime.close()


def test_failed_verification_then_successful_write_invalidates_evidence(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("run_command", {"command": "false", "purpose": "verify"}, call_id="verify-1")]),
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


def test_successful_replacement_invalidates_prior_verification_evidence(tmp_path: Path) -> None:
    (tmp_path / "fixed.txt").write_text("before", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelStep(
                output=[function_call("run_command", {"command": "printf VERIFIED", "purpose": "verify"}, call_id="verify-1")]
            ),
            ModelStep(
                output=[
                    function_call(
                        "replace_in_file",
                        {"path": "fixed.txt", "old_text": "before", "new_text": "after"},
                        call_id="replace-1",
                    )
                ]
            ),
            ModelStep(output=[assistant_message("Changed after verification.")]),
        ]
    )
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession("runtime-replace-invalidate"))

    events = run(collect(runtime, "Verify and then update the file."))

    assert ToolStarted("replace-1", "replace_in_file", "fixed.txt") in events
    assert ToolCompleted("replace-1", "replace_in_file", "fixed.txt", True) in events
    assert (tmp_path / "fixed.txt").read_text(encoding="utf-8") == "after"
    completed = next(event for event in events if isinstance(event, RequestCompleted))
    assert completed.evidence is None
    runtime.close()


def test_runtime_reports_journal_changes_and_exposes_safe_undo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("write_file", {"path": "task.txt", "content": "changed"}, call_id="write-1")]),
            ModelStep(
                output=[function_call("run_command", {"command": "printf VERIFIED", "purpose": "verify"}, call_id="verify-1")]
            ),
            ModelStep(output=[assistant_message("Changed and verified.")]),
        ]
    )

    def fake_create_agent(workspace: Path, *, journal: object) -> Agent:
        return Agent(
            name="journal test agent",
            instructions="Use the provided tools.",
            model=model,
            tools=[make_write_file_tool(workspace, journal), make_run_command_tool(workspace)],
        )

    monkeypatch.setattr(runtime_module, "create_agent", fake_create_agent)
    runtime = HansRuntime(workspace=str(tmp_path), session=SQLiteSession("runtime-journal"))

    events = run(collect(runtime, "Change and verify."))

    summary = next(event for event in events if isinstance(event, TaskChangeSummary))
    assert "changed_files: task.txt" in summary.summary
    diff = runtime.task_diff(max_chars=100)
    assert isinstance(diff, TaskDiff)
    assert "+++ b/task.txt" in diff.diff
    completed = next(event for event in events if isinstance(event, RequestCompleted))
    assert completed.evidence is not None and completed.evidence.success
    assert runtime.undo_task() == TaskUndoSucceeded((), ("task.txt",))
    assert not (tmp_path / "task.txt").exists()
    assert runtime._evidence is None
    assert runtime.undo_task() == TaskUndoRefused(("task.txt",))
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


@pytest.mark.parametrize(
    ("exc", "category"),
    [
        (type("AuthError", (RuntimeError,), {"status_code": 401})("denied"), "authentication"),
        (type("RequestError", (RuntimeError,), {"status": 400})("bad request"), "model"),
        (TimeoutError("too slow"), "connection"),
    ],
)
def test_runtime_classifies_generic_exception_attributes_without_provider_branches(exc, category) -> None:
    from bolt_next.runtime import _failure_details

    failure_category, _message, _debug = _failure_details(exc)

    assert failure_category == category


def test_runtime_redacts_credentials_from_debug_failure_details() -> None:
    from bolt_next.runtime import _failure_details

    category, _message, debug = _failure_details(
        RuntimeError("Authorization: Bearer sk-secret-value-123 api_key=another-secret")
    )

    assert category == "runtime"
    assert "sk-secret-value-123" not in debug
    assert "another-secret" not in debug
    assert "[REDACTED]" in debug


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


def test_cancel_active_interrupts_a_noncooperative_stream_and_allows_next_request() -> None:
    class _BlockedResult:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

        async def stream_events(self):
            await asyncio.Event().wait()
            yield None

    class _CompletedResult:
        def cancel(self) -> None:
            pass

        async def stream_events(self):
            return
            yield None

    class _Runner:
        def __init__(self) -> None:
            self.results = [_BlockedResult(), _CompletedResult()]

        def run_streamed(self, *_args, **_kwargs):
            return self.results.pop(0)

    async def scenario():
        runner = _Runner()
        runtime = HansRuntime(agent=object(), session=object(), runner=runner)
        blocked = asyncio.create_task(collect(runtime, "Wait."))
        await asyncio.sleep(0.01)
        runtime.cancel_active()
        cancelled_events = await asyncio.wait_for(blocked, timeout=1)
        follow_up_events = await asyncio.wait_for(collect(runtime, "Continue."), timeout=1)
        return cancelled_events, follow_up_events

    cancelled_events, follow_up_events = run(scenario())
    assert sum(isinstance(event, RequestCancelled) for event in cancelled_events) == 1
    assert not any(isinstance(event, RequestCompleted) for event in cancelled_events)
    assert any(isinstance(event, RequestCompleted) for event in follow_up_events)


def test_interleaved_tool_outputs_are_correlated_by_call_id_not_order() -> None:
    runtime = HansRuntime(agent=object(), session=object(), runner=object())

    def called(call_id: str, provider_id: str, command: str, purpose: str = "verify"):
        return SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=SimpleNamespace(
                call_id=call_id,
                tool_name="run_command",
                raw_item={
                    "id": provider_id,
                    "call_id": call_id,
                    "arguments": {"command": command, "purpose": purpose},
                },
            ),
        )

    def output(call_id: str, provider_id: str, value: str):
        return SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=SimpleNamespace(call_id=call_id, output=value, raw_item={"id": provider_id, "call_id": call_id}),
        )

    assert ToolStarted("call-a", "run_command", "printf A", "verify") in runtime.translate_stream_event(
        called("call-a", "provider-a", "printf A")
    )
    assert ToolStarted("call-b", "run_command", "printf B", "verify") in runtime.translate_stream_event(
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
    for relative_path in (
        "src/bolt_next/tui.py",
        "src/bolt_next/tui_screen.py",
        "src/bolt_next/textual_tui.py",
    ):
        tree = ast.parse(Path(relative_path).read_text(encoding="utf-8"), filename=relative_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(
                    not alias.name.startswith(("agents", "openai", "anthropic", "litellm"))
                    for alias in node.names
                )
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(("agents", "openai", "anthropic", "litellm"))
            if isinstance(node, ast.Name):
                assert node.id not in {"raw_item", "Runner", "SQLiteSession"}
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"raw_item", "Runner", "SQLiteSession"}
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"exit_code=", "stdout:", "stderr:"}
