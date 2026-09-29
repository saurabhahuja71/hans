from __future__ import annotations

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import Agent, ModelSettings, SQLiteSession, set_tracing_disabled
from agents.model_settings import Reasoning
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from bolt_next.events import (
    AssistantMessageComplete,
    AssistantMessageDelta,
    ConnectionChanged,
    ModelStatus,
    PermissionPolicyChanged,
    ReasoningModeChanged,
    ReasoningModeStatus,
    RequestCancelled,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
    RuntimeControlRejected,
    RuntimeControlStatus,
    SessionCleared,
    TaskChangeSummary,
    TaskDiff,
    TaskUndoRefused,
    TaskUndoSucceeded,
    ToolApprovalDisplay,
    ToolApprovalRequested,
    ToolApprovalResolved,
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


async def collect_approval_resolution(
    runtime: HansRuntime, request_id: str, call_id: str, approved: bool
):
    return [event async for event in runtime.resolve_tool_approval(request_id, call_id, approved)]


class _ApprovalState:
    def __init__(self) -> None:
        self.approved: list[object] = []
        self.rejected: list[object] = []

    def approve(self, item: object) -> None:
        self.approved.append(item)

    def reject(self, item: object) -> None:
        self.rejected.append(item)


class _PausedApprovalResult:
    def __init__(self, item: object | tuple[object, ...], state: _ApprovalState) -> None:
        self.interruptions = item if isinstance(item, tuple) else (item,)
        self._state = state
        self.to_state_calls = 0

    def cancel(self) -> None:
        pass

    def to_state(self) -> _ApprovalState:
        self.to_state_calls += 1
        return self._state

    async def stream_events(self):
        return
        yield None


class _FinishedResult:
    def cancel(self) -> None:
        pass

    async def stream_events(self):
        return
        yield None


class _ApprovalRunner:
    def __init__(self, *results: object) -> None:
        self._results = list(results)
        self.calls: list[tuple[object, object]] = []

    def run_streamed(self, agent: object, input_or_state: object, **_kwargs: object) -> object:
        self.calls.append((agent, input_or_state))
        return self._results.pop(0)


@pytest.mark.parametrize("approved", (True, False))
def test_native_tool_approval_decision_resumes_the_saved_sdk_state(approved: bool) -> None:
    item = SimpleNamespace(
        tool_name="write_file",
        call_id="sdk-call-id",
        raw_item=SimpleNamespace(arguments={"path": "task.txt", "content": "sk-secret-content"}),
    )
    state = _ApprovalState()
    paused_result = _PausedApprovalResult(item, state)
    runner = _ApprovalRunner(paused_result, _FinishedResult())
    agent = SimpleNamespace(tools=())
    runtime = HansRuntime(agent=agent, session=object(), runner=runner, interactive=lambda: True)

    paused_events = run(collect(runtime, "Change the file."))

    request = next(event for event in paused_events if isinstance(event, ToolApprovalRequested))
    assert request.request_id == "request-1"
    assert request.call_id == item.call_id
    assert request.tool_name == "write_file"
    assert request.category == "write"
    assert request.display == ToolApprovalDisplay((("path", "task.txt"),))
    assert "content" not in repr(request)
    assert "sk-secret-content" not in repr(request)
    assert paused_result.to_state_calls == 1
    assert state.approved == []
    assert state.rejected == []
    assert not any(isinstance(event, RequestCompleted) for event in paused_events)
    assert run(collect_approval_resolution(runtime, "request-stale", request.call_id, approved)) == [
        RuntimeControlRejected("The approval request is stale or does not match the active request.")
    ]
    assert run(collect_approval_resolution(runtime, request.request_id, "call-stale", approved)) == [
        RuntimeControlRejected("The approval request is stale or does not match the active request.")
    ]
    assert state.approved == []
    assert state.rejected == []

    resolved_events = run(collect_approval_resolution(runtime, request.request_id, request.call_id, approved))

    assert resolved_events[0] == ToolApprovalResolved(request.request_id, request.call_id, approved)
    assert (state.approved, state.rejected) == (([item], []) if approved else ([], [item]))
    assert runner.calls[1] == (agent, state)
    assert any(isinstance(event, RequestCompleted) for event in resolved_events)
    runtime.close()


@pytest.mark.parametrize(
    ("tool_name", "arguments", "expected_fields"),
    (
        ("write_file", {"path": "task.txt", "content": "private"}, (("path", "task.txt"),)),
        ("replace_in_file", {"path": "task.txt", "old": "private"}, (("path", "task.txt"),)),
        ("read_file", {"path": "task.txt", "start_line": 4, "end_line": 8}, (("path", "task.txt"), ("range", "4-8"))),
        ("list_directory", {"path": "src"}, (("path", "src"),)),
        ("search_files", {"path": "src", "query": "token=private"}, (("path", "src"), ("query", "token=[REDACTED]"))),
        ("run_command", {"command": "API_TOKEN=private pytest -q", "purpose": "other"}, (("command", "API_TOKEN=[REDACTED] pytest -q"), ("purpose", "inspect"))),
    ),
)
def test_tool_approval_requests_expose_only_bounded_redacted_semantic_display_data(
    tool_name: str, arguments: dict[str, object], expected_fields: tuple[tuple[str, str], ...]
) -> None:
    item = SimpleNamespace(
        tool_name=tool_name,
        raw_item=SimpleNamespace(call_id=f"call-{tool_name}", arguments=arguments),
    )
    state = _ApprovalState()
    runtime = HansRuntime(
        agent=SimpleNamespace(tools=()),
        session=object(),
        runner=_ApprovalRunner(_PausedApprovalResult(item, state)),
        interactive=lambda: True,
    )

    request = next(event for event in run(collect(runtime, "Proceed.")) if isinstance(event, ToolApprovalRequested))

    assert request.call_id == f"call-{tool_name}"
    assert request.display == ToolApprovalDisplay(expected_fields)
    assert "private" not in repr(request)
    runtime.close()


def test_tool_approval_display_bounds_command_values() -> None:
    command = "x" * 300
    item = SimpleNamespace(
        tool_name="run_command",
        raw_item=SimpleNamespace(call_id="call-long", arguments={"command": command, "purpose": "verify"}),
    )
    runtime = HansRuntime(
        agent=SimpleNamespace(tools=()),
        session=object(),
        runner=_ApprovalRunner(_PausedApprovalResult(item, _ApprovalState())),
        interactive=lambda: True,
    )

    request = next(event for event in run(collect(runtime, "Proceed.")) if isinstance(event, ToolApprovalRequested))

    assert request.display.fields[-1] == ("purpose", "verify")
    assert len(request.display.fields[0][1]) == 240
    assert request.display.fields[0][1].endswith("…")
    runtime.close()


def test_noninteractive_tool_approval_rejects_with_the_native_sdk_signature() -> None:
    item = SimpleNamespace(tool_name="run_command", call_id="sdk-call-id")
    state = _ApprovalState()
    runner = _ApprovalRunner(_PausedApprovalResult(item, state), _FinishedResult())
    agent = SimpleNamespace(tools=())
    runtime = HansRuntime(agent=agent, session=object(), runner=runner, interactive=lambda: False)

    events = run(collect(runtime, "Run the command."))

    assert state.rejected == [item]
    assert runner.calls[1] == (agent, state)
    assert not any(isinstance(event, ToolApprovalRequested) for event in events)
    assert any(isinstance(event, RequestCompleted) for event in events)
    runtime.close()


def test_cancelled_paused_approval_cannot_mutate_or_resume_the_sdk_state() -> None:
    item = SimpleNamespace(tool_name="write_file", call_id="sdk-call-id")
    state = _ApprovalState()
    runner = _ApprovalRunner(_PausedApprovalResult(item, state), _FinishedResult())
    runtime = HansRuntime(agent=SimpleNamespace(tools=()), session=object(), runner=runner, interactive=lambda: True)

    paused_events = run(collect(runtime, "Change the file."))
    request = next(event for event in paused_events if isinstance(event, ToolApprovalRequested))

    assert runtime.cancel_active() == RequestCancelled()
    assert run(collect_approval_resolution(runtime, request.request_id, request.call_id, True)) == [
        RuntimeControlRejected("The approval request is no longer active.")
    ]
    assert state.approved == []
    assert state.rejected == []
    assert len(runner.calls) == 1
    runtime.close()


def test_closed_paused_approval_cannot_mutate_or_resume_the_sdk_state() -> None:
    item = SimpleNamespace(tool_name="write_file", call_id="sdk-call-id")
    state = _ApprovalState()
    runner = _ApprovalRunner(_PausedApprovalResult(item, state), _FinishedResult())
    runtime = HansRuntime(agent=SimpleNamespace(tools=()), session=object(), runner=runner, interactive=lambda: True)

    paused_events = run(collect(runtime, "Change the file."))
    request = next(event for event in paused_events if isinstance(event, ToolApprovalRequested))
    runtime.close()

    assert run(collect_approval_resolution(runtime, request.request_id, request.call_id, False)) == [
        RuntimeControlRejected("The approval request is no longer active.")
    ]
    assert state.approved == []
    assert state.rejected == []
    assert len(runner.calls) == 1


@pytest.mark.parametrize("context", ("model", "session"))
def test_paused_approval_cannot_resume_after_model_or_session_changes(context: str) -> None:
    item = SimpleNamespace(tool_name="write_file", call_id="sdk-call-id")
    state = _ApprovalState()
    runner = _ApprovalRunner(_PausedApprovalResult(item, state), _FinishedResult())
    runtime = HansRuntime(
        agent=SimpleNamespace(tools=()), session=object(), runner=runner, interactive=lambda: True
    )

    paused_events = run(collect(runtime, "Change the file."))
    request = next(event for event in paused_events if isinstance(event, ToolApprovalRequested))
    if context == "model":
        runtime._agent = SimpleNamespace(tools=())
        runtime._model_generation += 1
    else:
        runtime._session = object()
        runtime._session_generation += 1

    assert run(collect_approval_resolution(runtime, request.request_id, request.call_id, True)) == [
        RuntimeControlRejected("The approval request is stale or does not match the active request.")
    ]
    assert state.approved == []
    assert state.rejected == []
    assert len(runner.calls) == 1
    runtime.close()


@pytest.mark.parametrize(
    ("first_approved", "second_approved"),
    ((True, True), (True, False), (False, True), (False, False)),
)
def test_sequential_tool_approvals_use_one_saved_sdk_state_and_resume_once(
    first_approved: bool, second_approved: bool
) -> None:
    first_item = SimpleNamespace(tool_name="write_file", call_id="sdk-call-one")
    second_item = SimpleNamespace(tool_name="run_command", call_id="sdk-call-two")
    state = _ApprovalState()
    paused_result = _PausedApprovalResult((first_item, second_item), state)
    runner = _ApprovalRunner(paused_result, _FinishedResult())
    agent = SimpleNamespace(tools=())
    runtime = HansRuntime(agent=agent, session=object(), runner=runner, interactive=lambda: True)

    paused_events = run(collect(runtime, "Change the file and verify it."))
    first_request = next(event for event in paused_events if isinstance(event, ToolApprovalRequested))
    first_resolution = run(
        collect_approval_resolution(runtime, first_request.request_id, first_request.call_id, first_approved)
    )
    second_request = next(event for event in first_resolution if isinstance(event, ToolApprovalRequested))

    assert second_request.request_id == first_request.request_id
    assert second_request.call_id != first_request.call_id
    assert first_resolution[:2] == [
        ToolApprovalResolved(first_request.request_id, first_request.call_id, first_approved),
        second_request,
    ]
    assert paused_result.to_state_calls == 1
    assert (state.approved, state.rejected) == (([first_item], []) if first_approved else ([], [first_item]))
    assert len(runner.calls) == 1
    assert run(
        collect_approval_resolution(runtime, first_request.request_id, first_request.call_id, first_approved)
    ) == [RuntimeControlRejected("The approval request is stale or does not match the active request.")]
    assert len(runner.calls) == 1

    final_events = run(
        collect_approval_resolution(runtime, second_request.request_id, second_request.call_id, second_approved)
    )

    assert final_events[0] == ToolApprovalResolved(
        second_request.request_id, second_request.call_id, second_approved
    )
    assert state.approved == [item for item, approved in ((first_item, first_approved), (second_item, second_approved)) if approved]
    assert state.rejected == [item for item, approved in ((first_item, first_approved), (second_item, second_approved)) if not approved]
    assert runner.calls[1] == (agent, state)
    assert len(runner.calls) == 2
    assert any(isinstance(event, RequestCompleted) for event in final_events)
    runtime.close()


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

    def fake_create_agent(workspace: Path, *, journal: object, profile: object) -> Agent:
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
    a_completed = next(event for event in a_events if isinstance(event, ToolCompleted))
    assert a_completed == ToolCompleted(
        "call-a", "run_command", "printf A", False, 1, "Command exited with code 1."
    )
    a_verification = next(event for event in a_events if isinstance(event, VerificationFailed))
    assert a_verification.evidence.command == "printf A"


def test_tool_failures_have_safe_semantic_reasons_and_redacted_output() -> None:
    runtime = HansRuntime(agent=object(), session=object(), runner=object())

    def called(call_id: str, name: str, arguments: dict[str, str]):
        return SimpleNamespace(
            type="run_item_stream_event",
            name="tool_called",
            item=SimpleNamespace(
                call_id=call_id,
                tool_name=name,
                raw_item={"call_id": call_id, "arguments": arguments},
            ),
        )

    def output(call_id: str, value: str):
        return SimpleNamespace(
            type="run_item_stream_event",
            name="tool_output",
            item=SimpleNamespace(call_id=call_id, output=value),
        )

    def failure(call_id: str, name: str, arguments: dict[str, str], value: str) -> ToolCompleted:
        runtime.translate_stream_event(called(call_id, name, arguments))
        events = tuple(runtime.translate_stream_event(output(call_id, value)))
        completed = next(event for event in events if isinstance(event, ToolCompleted))
        assert completed.success is False
        return completed

    missing = failure("read-missing", "read_file", {"path": "missing.py"}, "Error: file does not exist: missing.py")
    assert missing.failure_reason == "File does not exist."

    read_call_id = "read-1"
    runtime.translate_stream_event(called(read_call_id, "read_file", {"path": "secret.txt"}))
    read_events = tuple(
        runtime.translate_stream_event(
            output(
                read_call_id,
                "Error reading 'secret.txt': Authorization: Bearer sk-secret-value-123 "
                "password=hidden https://user:pass@example.test",
            )
        )
    )
    read_output = next(event for event in read_events if isinstance(event, ToolOutput))
    read_completed = next(event for event in read_events if isinstance(event, ToolCompleted))
    assert read_completed.success is False
    assert read_completed.failure_reason is not None
    for secret in ("sk-secret-value-123", "hidden", "user:pass"):
        assert secret not in read_output.output
        assert secret not in read_completed.failure_reason
    assert "[REDACTED]" in read_output.output
    assert "[REDACTED]" in read_completed.failure_reason

    assert failure(
        "write", "write_file", {"path": "../outside.py"}, "Error writing '../outside.py': Path is outside the workspace"
    ).failure_reason == "Path is outside the workspace."
    assert failure(
        "replace", "replace_in_file", {"path": "main.py"},
        "Error replacing 'main.py': old_text must occur exactly once (found 2)",
    ).failure_reason == "Old_text must occur exactly once (found 2)."
    assert failure(
        "list", "list_directory", {"path": "file.txt"},
        "Error listing 'file.txt': path is a file; use read_file instead",
    ).failure_reason == "Path is a file; use read_file instead."
    assert failure(
        "search", "search_files", {"query": ""}, "Error searching: query must be a non-empty string"
    ).failure_reason == "Query must be a non-empty string."

    runtime.translate_stream_event(called("run-1", "run_command", {"command": "false", "purpose": "verify"}))
    command_events = tuple(runtime.translate_stream_event(output("run-1", "exit_code=1\nstderr:\nfailed\n")))
    command_completed = next(event for event in command_events if isinstance(event, ToolCompleted))
    assert command_completed.exit_code == 1
    assert command_completed.failure_reason == "Command exited with code 1."
    assert any(isinstance(event, VerificationFailed) for event in command_events)

    assert failure(
        "run-timeout", "run_command", {"command": "sleep 1"}, "Error: command timed out after 120 seconds"
    ).failure_reason == "Command timed out after 120 seconds."
    assert failure(
        "run-rejected", "run_command", {"command": "sh -c true"},
        "Error: run_command does not run a shell. Pass the program and its arguments directly.",
    ).failure_reason == "Run_command does not run a shell. Pass the program and its arguments directly."

    traceback_events = tuple(
        runtime.translate_stream_event(
            output(
                "read-traceback",
                "Error reading broken.txt\nTraceback (most recent call last):\nprivate details",
            )
        )
    )
    assert ToolOutput("read-traceback", "Tool diagnostic omitted.") in traceback_events

    unknown = failure("read-unknown", "read_file", {"path": "broken.txt"}, "Error reading broken.txt")
    assert unknown.failure_reason == "The tool operation failed without a detailed diagnostic."


@pytest.mark.parametrize(
    ("category", "tool_name", "arguments", "assert_denied", "assert_allowed"),
    (
        ("read", "list_directory", {"path": "."}, lambda root: None, lambda root: None),
        ("read", "search_files", {"query": "guarded", "path": "."}, lambda root: None, lambda root: None),
        ("read", "read_file", {"path": "guarded.txt"}, lambda root: None, lambda root: None),
        (
            "write",
            "write_file",
            {"path": "created.txt", "content": "created"},
            lambda root: assert_path_text(root / "created.txt", None),
            lambda root: assert_path_text(root / "created.txt", "created"),
        ),
        (
            "write",
            "replace_in_file",
            {"path": "guarded.txt", "old_text": "guarded", "new_text": "changed"},
            lambda root: assert_path_text(root / "guarded.txt", "guarded"),
            lambda root: assert_path_text(root / "guarded.txt", "changed"),
        ),
        (
            "execute",
            "run_command",
            {"command": "touch executed.txt"},
            lambda root: assert_path_text(root / "executed.txt", None),
            lambda root: assert_path_text(root / "executed.txt", ""),
        ),
    ),
)
def test_permission_policy_blocks_every_workspace_tool_and_allow_restores_execution(
    tmp_path: Path, category: str, tool_name: str, arguments: dict[str, str], assert_denied, assert_allowed
) -> None:
    (tmp_path / "guarded.txt").write_text("guarded", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelStep(output=[function_call(tool_name, arguments, call_id="denied")]),
            ModelStep(output=[assistant_message("denied result received")]),
            ModelStep(output=[function_call(tool_name, arguments, call_id="allowed")]),
            ModelStep(output=[assistant_message("allowed result received")]),
        ]
    )
    runtime = HansRuntime(agent=make_agent(model, tmp_path), session=SQLiteSession(f"permission-{tool_name}"))

    assert runtime.set_permission(category, "deny") == PermissionPolicyChanged(category, "deny")
    denied_events = run(collect(runtime, "Try the tool."))
    assert ToolOutput("denied", f"Permission denied: {category} operations are disabled.") in denied_events
    assert any(
        isinstance(event, ToolCompleted) and event.call_id == "denied" and not event.success for event in denied_events
    )
    assert_denied(tmp_path)

    assert runtime.set_permission(category, "allow") == PermissionPolicyChanged(category, "allow")
    allowed_events = run(collect(runtime, "Try the tool again."))
    assert any(
        isinstance(event, ToolCompleted) and event.call_id == "allowed" and event.success for event in allowed_events
    )
    assert_allowed(tmp_path)
    runtime.close()


def assert_path_text(path: Path, expected: str | None) -> None:
    if expected is None:
        assert not path.exists()
    else:
        assert path.read_text(encoding="utf-8") == expected


def test_clear_session_history_removes_sdk_history_and_preserves_runtime_controls_and_journal(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("write_file", {"path": "task.txt", "content": "BLUE"}, call_id="write")]),
            ModelStep(output=[assistant_message("The remembered value is BLUE.")]),
            ModelStep(output=[assistant_message("There is no prior conversation context.")]),
        ]
    )
    session = SQLiteSession("runtime-clear-history")

    def fake_create_agent(workspace: Path, *, journal: object, profile: object) -> Agent:
        return Agent(
            name="clear history test agent",
            instructions="Use the provided tools.",
            model=model,
            tools=[make_write_file_tool(workspace, journal)],
        )

    original_create_agent = runtime_module.create_agent
    runtime_module.create_agent = fake_create_agent
    try:
        runtime = HansRuntime(workspace=str(tmp_path), session=session)
        run(collect(runtime, "Remember that the value is BLUE."))
    finally:
        runtime_module.create_agent = original_create_agent
    assert session_items(session)
    assert "task.txt" in runtime.task_diff().diff
    assert runtime.set_permission("write", "deny") == PermissionPolicyChanged("write", "deny")

    assert run(runtime.clear_session_history()) == SessionCleared()
    assert session_items(session) == []
    assert runtime.get_control_status() == RuntimeControlStatus("allow", "deny", "allow")
    assert "task.txt" in runtime.task_diff().diff
    follow_up_events = run(collect(runtime, "What was the remembered value?"))

    assert any(isinstance(event, RequestCompleted) for event in follow_up_events)
    assert "BLUE" not in repr(model.calls[-1].input)
    assert "Remember that the value" not in repr(model.calls[-1].input)
    assert (tmp_path / "task.txt").read_text(encoding="utf-8") == "BLUE"
    session.close()


def session_items(session: SQLiteSession):
    return run(session.get_items())


def test_controls_reject_while_active_and_work_after_cancellation() -> None:
    class _Result:
        def cancel(self) -> None:
            pass

        async def stream_events(self):
            await asyncio.Event().wait()
            yield None

    class _Runner:
        def run_streamed(self, *_args, **_kwargs):
            return _Result()

    async def scenario():
        session = SQLiteSession("runtime-control-cancellation")
        runtime = HansRuntime(agent=object(), session=session, runner=_Runner())
        task = asyncio.create_task(collect(runtime, "Wait."))
        await asyncio.sleep(0.01)
        busy_permission = runtime.set_permission("write", "deny")
        busy_clear = await runtime.clear_session_history()
        runtime.cancel_active()
        events = await asyncio.wait_for(task, timeout=1)
        idle_permission = runtime.set_permission("write", "deny")
        idle_clear = await runtime.clear_session_history()
        session.close()
        return busy_permission, busy_clear, events, idle_permission, idle_clear

    busy_permission, busy_clear, events, idle_permission, idle_clear = run(scenario())
    assert busy_permission == RuntimeControlRejected("Permission changes are available when HANS is idle.")
    assert busy_clear == RuntimeControlRejected("Cannot clear the session while HANS is busy.")
    assert any(isinstance(event, RequestCancelled) for event in events)
    assert idle_permission == PermissionPolicyChanged("write", "deny")
    assert idle_clear == SessionCleared()


def test_runtime_model_and_reasoning_status_are_sdk_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL", "runtime-model")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "32768")
    monkeypatch.setenv("BOLT_MODEL_REASONING_MODES", "low, high, none")
    monkeypatch.setenv("BOLT_MODEL_REASONING_NONE_SEMANTICS", "literal")
    agent = SimpleNamespace(model_settings=ModelSettings(reasoning=Reasoning(effort="low")), tools=())
    runtime = HansRuntime(agent=agent, session=object(), runner=object())

    model_status = runtime.get_model_status()
    mode_status = runtime.get_reasoning_mode_status()

    assert isinstance(model_status, ModelStatus)
    assert model_status.model.id == "runtime-model"
    assert model_status.model.context_tokens == 32768
    assert model_status.current_reasoning_mode == "low"
    assert isinstance(mode_status, ReasoningModeStatus)
    assert mode_status == ReasoningModeStatus("low", ("low", "high", "none"), "literal")
    assert not hasattr(model_status, "model_settings")
    assert not hasattr(mode_status, "model_settings")


@pytest.mark.parametrize(
    ("none_semantics", "expected_reasoning"),
    (("literal", "none"), ("omit", None)),
)
def test_reasoning_mode_override_uses_base_settings_without_rebuilding_runtime_objects(
    monkeypatch: pytest.MonkeyPatch, none_semantics: str, expected_reasoning: str | None
) -> None:
    monkeypatch.setenv("BOLT_MODEL", "runtime-model")
    monkeypatch.setenv("BOLT_MODEL_REASONING_MODES", "low,none,high")
    monkeypatch.setenv("BOLT_MODEL_REASONING_NONE_SEMANTICS", none_semantics)

    class _Result:
        async def stream_events(self):
            return
            yield None

    class _Runner:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def run_streamed(self, agent, _message, *, session, run_config):
            self.calls.append((agent, session))
            return _Result()

    base_settings = ModelSettings(reasoning=Reasoning(effort="low"))
    agent = SimpleNamespace(model_settings=base_settings, tools=())
    session = object()
    runner = _Runner()
    runtime = HansRuntime(agent=agent, session=session, runner=runner)

    run(collect(runtime, "Use the configured default."))
    assert agent.model_settings is base_settings
    assert runtime.set_reasoning_mode(" NONE ") == ReasoningModeChanged("none")
    assert runner.calls == [(agent, session)]
    assert agent.model_settings is not base_settings
    assert (agent.model_settings.reasoning.effort if agent.model_settings.reasoning else None) == expected_reasoning

    run(collect(runtime, "Use the none override."))
    assert runner.calls == [(agent, session), (agent, session)]
    assert runtime.set_reasoning_mode("high") == ReasoningModeChanged("high")
    assert agent.model_settings.reasoning is not None
    assert agent.model_settings.reasoning.effort == "high"

    run(collect(runtime, "Use the high override."))
    assert runner.calls == [(agent, session), (agent, session), (agent, session)]
    assert runtime.set_reasoning_mode(None) == ReasoningModeChanged(None)
    assert agent.model_settings is base_settings


def test_reasoning_mode_rejects_undeclared_values_and_active_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL", "qwen3.6-27b")
    monkeypatch.delenv("BOLT_MODEL_REASONING_MODES", raising=False)
    undeclared_runtime = HansRuntime(agent=object(), session=object(), runner=object())
    assert undeclared_runtime.set_reasoning_mode("low") == RuntimeControlRejected(
        "Reasoning mode support is not declared for the active configured model."
    )

    monkeypatch.setenv("BOLT_MODEL_REASONING_MODES", "low,high")
    runtime = HansRuntime(agent=object(), session=object(), runner=object())
    assert runtime.set_reasoning_mode("medium") == RuntimeControlRejected(
        "Unsupported reasoning mode: medium. Supported modes: low, high."
    )

    class _Result:
        def cancel(self) -> None:
            pass

        async def stream_events(self):
            await asyncio.Event().wait()
            yield None

    class _Runner:
        def run_streamed(self, *_args, **_kwargs):
            return _Result()

    async def scenario():
        active_runtime = HansRuntime(agent=object(), session=object(), runner=_Runner())
        task = asyncio.create_task(collect(active_runtime, "Wait."))
        await asyncio.sleep(0.01)
        rejected = active_runtime.set_reasoning_mode("high")
        active_runtime.cancel_active()
        await asyncio.wait_for(task, timeout=1)
        return rejected

    assert run(scenario()) == RuntimeControlRejected("Reasoning mode can be changed when HANS is idle.")


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
