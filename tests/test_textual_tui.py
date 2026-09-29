from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea

from bolt_next.events import (
    AssistantMessageComplete,
    AssistantMessageDelta,
    ConnectionChanged,
    ModelChanged,
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
    VerificationEvidence,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)
from bolt_next.commands import format_help
from bolt_next.model_catalog import ModelInfo
from bolt_next.textual_tui import DetailScreen, HansTextualApp, TaskDiffScreen, ThemeScreen


class FakeRuntime:
    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events or []
        self.prompts: list[str] = []
        self.cancelled = 0
        self.closed = 0
        self.task_diff_calls: list[int | None] = []
        self.undo_calls = 0
        self.permissions = {"read": "allow", "write": "allow", "execute": "allow"}
        self.control_calls: list[tuple[object, ...]] = []
        self.models = (
            ModelInfo(
                id="configured-model",
                display_name="Configured Model",
                endpoint_profile="configured endpoint",
                context_tokens=128_000,
                supported_reasoning_modes=("none", "high"),
                none_semantics="omit",
            ),
            ModelInfo(
                id="large",
                display_name="Large Model",
                endpoint_profile="configured endpoint",
                context_tokens=256_000,
                supported_reasoning_modes=("none", "low"),
                none_semantics="literal",
            ),
        )
        self.model = self.models[0]
        self.reasoning_mode: str | None = None
        self.diff_event = TaskDiff("--- a/task.txt\n+++ b/task.txt\n+updated")
        self.undo_event: TaskUndoSucceeded | TaskUndoRefused = TaskUndoSucceeded((), ())
        self.approval_calls: list[tuple[str, str, bool]] = []
        self.approval_events: list[object] = []
        self.approval_event_batches: list[list[object]] = []
        self.approval_started = asyncio.Event()
        self.approval_release = asyncio.Event()

    async def submit(self, message: str) -> AsyncIterator[object]:
        self.prompts.append(message)
        for event in self.events:
            yield event
            await asyncio.sleep(0)

    def cancel_active(self) -> None:
        self.cancelled += 1

    async def resolve_tool_approval(
        self, request_id: str, call_id: str, approved: bool
    ) -> AsyncIterator[object]:
        self.approval_calls.append((request_id, call_id, approved))
        self.approval_started.set()
        await self.approval_release.wait()
        events = self.approval_event_batches.pop(0) if self.approval_event_batches else self.approval_events
        for event in events:
            yield event

    def task_diff(self, *, max_chars: int | None = None) -> TaskDiff:
        self.task_diff_calls.append(max_chars)
        return self.diff_event

    def undo_task(self) -> TaskUndoSucceeded | TaskUndoRefused:
        self.undo_calls += 1
        return self.undo_event

    def get_control_status(self) -> RuntimeControlStatus:
        return RuntimeControlStatus(
            read_policy=self.permissions["read"],
            write_policy=self.permissions["write"],
            execute_policy=self.permissions["execute"],
        )

    def set_permission(self, category: str, policy: str) -> PermissionPolicyChanged:
        self.control_calls.append(("permission", category, policy))
        self.permissions[category] = policy
        return PermissionPolicyChanged(category, policy)

    async def clear_session_history(self) -> SessionCleared:
        self.control_calls.append(("clear", None, None))
        return SessionCleared()

    def get_model_status(self) -> ModelStatus:
        self.control_calls.append(("model_status",))
        return ModelStatus(self.model, self.reasoning_mode, self.models)

    def select_model(self, model_id: str) -> ModelChanged | RuntimeControlRejected | None:
        self.control_calls.append(("select_model", model_id))
        selected = next((model for model in self.models if model.id == model_id), None)
        if selected is None:
            return RuntimeControlRejected("Model selection is unavailable.")
        if selected.id == self.model.id:
            return None
        previous_model_id = self.model.id
        self.model = selected
        self.reasoning_mode = None
        return ModelChanged(previous_model_id, selected, new_session_started=True, reasoning_reset=True)

    def get_reasoning_mode_status(self) -> ReasoningModeStatus:
        self.control_calls.append(("reasoning_mode_status",))
        return ReasoningModeStatus(
            self.reasoning_mode, self.model.supported_reasoning_modes, self.model.none_semantics
        )

    def set_reasoning_mode(self, mode: str) -> ReasoningModeChanged | RuntimeControlRejected:
        self.control_calls.append(("reasoning_mode", mode))
        if mode not in self.model.supported_reasoning_modes:
            return RuntimeControlRejected(f"Unsupported reasoning mode: {mode}")
        self.reasoning_mode = mode
        return ReasoningModeChanged(mode)

    def close(self) -> None:
        self.closed += 1


def rendered(widget: Static) -> str:
    return str(widget.render())


def transcript_text(app: HansTextualApp) -> str:
    transcript = app.query_one("#transcript", VerticalScroll)
    return "\n".join(rendered(child) for child in transcript.children if isinstance(child, Static))


def test_textual_chrome_is_compact_data_driven_and_preserves_composer_keys(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            header = rendered(app.query_one("#hans-header", Static))
            assert "HANS" in header
            assert "model test-model" in header
            assert "DISCONNECTED" in header
            assert "workspace" in header
            assert rendered(app.query_one("#state-line", Static)) == "IDLE"
            assert (
                rendered(app.query_one("#footer", Static))
                == "Enter send · Shift+Enter newline · Ctrl-D send / empty exit · Ctrl-Q quit"
            )
            assert not app.query("#tool-area")
            assert not app.query("#controls")
            composer = app.query_one("#composer", TextArea)

            await pilot.press(*"line one", "shift+enter", *"line two")
            await pilot.pause()
            assert composer.text == "line one\nline two"
            await pilot.press("enter")
            await pilot.pause()
            assert runtime.prompts == ["line one\nline two"]
            assert composer.text == ""

            await pilot.press("o", "k", "ctrl+d")
            await pilot.pause()
            assert runtime.prompts == ["line one\nline two", "ok"]
            assert composer.text == ""

        assert runtime.closed == 1

    asyncio.run(scenario())


def test_empty_exit_and_ctrl_q_do_not_submit(tmp_path: Path) -> None:
    async def scenario(keys: tuple[str, ...]) -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press(*keys)
            await pilot.pause()
            assert runtime.prompts == []
        assert runtime.closed == 1

    asyncio.run(scenario(("ctrl+d",)))
    asyncio.run(scenario(tuple("exit") + ("enter",)))
    asyncio.run(scenario(tuple("quit") + ("enter",)))
    asyncio.run(scenario(tuple("/exit") + ("enter",)))
    asyncio.run(scenario(tuple("/quit") + ("enter",)))
    asyncio.run(scenario(("ctrl+q",)))


def test_textual_theme_cycle_and_todo_panel_are_local_and_persist_through_clear(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            app._todos.add("review compact TODO panel")
            await pilot.press("ctrl+t")
            await pilot.pause()
            todo_view = app.query_one("#todo-view", Static)
            assert app._todos_visible is True
            assert todo_view.styles.display == "block"
            assert rendered(todo_view) == "TODO (1)\n  #1 review compact TODO panel"

            await pilot.press("ctrl+b")
            await pilot.pause()
            assert app._theme_name == "light"
            assert app.theme == "hans-light"
            await pilot.press("ctrl+b", "ctrl+b", "ctrl+b")
            await pilot.pause()
            assert app._theme_name == "dark"

            await pilot.press(*"/clear", "enter")
            await pilot.pause()
            assert app._todos_visible is True
            assert rendered(todo_view) == "TODO (1)\n  #1 review compact TODO panel"
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert app._todos_visible is False
            assert todo_view.styles.display == "none"
            assert runtime.prompts == []
            assert runtime.control_calls == [("clear", None, None)]

    asyncio.run(scenario())


def test_textual_theme_and_todo_shortcuts_do_not_bypass_active_or_approval_input(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestStarted("work"))
            await pilot.press("ctrl+b", "ctrl+t")
            await pilot.pause()
            assert app._theme_name == "dark"
            assert app._todos_visible is False

            await app._render_event(RequestCompleted(None))
            await app._render_event(approval_event("request-1", "call-1", "write_file", "write"))
            await pilot.press("ctrl+b", "ctrl+t")
            await pilot.pause()
            assert app._approval_pending == ("request-1", "call-1")
            assert app._theme_name == "dark"
            assert app._todos_visible is False

            app._approval_pending = None
            app._approval_resolving = True
            await pilot.press("ctrl+b", "ctrl+t")
            await pilot.pause()
            assert app._theme_name == "dark"
            assert app._todos_visible is False
            assert runtime.prompts == []

    asyncio.run(scenario())


def approval_event(
    request_id: str, call_id: str, tool_name: str, category: str, fields: tuple[tuple[str, str], ...] = ()
) -> ToolApprovalRequested:
    return ToolApprovalRequested(request_id, call_id, tool_name, category, ToolApprovalDisplay(fields))


@pytest.mark.parametrize(("key", "approved"), (("y", True), ("n", False)))
def test_textual_resolves_tool_approval_without_leaving_a_stale_prompt(
    tmp_path: Path, key: str, approved: bool
) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        runtime.approval_events = [ToolApprovalResolved("request-1", "call-1", approved), RequestCompleted(None)]
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(
                approval_event("request-1", "call-1", "write_file", "write", (("path", "task.txt"),))
            )
            assert app._approval_pending == ("request-1", "call-1")
            assert rendered(app.query_one("#footer", Static)).startswith("? APPROVAL REQUIRED")
            assert "path: task.txt" in transcript_text(app)
            assert "private content" not in transcript_text(app)

            await pilot.press(key)
            await asyncio.wait_for(runtime.approval_started.wait(), timeout=1)

            assert runtime.approval_calls == [("request-1", "call-1", approved)]
            assert app._approval_pending is None
            assert app._approval_resolving is True
            assert rendered(app.query_one("#state-line", Static)).startswith("RESUMING")
            assert rendered(app.query_one("#footer", Static)).startswith("◉ RESUMING")
            await pilot.press(*"blocked", "ctrl+d")
            assert app.query_one("#composer", TextArea).text == ""
            assert runtime.prompts == []

            runtime.approval_release.set()
            await pilot.pause()
            await pilot.pause()
            assert app._approval_resolving is False
            assert app._request_active is False
            assert rendered(app.query_one("#state-line", Static)) == "COMPLETE"

    asyncio.run(scenario())


def test_textual_replaces_a_resolved_approval_with_the_next_semantic_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        runtime.approval_event_batches = [
            [
                ToolApprovalResolved("request-1", "call-1", True),
                approval_event("request-1", "call-2", "run_command", "execute", (("command", "pytest -q"),)),
            ],
            [ToolApprovalResolved("request-1", "call-2", False), RequestCompleted(None)],
        ]
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(approval_event("request-1", "call-1", "write_file", "write"))
            await pilot.press("y")
            await asyncio.wait_for(runtime.approval_started.wait(), timeout=1)
            runtime.approval_release.set()
            await pilot.pause()
            await pilot.pause()

            assert runtime.approval_calls == [("request-1", "call-1", True)]
            assert app._approval_pending == ("request-1", "call-2")
            assert app._approval_resolving is False
            assert app._request_active is True
            assert rendered(app.query_one("#footer", Static)).startswith("? APPROVAL REQUIRED")
            await pilot.press(*"blocked", "ctrl+d")
            assert app.query_one("#composer", TextArea).text == ""
            assert runtime.prompts == []

            await pilot.press("n")
            await pilot.pause()
            await pilot.pause()
            assert runtime.approval_calls == [
                ("request-1", "call-1", True),
                ("request-1", "call-2", False),
            ]
            assert app._approval_pending is None
            assert app._approval_resolving is False
            assert app._request_active is False
            assert rendered(app.query_one("#state-line", Static)) == "COMPLETE"

    asyncio.run(scenario())


def test_textual_renders_inline_stable_tools_without_normal_raw_output(tmp_path: Path) -> None:
    async def scenario() -> None:
        evidence = VerificationEvidence("pytest -q", 0, True, True, True, "2026-09-28T00:00:00+00:00")
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            for event in (
                UserMessageSubmitted("inspect this"),
                RequestStarted("inspect this"),
                AssistantMessageDelta("Hello"),
                AssistantMessageDelta(" there"),
                AssistantMessageComplete("Hello there"),
                ToolStarted("call-a", "read_file", "a.py"),
                ToolOutput("call-a", "private source bytes"),
                ToolCompleted("call-a", "read_file", "a.py", True),
                VerificationStarted("verify-1", "pytest -q"),
                VerificationPassed("verify-1", evidence),
                TaskChangeSummary("changed_files: a.py"),
                RequestCompleted(evidence),
                ConnectionChanged(True),
            ):
                await app._render_event(event)
            await pilot.pause()

            text = transcript_text(app)
            assert "YOU\n> inspect this" in text
            assert text.count("HANS\nHello there") == 1
            assert "✓ read_file a.py" in text
            assert "1 line" in text
            assert "call-a" not in text
            assert "private source bytes" not in text
            assert "VERIFICATION\n✓ pytest -q passed" in text
            assert "CHANGES\nM a.py  HANS" in text
            assert "FINAL RESULT\nCOMPLETE\nVerified: pytest -q" in text
            assert rendered(app.query_one("#state-line", Static)) == "COMPLETE"
            assert "CONNECTED" in rendered(app.query_one("#hans-header", Static))

    asyncio.run(scenario())


def test_textual_debug_output_is_gated_and_bounded(tmp_path: Path, monkeypatch) -> None:
    async def scenario(debug: bool) -> str:
        if debug:
            monkeypatch.setenv("HANS_DEBUG", "1")
        else:
            monkeypatch.delenv("HANS_DEBUG", raising=False)
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        app.MAX_TOOL_OUTPUT_CHARS = 8
        async with app.run_test() as pilot:
            await app._render_event(ToolStarted("call", "run_command", "pytest -q", "verify"))
            await app._render_event(ToolOutput("call", "abcdefghijk"))
            await pilot.pause()
            return transcript_text(app)

    normal = asyncio.run(scenario(False))
    assert "abcdefgh" not in normal
    debug = asyncio.run(scenario(True))
    assert "DEBUG TOOL [call] OUTPUT" in debug
    assert "abcdefgh" in debug
    assert "display truncated (3 characters omitted)" in debug
    assert "ijk" not in debug


def test_completed_footer_uses_task_change_summary(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test() as pilot:
            for event in (
                RequestStarted("change a file"),
                TaskChangeSummary("changed_files: a.py"),
                RequestCompleted(None),
            ):
                await app._render_event(event)
            assert (
                rendered(app.query_one("#footer", Static))
                == "✓ COMPLETE · Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"
            )

            await app._render_event(RequestStarted("inspect only"))
            await app._render_event(RequestCompleted(None))
            assert rendered(app.query_one("#footer", Static)) == "✓ COMPLETE · Enter new task · Ctrl-Q quit"
            await pilot.pause()

    asyncio.run(scenario())


def test_lifecycle_states_are_derived_from_semantic_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        failed = VerificationEvidence("pytest -q", 1, True, True, False, "2026-09-28T00:00:00+00:00")
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestStarted("inspect"))
            assert rendered(app.query_one("#state-line", Static)) == "INVESTIGATING"
            assert (
                rendered(app.query_one("#footer", Static))
                == "◉ INVESTIGATING · Ctrl-C cancel · Ctrl-Q quit"
            )
            await app._render_event(ToolStarted("write", "write_file", "a.py"))
            assert rendered(app.query_one("#state-line", Static)) == "EDITING"
            await app._render_event(VerificationStarted("verify", "pytest -q"))
            assert rendered(app.query_one("#state-line", Static)).startswith("VERIFYING")
            assert (
                rendered(app.query_one("#footer", Static))
                == "◉ VERIFYING · Ctrl-C cancel · Ctrl-Q quit"
            )
            assert "pytest" not in rendered(app.query_one("#footer", Static))
            await app._render_event(VerificationFailed("verify", failed))
            assert rendered(app.query_one("#state-line", Static)).startswith("CORRECTING")
            await app._render_event(RequestFailed("connection", "unavailable"))
            assert rendered(app.query_one("#state-line", Static)).startswith("FAILED")
            assert (
                rendered(app.query_one("#footer", Static))
                == "✗ FAILED · Enter retry/new task · Ctrl-Q quit"
            )
            await app._render_event(RequestCancelled())
            assert rendered(app.query_one("#state-line", Static)) == "CANCELLED"
            assert (
                rendered(app.query_one("#footer", Static))
                == "⏸ CANCELLED · Enter new task · Ctrl-Q quit"
            )
            assert "VERIFICATION\n✗ pytest -q failed" in transcript_text(app)
            assert "ERROR\n✗ connection failed\nunavailable" in transcript_text(app)
            await pilot.pause()

    asyncio.run(scenario())


def test_ctrl_g_uses_a_bounded_modal_and_escape_returns(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            base_screen = app.screen
            await pilot.press("ctrl+g")
            await pilot.pause()
            assert runtime.task_diff_calls == [app.MAX_TASK_DIFF_CHARS]
            assert isinstance(app.screen, TaskDiffScreen)
            diff_view = app.screen.query_one("#task-diff")
            assert diff_view.region.x == 0
            assert diff_view.region.y == 0
            assert diff_view.region.width == app.size.width
            assert diff_view.region.height == app.size.height
            assert rendered(app.screen.query_one("#task-diff-title", Static)) == "DIFF  ·  Ctrl-Y copy  ·  Esc back"
            assert "+++ b/task.txt" in rendered(app.screen.query_one(".diff-content", Static))
            await pilot.press("escape")
            await pilot.pause()
            assert app.screen is base_screen

    asyncio.run(scenario())


def test_ctrl_z_reports_safe_task_undo_and_does_not_run_while_active(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        runtime.undo_event = TaskUndoSucceeded(("existing.py",), ("new.py",))
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+z")
            await pilot.pause()
            assert runtime.undo_calls == 1
            assert "UNDO\n✓ Undo completed · restored: existing.py; removed: new.py" in transcript_text(app)
            assert rendered(app.query_one("#state-line", Static)).startswith("IDLE")
            assert (
                rendered(app.query_one("#footer", Static))
                == "Enter send · Shift+Enter newline · Ctrl-D send / empty exit · Ctrl-Q quit"
            )

            app._request_active = True
            await pilot.press("ctrl+z", "ctrl+g")
            await pilot.pause()
            assert runtime.undo_calls == 1
            assert runtime.task_diff_calls == []

            app._request_active = False
            for event in (
                RequestStarted("change a file"),
                TaskChangeSummary("changed_files: changed.py"),
                RequestCompleted(None),
                TaskUndoRefused(("changed.py",)),
            ):
                await app._render_event(event)
            assert app._has_task_changes is True
            assert "UNDO\n✗ Undo refused\nConflicts: changed.py" in transcript_text(app)
            assert (
                rendered(app.query_one("#footer", Static))
                == "Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"
            )

    asyncio.run(scenario())


def test_textual_css_and_lifecycle_footer_remain_semantic(tmp_path: Path) -> None:
    css = HansTextualApp.CSS
    for color in ("$background", "$surface", "$accent", "$success", "$warning", "$error"):
        assert color in css
    for selector in (".user", ".tool", ".verification", ".final", ".error"):
        assert selector in css

    app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
    app._request_active = True
    app._state = "VERIFYING  ·  pytest -q"
    assert app._footer_text() == "◉ VERIFYING · Ctrl-C cancel · Ctrl-Q quit"

    app._request_active = False
    app._state = "COMPLETE"
    app._has_task_changes = True
    assert "✓ COMPLETE" in app._footer_text()
    app._state = "CANCELLED"
    assert "⏸ CANCELLED" in app._footer_text()
    app._state = "FAILED"
    assert "✗ FAILED" in app._footer_text()


def test_transcript_follow_tail_manual_anchor_resize_and_retention(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        app.MAX_TRANSCRIPT_ROWS = 3
        async with app.run_test(size=(48, 10)) as pilot:
            for number in range(5):
                await app._append_transcript(f"row {number}", "assistant")
            await pilot.pause(0.1)
            transcript = app.query_one("#transcript", VerticalScroll)
            transcript.scroll_home(animate=False, force=True, immediate=True)
            await pilot.pause(0.1)
            assert transcript.scroll_y == 0
            await app._append_transcript("new while reviewing", "assistant")
            await pilot.resize_terminal(72, 12)
            await pilot.pause()
            assert transcript.scroll_y < transcript.max_scroll_y
            rows = [rendered(child) for child in transcript.children if isinstance(child, Static)]
            assert rows == ["row 3", "row 4", "new while reviewing"]

            transcript.scroll_end(animate=False, force=True, immediate=True)
            await app._append_transcript("at bottom", "assistant")
            await pilot.pause()
            assert transcript.scroll_y == transcript.max_scroll_y
            assert isinstance(app.query_one("#composer"), TextArea)

    asyncio.run(scenario())


def test_completed_tool_retention_keeps_active_rows(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        app.MAX_TOOL_ROWS = 2
        async with app.run_test() as pilot:
            for number in range(3):
                call_id = f"tool-{number}"
                await app._render_event(ToolStarted(call_id, "read_file", f"{number}.txt"))
                await app._render_event(ToolCompleted(call_id, "read_file", f"{number}.txt", True))
            await app._render_event(ToolStarted("active", "read_file", "active.txt"))
            await pilot.pause()
            assert set(app._tool_widgets) == {"tool-1", "tool-2", "active"}
            text = transcript_text(app)
            assert "0.txt" not in text
            assert "◉ read_file active.txt" in text

    asyncio.run(scenario())


def test_cancel_leaves_composer_usable_and_errors_remain_visible(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestStarted("wait"))
            await pilot.press("ctrl+c")
            assert runtime.cancelled == 1
            assert rendered(app.query_one("#state-line", Static)).startswith("CANCELLED")
            await app._render_event(RequestCancelled())
            await pilot.press("o", "k", "enter")
            await pilot.pause()
            assert runtime.prompts == ["ok"]
            assert app.query_one("#composer", TextArea).text == ""
            assert "CANCELLED\nThe request was stopped" in transcript_text(app)

    asyncio.run(scenario())


def test_failed_verification_is_not_rendered_as_completion(tmp_path: Path) -> None:
    async def scenario() -> None:
        failed = VerificationEvidence("pytest -q", 1, True, True, False, "2026-09-28T00:00:00+00:00")
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestCompleted(failed))
            await pilot.pause()
            text = transcript_text(app)
            assert "FINAL RESULT\nVERIFICATION FAILED\npytest -q\nHANS did not claim completion." in text
            assert rendered(app.query_one("#state-line", Static)) == "FAILED"

    asyncio.run(scenario())


def test_tool_output_detail_is_bounded_clickable_and_copyable(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        app.MAX_TOOL_OUTPUT_CHARS = 8
        copied: list[str] = []
        monkeypatch.setattr(app, "copy_to_clipboard", copied.append)
        async with app.run_test() as pilot:
            await app._render_event(ToolStarted("call", "read_file", "a.py"))
            await app._render_event(ToolOutput("call", "abcdefghijk"))
            await app._render_event(ToolCompleted("call", "read_file", "a.py", True))
            await pilot.pause()

            assert app._tool_outputs["call"] == "abcdefgh\n… display truncated (3 characters omitted)"
            await pilot.press("ctrl+o")
            await pilot.pause()
            assert isinstance(app.screen, DetailScreen)
            assert app.screen.content == app._tool_outputs["call"]
            await pilot.press("ctrl+y")
            await pilot.pause()
            assert copied == ["abcdefgh\n… display truncated (3 characters omitted)"]
            await pilot.press("escape")
            await pilot.pause()

            await pilot.click(".tool")
            await pilot.pause()
            assert isinstance(app.screen, DetailScreen)
            await pilot.press("escape")

    asyncio.run(scenario())


def test_tool_output_retention_tracks_evicted_rows(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        app.MAX_TOOL_ROWS = 1
        async with app.run_test() as pilot:
            for call_id in ("old", "new"):
                await app._render_event(ToolStarted(call_id, "read_file", f"{call_id}.py"))
                await app._render_event(ToolOutput(call_id, call_id))
                await app._render_event(ToolCompleted(call_id, "read_file", f"{call_id}.py", True))
            await pilot.pause()
            assert "old" not in app._tool_outputs
            assert app._tool_outputs == {"new": "new"}
            assert app._latest_tool_call_id == "new"

    asyncio.run(scenario())


def test_textual_command_completion_accepts_selection_without_submitting(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press(*"/mo")
            await pilot.pause()
            assert app._command_suggestions == ("/models", "/mode")
            assert "/models" in rendered(app.query_one("#command-suggestions", Static))
            assert "/mode" in rendered(app.query_one("#command-suggestions", Static))
            assert runtime.prompts == []

            await pilot.press("down", "enter")
            await pilot.pause()
            assert app.query_one("#composer", TextArea).text == "/mode"
            assert app._command_suggestions == ()
            assert runtime.prompts == []

            await pilot.press("enter")
            await pilot.pause()
            assert runtime.prompts == []
            assert runtime.control_calls == [("reasoning_mode_status",)]

    asyncio.run(scenario())


def test_textual_command_completion_uses_dynamic_values_and_can_be_dismissed(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press(*"/models use")
            await pilot.pause()
            assert app._command_suggestions == ("/models use configured-model", "/models use large")
            assert runtime.control_calls == [("reasoning_mode_status",), ("model_status",)]
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#composer", TextArea).text == "/models use"
            assert app._command_suggestions == ()

            app.query_one("#composer", TextArea).text = ""
            await pilot.pause()
            await pilot.press(*"/mode")
            await pilot.pause()
            assert app._command_suggestions == ("/mode none", "/mode high")
            assert runtime.control_calls == [("reasoning_mode_status",), ("model_status",)]
            await pilot.press("escape")
            await pilot.pause()

            app.query_one("#composer", TextArea).text = ""
            await pilot.pause()
            await pilot.press(*"/models use l")
            await pilot.pause()
            assert app._command_suggestions == ("/models use large",)
            await pilot.press("tab")
            await pilot.pause()
            assert app.query_one("#composer", TextArea).text == "/models use large"
            assert app._command_suggestions == ()
            assert runtime.prompts == []

            app.query_one("#composer", TextArea).text = ""
            await pilot.pause()
            await pilot.press(*"/th")
            await pilot.pause()
            assert app._command_suggestions == ("/theme",)
            await pilot.press("escape")
            await pilot.pause()
            assert app.query_one("#composer", TextArea).text == "/th"
            assert app._command_suggestions == ()

            app.query_one("#composer", TextArea).text = ""
            await pilot.pause()
            await pilot.press(*"/mo", "shift+enter")
            await pilot.pause()
            assert app._command_suggestions == ()
            assert runtime.prompts == []

    asyncio.run(scenario())


def test_textual_approval_input_precedes_command_completion(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        runtime.approval_events = [ToolApprovalResolved("request-1", "call-1", True), RequestCompleted(None)]
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press(*"/mo")
            await pilot.pause()
            assert app._command_suggestions == ("/models", "/mode")
            await app._render_event(approval_event("request-1", "call-1", "write_file", "write"))
            assert app._command_suggestions == ()

            await pilot.press("y")
            await asyncio.wait_for(runtime.approval_started.wait(), timeout=1)
            assert runtime.approval_calls == [("request-1", "call-1", True)]
            assert app.query_one("#composer", TextArea).text == "/mo"
            assert runtime.prompts == []
            runtime.approval_release.set()
            await pilot.pause()
            await pilot.pause()

    asyncio.run(scenario())


def test_slash_commands_are_local_and_themes_are_session_only(tmp_path: Path) -> None:
    async def submit(pilot, text: str) -> None:
        await pilot.press(*text, "enter")
        await pilot.pause()

    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await submit(pilot, "/help")
            await submit(pilot, "/todo add write tests")
            await submit(pilot, "/todo list")
            await submit(pilot, "/todo done 1")
            await submit(pilot, "/todo list")
            await submit(pilot, "/todo remove 1")
            await submit(pilot, "/todo clear")
            await submit(pilot, "/todo add survive clear")
            await submit(pilot, "/theme light")
            await submit(pilot, "/permissions")
            await submit(pilot, "/permissions WRITE deny")
            await submit(pilot, "/clear")
            await submit(pilot, "/todo list")
            await submit(pilot, "/unknown")
            assert runtime.prompts == []
            assert runtime.permissions == {"read": "allow", "write": "deny", "execute": "allow"}
            assert runtime.control_calls == [
                ("permission", "write", "deny"),
                ("clear", None, None),
            ]
            text = transcript_text(app)
            assert format_help() in text
            assert "TODO added #1: write tests" in text
            assert "TODO\nx #1 write tests" in text
            assert "TODO completed #1: write tests" in text
            assert "TODO removed #1" in text
            assert "TODO cleared (0 items)" in text
            assert "TODO\n  #2 survive clear" in text
            assert app._theme_name == "light"
            assert app.theme == "hans-light"
            assert "Permissions\nread       allow\nwrite      allow\nexecute    allow" in text
            assert "write permission set to deny." in text
            assert "Conversation history cleared." in text
            assert "Workspace and local HANS state preserved." in text
            assert "Unknown command: /unknown" in text

            await submit(pilot, "/models")
            await submit(pilot, "/mode")
            await submit(pilot, "/mode HIGH")
            await submit(pilot, "/mode unsupported")
            await submit(pilot, "/models use large")
            await submit(pilot, "/models use does-not-exist")
            await pilot.press(*"/models use")
            await pilot.pause()
            await pilot.press("escape", "enter")
            await pilot.pause()
            await submit(pilot, "/mode high extra")
            assert runtime.prompts == []
            assert runtime.control_calls == [
                ("permission", "write", "deny"),
                ("clear", None, None),
                ("reasoning_mode_status",),
                ("model_status",),
                ("reasoning_mode_status",),
                ("reasoning_mode", "high"),
                ("reasoning_mode", "unsupported"),
                ("model_status",),
                ("select_model", "large"),
                ("reasoning_mode_status",),
                ("model_status",),
                ("select_model", "does-not-exist"),
            ]
            text = transcript_text(app)
            assert (
                "MODELS\n* Active: Configured Model (configured-model)\nEndpoint: configured endpoint"
                "\nContext: 128,000 tokens\nReasoning support: none, high"
                "\nCurrent reasoning: configured default\n- Large Model (large)"
                "\n  Endpoint: configured endpoint\n  Context: 256,000 tokens"
                "\n  Reasoning support: none, low"
            ) in text
            assert "REASONING MODE\nCurrent: configured default\nSupported: none, high" in text
            assert "REASONING\n✓ Mode set to high." in text
            assert "Unsupported reasoning mode: unsupported" in text
            assert "MODEL\n✓ Switched from configured-model to Large Model (large)." in text
            assert "✓ New conversation started." in text
            assert "✓ Reasoning reset to configured default." in text
            assert "Model selection is unavailable." in text
            assert "Usage: /models [use <model>]" in text
            assert "Usage: /mode [mode]" in text
            assert "configured endpoint" in text
            assert "http" not in text
            assert "api_key" not in text
            assert "model Large Model" in rendered(app.query_one("#hans-header", Static))

            await submit(pilot, "/theme light")
            assert app._theme_name == "light"
            assert app.theme == "hans-light"
            await submit(pilot, "/theme")
            assert isinstance(app.screen, ThemeScreen)
            await pilot.press("h")
            await pilot.pause()
            assert app._theme_name == "high-contrast"
            assert app.theme == "hans-high-contrast"
            assert runtime.prompts == []

            await submit(pilot, "ordinary prompt")
            assert runtime.prompts == ["ordinary prompt"]

    asyncio.run(scenario())


def test_copy_reports_clipboard_failure_without_runtime_submission(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        monkeypatch.setattr(app, "copy_to_clipboard", lambda _text: (_ for _ in ()).throw(RuntimeError()))
        async with app.run_test() as pilot:
            await app._render_event(AssistantMessageComplete("plain assistant text"))
            await pilot.press("ctrl+y")
            await pilot.pause()
            assert "clipboard unavailable" in rendered(app.query_one("#state-line", Static))
            assert runtime.prompts == []

    asyncio.run(scenario())
