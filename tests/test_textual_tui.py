from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from textual.containers import VerticalScroll
from textual.widgets import Static, TextArea

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
    VerificationEvidence,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)
from bolt_next.textual_tui import HansTextualApp, TaskDiffScreen


class FakeRuntime:
    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events or []
        self.prompts: list[str] = []
        self.cancelled = 0
        self.closed = 0
        self.task_diff_calls: list[int | None] = []
        self.undo_calls = 0
        self.diff_event = TaskDiff("--- a/task.txt\n+++ b/task.txt\n+updated")
        self.undo_event: TaskUndoSucceeded | TaskUndoRefused = TaskUndoSucceeded((), ())

    async def submit(self, message: str) -> AsyncIterator[object]:
        self.prompts.append(message)
        for event in self.events:
            yield event
            await asyncio.sleep(0)

    def cancel_active(self) -> None:
        self.cancelled += 1

    def task_diff(self, *, max_chars: int | None = None) -> TaskDiff:
        self.task_diff_calls.append(max_chars)
        return self.diff_event

    def undo_task(self) -> TaskUndoSucceeded | TaskUndoRefused:
        self.undo_calls += 1
        return self.undo_event

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
            assert "Ctrl-G diff · Ctrl-Z undo" in rendered(app.query_one("#footer", Static))
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
    asyncio.run(scenario(("ctrl+q",)))


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


def test_lifecycle_states_are_derived_from_semantic_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        failed = VerificationEvidence("pytest -q", 1, True, True, False, "2026-09-28T00:00:00+00:00")
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestStarted("inspect"))
            assert rendered(app.query_one("#state-line", Static)) == "INVESTIGATING"
            await app._render_event(ToolStarted("write", "write_file", "a.py"))
            assert rendered(app.query_one("#state-line", Static)) == "EDITING"
            await app._render_event(VerificationStarted("verify", "pytest -q"))
            assert rendered(app.query_one("#state-line", Static)).startswith("VERIFYING")
            await app._render_event(VerificationFailed("verify", failed))
            assert rendered(app.query_one("#state-line", Static)).startswith("CORRECTING")
            await app._render_event(RequestFailed("connection", "unavailable"))
            assert rendered(app.query_one("#state-line", Static)).startswith("FAILED")
            await app._render_event(RequestCancelled())
            assert rendered(app.query_one("#state-line", Static)) == "CANCELLED"
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

            app._request_active = True
            await pilot.press("ctrl+z", "ctrl+g")
            await pilot.pause()
            assert runtime.undo_calls == 1
            assert runtime.task_diff_calls == []

            app._request_active = False
            await app._render_event(TaskUndoRefused(("changed.py",)))
            assert "UNDO\n✗ Undo refused\nConflicts: changed.py" in transcript_text(app)

    asyncio.run(scenario())


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
