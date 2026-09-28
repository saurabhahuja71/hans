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
    ToolCompleted,
    ToolOutput,
    ToolStarted,
    UserMessageSubmitted,
    VerificationEvidence,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)
from bolt_next.textual_tui import HansTextualApp


class FakeRuntime:
    def __init__(self, events: list[object] | None = None) -> None:
        self.events = events or []
        self.prompts: list[str] = []
        self.cancelled = 0
        self.closed = 0

    async def submit(self, message: str) -> AsyncIterator[object]:
        self.prompts.append(message)
        for event in self.events:
            yield event
            await asyncio.sleep(0)

    def cancel_active(self) -> None:
        self.cancelled += 1

    def close(self) -> None:
        self.closed += 1


def rendered(widget: Static) -> str:
    return str(widget.render())


def test_textual_shell_submits_on_enter_and_inserts_newlines_with_shift_enter(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            assert "HANS" in rendered(app.query_one("#hans-header", Static))
            assert isinstance(app.query_one("#transcript"), VerticalScroll)
            assert isinstance(app.query_one("#tool-area"), VerticalScroll)
            assert "Enter send · Shift+Enter newline · Ctrl-D send · Ctrl-C cancel · Ctrl-Q exit" == rendered(
                app.query_one("#controls", Static)
            )
            composer = app.query_one("#composer", TextArea)

            await pilot.press("h", "i", "enter")
            await pilot.pause()
            assert runtime.prompts == ["hi"]
            assert composer.text == ""

            await pilot.press(*"line one", "shift+enter", *"line two")
            assert composer.text == "line one\nline two"
            assert runtime.prompts == ["hi"]
            await pilot.press("enter")
            await pilot.pause()
            assert runtime.prompts == ["hi", "line one\nline two"]
            assert composer.text == ""

            await pilot.press("o", "k", "ctrl+d")
            await pilot.pause()
            assert runtime.prompts == ["hi", "line one\nline two", "ok"]
            assert composer.text == ""

        assert runtime.closed == 1

    asyncio.run(scenario())


def test_empty_or_exit_command_submits_nothing_and_exits(tmp_path: Path) -> None:
    async def empty_scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+d")
            await pilot.pause()
            assert runtime.prompts == []
        assert runtime.closed == 1

    async def command_scenario(command: str) -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press(*command, "enter")
            await pilot.pause()
            assert runtime.prompts == []
        assert runtime.closed == 1

    async def ctrl_q_scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+q")
            await pilot.pause()
            assert runtime.prompts == []
        assert runtime.closed == 1

    asyncio.run(empty_scenario())
    asyncio.run(command_scenario("exit"))
    asyncio.run(command_scenario("quit"))
    asyncio.run(ctrl_q_scenario())


def test_textual_shell_renders_events_without_duplicate_assistant_or_tool_rows(tmp_path: Path) -> None:
    async def scenario() -> None:
        evidence = VerificationEvidence("pytest -q", 0, True, True, True, "2026-09-28T00:00:00+00:00")
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            events = [
                UserMessageSubmitted("inspect this"),
                RequestStarted("inspect this"),
                AssistantMessageDelta("Hello"),
                AssistantMessageDelta(" there"),
                AssistantMessageComplete("Hello there"),
                ToolStarted("call-a", "read_file", "a.py"),
                ToolStarted("call-b", "run_command", "pytest -q"),
                ToolOutput("call-b", "tests passed"),
                ToolOutput("call-a", "source"),
                ToolCompleted("call-b", "run_command", "pytest -q", True, 0),
                ToolCompleted("call-a", "read_file", "a.py", True),
                VerificationPassed("call-b", evidence),
                RequestCompleted(evidence),
                ConnectionChanged(True),
            ]
            for event in events:
                await app._render_event(event)
            await pilot.pause()

            transcript = app.query_one("#transcript", VerticalScroll)
            transcript_text = "\n".join(rendered(child) for child in transcript.children if isinstance(child, Static))
            assert transcript_text.count("Hello there") == 1
            assert "> inspect this" in transcript_text

            assert set(app._tool_widgets) == {"call-a", "call-b"}
            tool_a = rendered(app._tool_widgets["call-a"])
            tool_b = rendered(app._tool_widgets["call-b"])
            assert "read_file  a.py" in tool_a
            assert "source" in tool_a
            assert "run_command  pytest -q" in tool_b
            assert "tests passed" in tool_b
            assert "exit 0" in tool_b
            assert "completed" == rendered(app.query_one("#status", Static))
            assert "connected" in rendered(app.query_one("#hans-header", Static))

    asyncio.run(scenario())


def test_transcript_resize_and_scroll_position_are_preserved_when_scrolled_up(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test(size=(48, 16)) as pilot:
            for number in range(30):
                await app._append_transcript(f"line {number}", "assistant")
            await pilot.pause()
            transcript = app.query_one("#transcript", VerticalScroll)
            transcript.scroll_home(animate=False, force=True, immediate=True)
            await pilot.pause()
            await app._append_transcript("new line while reviewing", "assistant")
            await pilot.resize_terminal(72, 22)
            await pilot.pause()
            assert transcript.scroll_y < transcript.max_scroll_y
            assert "new line while reviewing" in rendered(list(transcript.children)[-1])
            assert "HANS" in rendered(app.query_one("#hans-header", Static))
            assert isinstance(app.query_one("#composer"), TextArea)

            transcript.scroll_end(animate=False, force=True, immediate=True)
            await pilot.pause()
            await app._append_transcript("new line at bottom", "assistant")
            await pilot.pause()
            assert transcript.scroll_y == transcript.max_scroll_y

    asyncio.run(scenario())


def test_cancel_and_failure_events_stay_visible(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test(size=(80, 24)) as pilot:
            await app._render_event(RequestStarted("wait"))
            app.action_cancel_active()
            assert runtime.cancelled == 1
            await app._render_event(RequestCancelled())
            await app._render_event(RequestFailed("connection", "model endpoint unavailable"))
            await pilot.pause()

            transcript = app.query_one("#transcript", VerticalScroll)
            transcript_text = "\n".join(rendered(child) for child in transcript.children if isinstance(child, Static))
            assert "✗ cancelled" in transcript_text
            assert "✗ connection failed" in transcript_text
            assert "model endpoint unavailable" in transcript_text
            assert rendered(app.query_one("#status", Static)) == "connection failed"

    asyncio.run(scenario())


def test_verification_states_and_ctrl_c_are_rendered_from_semantic_events(tmp_path: Path) -> None:
    async def scenario() -> None:
        evidence = VerificationEvidence("pytest -q", 1, False, True, True, "2026-09-28T00:00:00+00:00")
        runtime = FakeRuntime()
        app = HansTextualApp(runtime, "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestStarted("verify"))
            await pilot.press("ctrl+c")
            assert runtime.cancelled == 1
            assert rendered(app.query_one("#status", Static)) == "cancelling…"

            await app._render_event(VerificationStarted("verify-1", "pytest -q"))
            assert rendered(app.query_one("#status", Static)) == "verifying: pytest -q"
            await app._render_event(VerificationFailed("verify-1", evidence))
            assert rendered(app.query_one("#status", Static)) == "verification failed: pytest -q"
            await app._render_event(RequestCancelled())
            assert rendered(app.query_one("#status", Static)) == "cancelled"
            await pilot.press("o", "k", "enter")
            await pilot.pause()
            assert runtime.prompts == ["ok"]
            assert app.query_one("#composer", TextArea).text == ""

    asyncio.run(scenario())


def test_tool_stages_follow_semantic_verification_state(tmp_path: Path) -> None:
    async def scenario() -> None:
        evidence = VerificationEvidence("pytest -q", 1, False, True, False, "2026-09-28T00:00:00+00:00")
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestStarted("inspect"))
            await app._render_event(ToolStarted("read", "read_file", "README.md"))
            assert rendered(app.query_one("#status", Static)) == "investigating"

            await app._render_event(ToolStarted("write", "write_file", "notes.txt"))
            assert rendered(app.query_one("#status", Static)) == "acting"

            await app._render_event(VerificationFailed("verify", evidence))
            await app._render_event(ToolStarted("correct", "write_file", "notes.txt"))
            assert rendered(app.query_one("#status", Static)) == "correcting"

            await app._render_event(ToolStarted("test", "run_command", "pytest -q"))
            assert rendered(app.query_one("#status", Static)) == "verifying"
            await app._render_event(ToolCompleted("test", "run_command", "pytest -q", False, 1))
            assert rendered(app.query_one("#status", Static)) == "tool failed: run_command"
            assert "✗ run_command  pytest -q  exit 1" in rendered(app._tool_widgets["test"])
            await pilot.pause()

    asyncio.run(scenario())


def test_failure_details_respect_debug_mode_and_connection_events(tmp_path: Path, monkeypatch) -> None:
    async def scenario(debug: bool) -> str:
        if debug:
            monkeypatch.setenv("HANS_DEBUG", "1")
        else:
            monkeypatch.delenv("HANS_DEBUG", raising=False)
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        async with app.run_test() as pilot:
            await app._render_event(RequestFailed("context", "conversation is too large", "provider trace details"))
            await app._render_event(ConnectionChanged(False))
            await pilot.pause()
            transcript = app.query_one("#transcript", VerticalScroll)
            transcript_text = "\n".join(rendered(child) for child in transcript.children if isinstance(child, Static))
            assert rendered(app.query_one("#status", Static)) == "context limit exceeded"
            assert "disconnected" in rendered(app.query_one("#hans-header", Static))
            return transcript_text

    plain = asyncio.run(scenario(False))
    assert "✗ context limit exceeded" in plain
    assert "conversation is too large" in plain
    assert "provider trace details" not in plain

    debug = asyncio.run(scenario(True))
    assert "debug (context): provider trace details" in debug


def test_transcript_retention_and_unicode_reflow(tmp_path: Path) -> None:
    async def scenario() -> None:
        app = HansTextualApp(FakeRuntime(), "test-model", tmp_path)
        app.MAX_TRANSCRIPT_ROWS = 3
        async with app.run_test(size=(48, 16)) as pilot:
            for number in range(5):
                await app._append_transcript(f"row {number}", "assistant")
            await app._append_transcript("日本語 and wide text: 漢字", "assistant")
            await pilot.resize_terminal(72, 22)
            await pilot.pause()

            transcript = app.query_one("#transcript", VerticalScroll)
            rows = [rendered(child) for child in transcript.children if isinstance(child, Static)]
            assert rows == ["row 3", "row 4", "日本語 and wide text: 漢字"]
            assert "HANS" in rendered(app.query_one("#hans-header", Static))
            assert isinstance(app.query_one("#composer"), TextArea)

    asyncio.run(scenario())
