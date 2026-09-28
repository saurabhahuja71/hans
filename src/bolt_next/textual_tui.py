"""Textual presentation shell for HANS semantic runtime events."""

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Protocol

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.events import Key
from textual.widgets import Static, TextArea

from bolt_next.events import (
    AssistantMessageComplete,
    AssistantMessageDelta,
    ConnectionChanged,
    HansEvent,
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
    VerificationStarted,
)
from bolt_next.tui_screen import is_exit_command


class Runtime(Protocol):
    def submit(self, message: str) -> AsyncIterator[HansEvent]: ...

    def cancel_active(self) -> None: ...

    def close(self) -> None: ...


class HansTextualApp(App[None]):
    """A semantic-event-only Textual UI for a HANS runtime."""

    CSS = """
    Screen {
        layout: vertical;
    }

    #hans-header, #status, #controls {
        padding: 0 1;
    }

    #transcript {
        height: 1fr;
        border: round $primary;
        padding: 0 1;
    }

    #tool-area {
        height: 8;
        border: round $secondary;
        padding: 0 1;
    }

    #composer {
        height: 7;
        border: round $accent;
    }

    .user {
        color: $accent;
    }

    .assistant {
        color: $text;
    }

    .error {
        color: $error;
    }

    .tool {
        color: $secondary;
    }

    .debug {
        color: $text-muted;
    }
    """

    MAX_TRANSCRIPT_ROWS = 300

    BINDINGS = [
        Binding("enter", "submit_or_exit", "send", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "newline", show=False, priority=True),
        Binding("ctrl+d", "submit_or_exit", "send", show=False),
        Binding("ctrl+c", "cancel_active", "cancel", show=False),
        Binding("ctrl+q", "exit_app", "exit", show=False),
    ]

    def __init__(self, runtime: Runtime, model: str, workspace: Path) -> None:
        super().__init__()
        self.runtime = runtime
        self.model = model
        self.workspace = workspace
        self._assistant_text = ""
        self._assistant_widget: Static | None = None
        self._pending_assistant_delta = ""
        self._assistant_flush_scheduled = False
        self._tool_widgets: dict[str, Static] = {}
        self._tool_text: dict[str, str] = {}
        self._transcript_rows: deque[Static] = deque()
        self._verification_failed = False
        self._request_active = False
        self._closed = False

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(False), id="hans-header")
        yield VerticalScroll(id="transcript")
        yield VerticalScroll(Static("tools", classes="tool"), id="tool-area")
        yield Static("ready", id="status")
        yield Static("Enter send · Shift+Enter newline · Ctrl-D send · Ctrl-C cancel · Ctrl-Q exit", id="controls")
        yield TextArea("", id="composer")

    def on_mount(self) -> None:
        self.query_one("#composer", TextArea).focus()

    def on_key(self, event: Key) -> None:
        actions = {
            "ctrl+d": self.action_submit_or_exit,
            "ctrl+c": self.action_cancel_active,
            "ctrl+q": self.action_exit_app,
        }
        action = actions.get(event.key)
        if action is not None:
            event.stop()
            event.prevent_default()
            action()

    def on_unmount(self) -> None:
        self._close_runtime()

    def _header_text(self, connected: bool) -> str:
        marker = "● connected" if connected else "○ disconnected"
        try:
            shown_workspace = "~/" + str(self.workspace.relative_to(Path.home()))
        except ValueError:
            shown_workspace = str(self.workspace)
        return f"HANS  model: {self.model}  connection: {marker}\nworkspace: {shown_workspace}"

    def _set_status(self, text: str) -> None:
        self.query_one("#status", Static).update(text)

    @staticmethod
    def _tool_stage(name: str, verification_failed: bool) -> str:
        if name == "read_file":
            return "investigating"
        if name == "write_file":
            return "correcting" if verification_failed else "acting"
        if name == "run_command":
            return "verifying"
        return "acting"

    @staticmethod
    def _failure_title(category: str) -> str:
        if category == "context":
            return "context limit exceeded"
        if category == "connection":
            return "connection failed"
        if category == "tool":
            return "tool failed"
        return "model request failed"

    @staticmethod
    def _debug_enabled() -> bool:
        return os.environ.get("HANS_DEBUG", "").strip().lower() in {"1", "true", "yes"}

    def _follow_transcript(self, transcript: VerticalScroll, follow: bool) -> None:
        if follow:
            self.call_after_refresh(transcript.scroll_end, animate=False, force=True, immediate=True)

    async def _trim_transcript(self) -> None:
        while len(self._transcript_rows) > self.MAX_TRANSCRIPT_ROWS:
            removed = self._transcript_rows.popleft()
            if removed is self._assistant_widget:
                self._assistant_widget = None
            await removed.remove()

    async def _append_transcript(self, text: str, classes: str) -> Static:
        transcript = self.query_one("#transcript", VerticalScroll)
        follow = transcript.scroll_y >= transcript.max_scroll_y
        widget = Static(text, classes=classes)
        await transcript.mount(widget)
        self._transcript_rows.append(widget)
        await self._trim_transcript()
        self._follow_transcript(transcript, follow)
        return widget

    def _update_assistant(self, text: str) -> None:
        if self._assistant_widget is None:
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        follow = transcript.scroll_y >= transcript.max_scroll_y
        self._assistant_widget.update(text)
        self._follow_transcript(transcript, follow)

    def _queue_assistant_delta(self, delta: str) -> None:
        self._pending_assistant_delta += delta
        if not self._assistant_flush_scheduled:
            self._assistant_flush_scheduled = True
            self.run_worker(self._flush_after_delay(), group="assistant-deltas", exclusive=False)

    async def _flush_after_delay(self) -> None:
        await asyncio.sleep(0.05)
        self._assistant_flush_scheduled = False
        await self._flush_assistant_deltas()

    async def _flush_assistant_deltas(self) -> None:
        if not self._pending_assistant_delta:
            return
        self._assistant_text += self._pending_assistant_delta
        self._pending_assistant_delta = ""
        if self._assistant_widget is None:
            self._assistant_widget = await self._append_transcript(self._assistant_text, "assistant")
        else:
            self._update_assistant(self._assistant_text)

    async def _tool_widget(self, call_id: str) -> Static:
        widget = self._tool_widgets.get(call_id)
        if widget is not None:
            return widget
        tool_area = self.query_one("#tool-area", VerticalScroll)
        text = f"◇ tool {call_id}"
        widget = Static(text, classes="tool")
        self._tool_widgets[call_id] = widget
        self._tool_text[call_id] = text
        await tool_area.mount(widget)
        return widget

    async def _render_event(self, event: HansEvent) -> None:
        if isinstance(event, UserMessageSubmitted):
            await self._append_transcript(f"> {event.message}", "user")
        elif isinstance(event, RequestStarted):
            await self._flush_assistant_deltas()
            self._assistant_text = ""
            self._assistant_widget = None
            self._verification_failed = False
            self._request_active = True
            self._set_status("thinking…")
        elif isinstance(event, AssistantMessageDelta):
            if event.delta:
                self._queue_assistant_delta(event.delta)
        elif isinstance(event, AssistantMessageComplete):
            await self._flush_assistant_deltas()
            final_text = event.text or self._assistant_text
            if self._assistant_widget is None and final_text:
                self._assistant_widget = await self._append_transcript(final_text, "assistant")
            elif self._assistant_widget is not None:
                self._update_assistant(final_text)
            self._assistant_text = final_text
        elif isinstance(event, ToolStarted):
            widget = await self._tool_widget(event.call_id)
            text = f"◇ {event.name}  {event.detail}".rstrip()
            self._tool_text[event.call_id] = text
            widget.update(text)
            self._set_status(self._tool_stage(event.name, self._verification_failed))
        elif isinstance(event, ToolOutput):
            widget = await self._tool_widget(event.call_id)
            text = self._tool_text.get(event.call_id, f"◇ tool {event.call_id}")
            text = f"{text}\n{event.output}".rstrip()
            self._tool_text[event.call_id] = text
            widget.update(text)
        elif isinstance(event, ToolCompleted):
            widget = await self._tool_widget(event.call_id)
            text = self._tool_text.get(event.call_id, f"◇ {event.name}  {event.detail}".rstrip())
            mark = "✓" if event.success else "✗"
            suffix = f"  exit {event.exit_code}" if event.exit_code is not None else ""
            text = f"{text}\n{mark} {event.name}  {event.detail}{suffix}".rstrip()
            self._tool_text[event.call_id] = text
            widget.update(text)
            stage = self._tool_stage(event.name, self._verification_failed)
            self._set_status(stage if event.success else f"tool failed: {event.name}")
        elif isinstance(event, VerificationStarted):
            self._set_status(f"verifying: {event.command}")
        elif isinstance(event, VerificationPassed):
            self._verification_failed = False
            self._set_status(f"verification passed: {event.evidence.command}")
        elif isinstance(event, VerificationFailed):
            self._verification_failed = True
            self._set_status(f"verification failed: {event.evidence.command}")
        elif isinstance(event, RequestCompleted):
            await self._flush_assistant_deltas()
            self._request_active = False
            if event.evidence is None:
                self._set_status("completed (verification not established)")
            elif event.evidence.success:
                self._set_status("completed")
            else:
                self._set_status("completed (verification failed)")
        elif isinstance(event, RequestCancelled):
            await self._flush_assistant_deltas()
            self._request_active = False
            self._set_status("cancelled")
            await self._append_transcript("✗ cancelled", "error")
        elif isinstance(event, RequestFailed):
            await self._flush_assistant_deltas()
            self._request_active = False
            title = self._failure_title(event.category)
            self._set_status(title)
            await self._append_transcript(f"✗ {title}\n{event.message}", "error")
            if self._debug_enabled() and event.debug_message:
                await self._append_transcript(f"debug ({event.category}): {event.debug_message}", "debug")
        elif isinstance(event, ConnectionChanged):
            self.query_one("#hans-header", Static).update(self._header_text(event.connected))

    def action_submit_or_exit(self) -> None:
        composer = self.query_one("#composer", TextArea)
        prompt = composer.text
        if not prompt.strip() or is_exit_command(prompt):
            self.action_exit_app()
            return
        if self._request_active:
            return
        composer.text = ""
        self._request_active = True
        self._set_status("thinking…")
        self._submit(prompt)

    def action_insert_newline(self) -> None:
        self.query_one("#composer", TextArea).insert("\n")

    def action_cancel_active(self) -> None:
        if self._request_active:
            self.runtime.cancel_active()
            self._set_status("cancelling…")
        else:
            self.query_one("#composer", TextArea).text = ""

    def action_exit_app(self) -> None:
        if self._request_active:
            self.runtime.cancel_active()
        self._close_runtime()
        self.exit()

    @work(exclusive=True)
    async def _submit(self, prompt: str) -> None:
        try:
            async for event in self.runtime.submit(prompt):
                await self._render_event(event)
        finally:
            self._request_active = False

    def _close_runtime(self) -> None:
        if not self._closed:
            self.runtime.close()
            self._closed = True


async def run_textual_tui(runtime: Runtime, model: str, workspace: Path) -> None:
    """Run the Textual application without exposing SDK runtime details."""
    await HansTextualApp(runtime, model, workspace).run_async()
