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
from textual.containers import Container, VerticalScroll
from textual.events import Key
from textual.screen import ModalScreen
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
from bolt_next.tui_screen import format_change_summary, is_exit_command


class Runtime(Protocol):
    def submit(self, message: str) -> AsyncIterator[HansEvent]: ...

    def cancel_active(self) -> None: ...

    def task_diff(self, *, max_chars: int | None = None) -> TaskDiff: ...

    def undo_task(self) -> TaskUndoSucceeded | TaskUndoRefused: ...

    def close(self) -> None: ...


class TaskDiffScreen(ModalScreen[None]):
    """Bounded task-scoped diff review, independent of the model."""

    BINDINGS = [Binding("escape", "close", "back", show=False)]

    CSS = """
    TaskDiffScreen {
        background: $background;
    }

    #task-diff {
        width: 100%;
        height: 100%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }

    #task-diff-title {
        color: $accent;
        text-style: bold;
        height: 1;
    }

    #task-diff-body {
        height: 1fr;
        margin-top: 1;
    }

    .diff-content {
        color: $text;
    }
    """

    def __init__(self, diff: str) -> None:
        super().__init__()
        self.diff = diff or "(no HANS task changes)"

    def compose(self) -> ComposeResult:
        with Container(id="task-diff"):
            yield Static("TASK DIFF  ·  Esc back", id="task-diff-title", markup=False)
            with VerticalScroll(id="task-diff-body"):
                yield Static(self.diff, classes="diff-content", markup=False)

    def action_close(self) -> None:
        self.dismiss()


class HansTextualApp(App[None]):
    """A semantic-event-only Textual UI for a HANS runtime."""

    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #hans-header {
        height: 3;
        padding: 0 2;
        background: $surface;
        border-bottom: tall $primary;
    }

    #transcript {
        height: 1fr;
        margin: 1 1 0 1;
        border: round $primary;
        padding: 0 1;
        background: $surface;
    }

    #state-line {
        height: 1;
        margin: 0 2;
        color: $accent;
        text-style: bold;
    }

    #footer {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $surface;
    }

    #composer {
        height: 3;
        min-height: 3;
        max-height: 8;
        margin: 0 1 1 1;
        border: round $accent;
    }

    .user {
        color: $accent;
        margin-top: 1;
    }

    .assistant {
        color: $text;
        margin-top: 1;
    }

    .tool {
        color: $secondary;
        margin-top: 1;
    }

    .verification {
        color: $success;
        margin-top: 1;
    }

    .change {
        color: $warning;
        margin-top: 1;
    }

    .final {
        color: $success;
        margin-top: 1;
    }

    .error {
        color: $error;
        margin-top: 1;
    }

    .debug {
        color: $text-muted;
        margin-top: 1;
    }
    """

    MAX_TRANSCRIPT_ROWS = 300
    MAX_TOOL_ROWS = 100
    MAX_TOOL_DETAIL_CHARS = 240
    MAX_TOOL_OUTPUT_CHARS = 4_000
    MAX_TASK_DIFF_CHARS = 4_000
    MAX_COMPOSER_ROWS = 6

    BINDINGS = [
        Binding("enter", "submit_or_exit", "send", show=False, priority=True),
        Binding("shift+enter", "insert_newline", "newline", show=False, priority=True),
        Binding("ctrl+d", "submit_or_exit", "send", show=False),
        Binding("ctrl+c", "cancel_active", "cancel", show=False),
        Binding("ctrl+g", "show_task_diff", "diff", show=False),
        Binding("ctrl+z", "undo_task", "undo", show=False),
        Binding("ctrl+q", "exit_app", "exit", show=False),
    ]

    def __init__(self, runtime: Runtime, model: str, workspace: Path) -> None:
        super().__init__()
        self.runtime = runtime
        self.model = model
        self.workspace = workspace
        self._connected = False
        self._assistant_text = ""
        self._assistant_widget: Static | None = None
        self._pending_assistant_delta = ""
        self._assistant_flush_scheduled = False
        self._tool_widgets: dict[str, Static] = {}
        self._tool_names: dict[str, str] = {}
        self._tool_details: dict[str, str] = {}
        self._tool_results: dict[str, str] = {}
        self._tool_purposes: dict[str, str] = {}
        self._completed_tool_rows: deque[str] = deque()
        self._transcript_rows: deque[Static] = deque()
        self._verification_failed = False
        self._request_active = False
        self._closed = False

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="hans-header", markup=False)
        yield VerticalScroll(id="transcript")
        yield Static("IDLE", id="state-line", markup=False)
        yield Static("Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit", id="footer", markup=False)
        yield TextArea("", id="composer")

    def on_mount(self) -> None:
        self.query_one("#composer", TextArea).focus()

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        rows = min(self.MAX_COMPOSER_ROWS, max(1, event.text_area.text.count("\n") + 1))
        event.text_area.styles.height = rows + 2

    def on_key(self, event: Key) -> None:
        actions = {
            "ctrl+d": self.action_submit_or_exit,
            "ctrl+c": self.action_cancel_active,
            "ctrl+g": self.action_show_task_diff,
            "ctrl+z": self.action_undo_task,
            "ctrl+q": self.action_exit_app,
        }
        action = actions.get(event.key)
        if action is not None:
            event.stop()
            event.prevent_default()
            action()

    def on_unmount(self) -> None:
        self._close_runtime()

    def _header_text(self) -> str:
        marker = "● CONNECTED" if self._connected else "○ DISCONNECTED"
        try:
            shown_workspace = "~/" + str(self.workspace.relative_to(Path.home()))
        except ValueError:
            shown_workspace = str(self.workspace)
        return f"HANS  |  model {self.model}  |  {marker}\nworkspace {shown_workspace}"

    def _set_state(self, state: str, detail: str = "") -> None:
        shown_detail = self._compact(detail, self.MAX_TOOL_DETAIL_CHARS)
        text = state if not shown_detail else f"{state}  ·  {shown_detail}"
        self.query_one("#state-line", Static).update(text)

    @staticmethod
    def _tool_stage(name: str, verification_failed: bool, purpose: str = "inspect") -> str:
        if name in {"list_directory", "search_files", "read_file"}:
            return "INVESTIGATING"
        if name in {"replace_in_file", "write_file"}:
            return "CORRECTING" if verification_failed else "EDITING"
        if name == "run_command":
            return "VERIFYING" if purpose == "verify" else "INVESTIGATING"
        return "EDITING"

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}…"

    def _display_tool_output(self, output: str) -> str:
        if len(output) <= self.MAX_TOOL_OUTPUT_CHARS:
            return output
        omitted = len(output) - self.MAX_TOOL_OUTPUT_CHARS
        return f"{output[:self.MAX_TOOL_OUTPUT_CHARS]}\n… display truncated ({omitted} characters omitted)"

    @staticmethod
    def _failure_title(category: str) -> str:
        if category == "configuration":
            return "configuration failed"
        if category == "authentication":
            return "authentication failed"
        if category == "context":
            return "context limit exceeded"
        if category == "connection":
            return "connection failed"
        if category == "tool":
            return "tool failed"
        if category == "runtime":
            return "runtime failed"
        return "model request failed"

    @staticmethod
    def _debug_enabled() -> bool:
        return os.environ.get("HANS_DEBUG", "").strip().lower() in {"1", "true", "yes"}

    def _follow_transcript(self, transcript: VerticalScroll, follow: bool, position: float) -> None:
        if follow:
            self.call_after_refresh(transcript.scroll_end, animate=False, force=True, immediate=True)
        else:
            self.call_after_refresh(
                lambda: transcript.scroll_to(
                    y=position, animate=False, force=True, immediate=True
                )
            )

    def _forget_row(self, widget: Static) -> None:
        try:
            self._transcript_rows.remove(widget)
        except ValueError:
            return
        for call_id, tool_widget in tuple(self._tool_widgets.items()):
            if tool_widget is widget:
                self._tool_widgets.pop(call_id, None)
                self._tool_names.pop(call_id, None)
                self._tool_details.pop(call_id, None)
                self._tool_results.pop(call_id, None)
                self._tool_purposes.pop(call_id, None)
                try:
                    self._completed_tool_rows.remove(call_id)
                except ValueError:
                    pass
                break
        if widget is self._assistant_widget:
            self._assistant_widget = None

    async def _trim_transcript(self) -> None:
        while len(self._transcript_rows) > self.MAX_TRANSCRIPT_ROWS:
            removed = self._transcript_rows.popleft()
            for call_id, tool_widget in tuple(self._tool_widgets.items()):
                if tool_widget is removed:
                    self._tool_widgets.pop(call_id, None)
                    self._tool_names.pop(call_id, None)
                    self._tool_details.pop(call_id, None)
                    self._tool_results.pop(call_id, None)
                    self._tool_purposes.pop(call_id, None)
                    try:
                        self._completed_tool_rows.remove(call_id)
                    except ValueError:
                        pass
                    break
            if removed is self._assistant_widget:
                self._assistant_widget = None
            await removed.remove()

    async def _append_transcript(self, text: str, classes: str) -> Static:
        transcript = self.query_one("#transcript", VerticalScroll)
        position = transcript.scroll_y
        follow = position >= transcript.max_scroll_y
        widget = Static(text, classes=classes, markup=False)
        await transcript.mount(widget)
        self._transcript_rows.append(widget)
        await self._trim_transcript()
        self._follow_transcript(transcript, follow, position)
        return widget

    def _update_assistant(self, text: str) -> None:
        if self._assistant_widget is None:
            return
        transcript = self.query_one("#transcript", VerticalScroll)
        position = transcript.scroll_y
        follow = position >= transcript.max_scroll_y
        self._assistant_widget.update(text)
        self._follow_transcript(transcript, follow, position)

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
            self._assistant_widget = await self._append_transcript(
                f"HANS\n{self._assistant_text}", "assistant"
            )
        else:
            self._update_assistant(f"HANS\n{self._assistant_text}")

    @staticmethod
    def _tool_result_summary(name: str, output: str) -> str:
        if output.startswith("Error"):
            return ""
        if name == "read_file":
            for line in output.splitlines()[:3]:
                if line.startswith("total_lines:"):
                    return f"{line.partition(':')[2].strip()} lines"
            line_count = len(output.splitlines())
            return f"{line_count} line{'s' if line_count != 1 else ''}"
        if name == "search_files":
            matches = sum(1 for line in output.splitlines() if line and not line.startswith("..."))
            return f"{matches} match{'es' if matches != 1 else ''}"
        if name == "list_directory":
            entries = sum(1 for line in output.splitlines() if line and not line.startswith("..."))
            return f"{entries} entr{'y' if entries == 1 else 'ies'}"
        if name in {"replace_in_file", "write_file"}:
            return "updated"
        return ""

    def _tool_text(self, call_id: str, state: str, exit_code: int | None = None) -> str:
        name = self._tool_names.get(call_id, "tool")
        detail = self._tool_details.get(call_id, "")
        result = self._tool_results.get(call_id, "")
        target = f" {detail}" if detail else ""
        suffix = f"\n  {result}" if result else ""
        if exit_code is not None:
            suffix += f"\n  exit {exit_code}"
        marker = {"RUNNING": "◉", "DONE": "✓", "FAILED": "✗"}[state]
        return f"{marker} {name}{target}{suffix}"

    async def _tool_widget(self, call_id: str) -> Static:
        widget = self._tool_widgets.get(call_id)
        if widget is not None:
            return widget
        widget = await self._append_transcript("◉ tool", "tool")
        self._tool_widgets[call_id] = widget
        return widget

    async def _trim_completed_tools(self) -> None:
        while len(self._completed_tool_rows) > self.MAX_TOOL_ROWS:
            call_id = self._completed_tool_rows.popleft()
            widget = self._tool_widgets.pop(call_id, None)
            self._tool_names.pop(call_id, None)
            self._tool_details.pop(call_id, None)
            self._tool_results.pop(call_id, None)
            self._tool_purposes.pop(call_id, None)
            if widget is not None:
                self._forget_row(widget)
                await widget.remove()

    async def _render_event(self, event: HansEvent) -> None:
        if isinstance(event, UserMessageSubmitted):
            await self._append_transcript(f"YOU\n> {event.message}", "user")
        elif isinstance(event, RequestStarted):
            await self._flush_assistant_deltas()
            self._assistant_text = ""
            self._assistant_widget = None
            self._verification_failed = False
            self._request_active = True
            self._set_state("INVESTIGATING")
        elif isinstance(event, AssistantMessageDelta):
            if event.delta:
                self._queue_assistant_delta(event.delta)
        elif isinstance(event, AssistantMessageComplete):
            await self._flush_assistant_deltas()
            final_text = event.text or self._assistant_text
            if self._assistant_widget is None and final_text:
                self._assistant_widget = await self._append_transcript(
                    f"HANS\n{final_text}", "assistant"
                )
            elif self._assistant_widget is not None:
                self._update_assistant(f"HANS\n{final_text}")
            self._assistant_text = final_text
        elif isinstance(event, ToolStarted):
            self._tool_names[event.call_id] = event.name
            self._tool_details[event.call_id] = self._compact(event.detail, self.MAX_TOOL_DETAIL_CHARS)
            self._tool_purposes[event.call_id] = event.purpose
            widget = await self._tool_widget(event.call_id)
            widget.update(self._tool_text(event.call_id, "RUNNING"))
            self._set_state(self._tool_stage(event.name, self._verification_failed, event.purpose))
        elif isinstance(event, ToolOutput):
            name = self._tool_names.get(event.call_id, "")
            summary = self._tool_result_summary(name, event.output)
            if summary:
                self._tool_results[event.call_id] = summary
            if self._debug_enabled():
                output = self._display_tool_output(event.output)
                await self._append_transcript(f"DEBUG TOOL [{event.call_id}] OUTPUT\n{output}", "debug")
        elif isinstance(event, ToolCompleted):
            self._tool_names.setdefault(event.call_id, event.name)
            self._tool_details.setdefault(event.call_id, self._compact(event.detail, self.MAX_TOOL_DETAIL_CHARS))
            widget = await self._tool_widget(event.call_id)
            status = "DONE" if event.success else "FAILED"
            widget.update(self._tool_text(event.call_id, status, event.exit_code))
            purpose = self._tool_purposes.get(event.call_id, "inspect")
            self._completed_tool_rows.append(event.call_id)
            await self._trim_completed_tools()
            if event.success:
                self._set_state(self._tool_stage(event.name, self._verification_failed, purpose))
            else:
                self._set_state("FAILED", f"tool {event.name}")
        elif isinstance(event, VerificationStarted):
            self._set_state("VERIFYING", event.command)
        elif isinstance(event, VerificationPassed):
            self._verification_failed = False
            await self._append_transcript(
                f"VERIFICATION\n✓ {event.evidence.command} passed", "verification"
            )
            self._set_state("VERIFYING", f"passed · {event.evidence.command}")
        elif isinstance(event, VerificationFailed):
            self._verification_failed = True
            await self._append_transcript(
                f"VERIFICATION\n✗ {event.evidence.command} failed", "error"
            )
            self._set_state("CORRECTING", event.evidence.command)
        elif isinstance(event, RequestCompleted):
            await self._flush_assistant_deltas()
            self._request_active = False
            if event.evidence is None:
                result = "COMPLETE\nVerification not established"
                result_class = "final"
                state = "COMPLETE"
            elif event.evidence.success:
                result = f"COMPLETE\nVerified: {event.evidence.command}"
                result_class = "final"
                state = "COMPLETE"
            else:
                result = (
                    f"VERIFICATION FAILED\n{event.evidence.command}\n"
                    "HANS did not claim completion."
                )
                result_class = "error"
                state = "FAILED"
            await self._append_transcript(f"FINAL RESULT\n{result}", result_class)
            self._set_state(state)
        elif isinstance(event, RequestCancelled):
            await self._flush_assistant_deltas()
            self._request_active = False
            self._set_state("CANCELLED")
            await self._append_transcript("CANCELLED\nThe request was stopped. You can send another prompt.", "error")
        elif isinstance(event, RequestFailed):
            await self._flush_assistant_deltas()
            self._request_active = False
            title = self._failure_title(event.category)
            self._set_state("FAILED", title)
            await self._append_transcript(f"ERROR\n✗ {title}\n{event.message}", "error")
            if self._debug_enabled() and event.debug_message:
                await self._append_transcript(
                    f"DEBUG ({event.category})\n{event.debug_message}", "debug"
                )
        elif isinstance(event, TaskChangeSummary):
            await self._append_transcript(
                f"CHANGES\n{format_change_summary(event.summary)}", "change"
            )
        elif isinstance(event, TaskDiff):
            self.push_screen(TaskDiffScreen(event.diff[: self.MAX_TASK_DIFF_CHARS]))
        elif isinstance(event, TaskUndoSucceeded):
            details = []
            if event.restored_files:
                details.append("restored: " + ", ".join(event.restored_files))
            if event.removed_files:
                details.append("removed: " + ", ".join(event.removed_files))
            feedback = "; ".join(details) if details else "no HANS task changes"
            self._set_state("IDLE", "undo complete")
            await self._append_transcript(f"UNDO\n✓ Undo completed · {feedback}", "change")
        elif isinstance(event, TaskUndoRefused):
            conflicts = ", ".join(event.conflicting_files) or "task changes"
            self._set_state("IDLE", "undo refused")
            await self._append_transcript(f"UNDO\n✗ Undo refused\nConflicts: {conflicts}", "error")
        elif isinstance(event, ConnectionChanged):
            self._connected = event.connected
            self.query_one("#hans-header", Static).update(self._header_text())

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
        self._set_state("INVESTIGATING")
        self._submit(prompt)

    def action_insert_newline(self) -> None:
        self.query_one("#composer", TextArea).insert("\n")

    def action_cancel_active(self) -> None:
        if self._request_active:
            self.runtime.cancel_active()
            self._set_state("CANCELLED", "cancellation requested")
        else:
            self.query_one("#composer", TextArea).text = ""

    def action_show_task_diff(self) -> None:
        if not self._request_active:
            self._render_task_event(self.runtime.task_diff(max_chars=self.MAX_TASK_DIFF_CHARS))

    def action_undo_task(self) -> None:
        if not self._request_active:
            self._render_task_event(self.runtime.undo_task())

    def action_exit_app(self) -> None:
        if self._request_active:
            self.runtime.cancel_active()
        self._close_runtime()
        self.exit()

    @work(exclusive=False)
    async def _render_task_event(self, event: TaskDiff | TaskUndoSucceeded | TaskUndoRefused) -> None:
        await self._render_event(event)

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
