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
from textual.events import Click, Key
from textual.screen import ModalScreen
from textual.theme import Theme
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
from bolt_next.tui_screen import (
    THEME_NAMES,
    TodoList,
    display_bounded,
    footer_text,
    format_change_summary,
    handle_local_command,
    is_exit_command,
    task_summary_has_hans_changes,
)


class Runtime(Protocol):
    def submit(self, message: str) -> AsyncIterator[HansEvent]: ...

    def cancel_active(self) -> None: ...

    def task_diff(self, *, max_chars: int | None = None) -> TaskDiff: ...

    def undo_task(self) -> TaskUndoSucceeded | TaskUndoRefused: ...

    def close(self) -> None: ...


class DetailScreen(ModalScreen[None]):
    """Reusable bounded plain-text review overlay."""

    BINDINGS = [
        Binding("escape", "close", "back", show=False),
        Binding("ctrl+y", "copy", "copy", show=False),
    ]

    CSS = """
    DetailScreen { background: $background; }
    #detail { width: 100%; height: 100%; border: heavy $accent; background: $surface; padding: 1 2; }
    #detail-title { color: $accent; text-style: bold; height: 1; }
    #detail-body { height: 1fr; margin-top: 1; }
    .detail-content { color: $text; }
    """

    def __init__(self, title: str, content: str) -> None:
        super().__init__()
        self.title = title
        self.content = content or "(no content)"

    def compose(self) -> ComposeResult:
        with Container(id="detail"):
            yield Static(f"{self.title}  ·  Ctrl-Y copy  ·  Esc back", id="detail-title", markup=False)
            with VerticalScroll(id="detail-body"):
                yield Static(self.content, classes="detail-content", markup=False)

    def action_close(self) -> None:
        self.dismiss()

    def action_copy(self) -> None:
        self.app.copy_plain_text(self.content)


class TaskDiffScreen(DetailScreen):
    """Task-diff detail overlay with retained query IDs for callers."""

    CSS = """
    TaskDiffScreen { background: $background; }
    #task-diff { width: 100%; height: 100%; border: heavy $accent; background: $surface; padding: 1 2; }
    #task-diff-title { color: $accent; text-style: bold; height: 1; }
    #task-diff-body { height: 1fr; margin-top: 1; }
    .diff-content { color: $text; }
    """

    def __init__(self, diff: str) -> None:
        super().__init__("DIFF", diff or "(no HANS task changes)")

    def compose(self) -> ComposeResult:
        with Container(id="task-diff"):
            yield Static("DIFF  ·  Ctrl-Y copy  ·  Esc back", id="task-diff-title", markup=False)
            with VerticalScroll(id="task-diff-body"):
                yield Static(self.content, classes="diff-content", markup=False)


class ToolRow(Static):
    """Compact tool row that opens its own retained output when clicked."""

    def __init__(self, call_id: str, text: str = "◉ tool") -> None:
        super().__init__(text, classes="tool", markup=False)
        self.call_id = call_id

    def on_click(self, event: Click) -> None:
        event.stop()
        self.app.open_tool_output(self.call_id)


class ThemeScreen(ModalScreen[None]):
    BINDINGS = [Binding("escape", "close", "back", show=False)]

    CSS = """
    ThemeScreen { background: $background; }
    #theme-selector { width: 100%; height: auto; border: heavy $accent; background: $surface; padding: 1 2; }
    #theme-title { color: $accent; text-style: bold; }
    """

    def compose(self) -> ComposeResult:
        with Container(id="theme-selector"):
            yield Static(
                "THEME  ·  d dark  l light  h high-contrast  t terminal  ·  Esc back",
                id="theme-title",
                markup=False,
            )
            yield Static("Themes are session-only and preserve text and state symbols.", markup=False)

    def on_key(self, event: Key) -> None:
        themes = {"d": "dark", "l": "light", "h": "high-contrast", "t": "terminal"}
        if event.key in themes:
            self.app.apply_theme(themes[event.key])
            self.dismiss()

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
        Binding("ctrl+o", "show_latest_output", "output", show=False),
        Binding("ctrl+y", "copy_visible", "copy", show=False),
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
        self._tool_outputs: dict[str, str] = {}
        self._tool_purposes: dict[str, str] = {}
        self._latest_tool_call_id: str | None = None
        self._todos = TodoList()
        self._theme_name = "dark"
        self._completed_tool_rows: deque[str] = deque()
        self._transcript_rows: deque[Static] = deque()
        self._verification_failed = False
        self._request_active = False
        self._has_task_changes = False
        self._state = "IDLE"
        self._closed = False

    def compose(self) -> ComposeResult:
        yield Static(self._header_text(), id="hans-header", markup=False)
        yield VerticalScroll(id="transcript")
        yield Static(self._state, id="state-line", markup=False)
        yield Static(self._footer_text(), id="footer", markup=False)
        yield TextArea("", id="composer")

    def on_mount(self) -> None:
        self._register_themes()
        self.apply_theme(self._theme_name, announce=False)
        self.query_one("#composer", TextArea).focus()

    def _register_themes(self) -> None:
        palettes = {
            "dark": dict(primary="#64b5f6", secondary="#b39ddb", warning="#ffcc80", error="#ff8a80", success="#81c784", accent="#80cbc4", foreground="#f5f5f5", background="#101418", surface="#1b2128", dark=True),
            "light": dict(primary="#1565c0", secondary="#6a1b9a", warning="#b45309", error="#b91c1c", success="#15803d", accent="#00796b", foreground="#17202a", background="#f8fafc", surface="#ffffff", dark=False),
            "high-contrast": dict(primary="#00ffff", secondary="#ffff00", warning="#ffff00", error="#ff5555", success="#55ff55", accent="#ffffff", foreground="#ffffff", background="#000000", surface="#000000", dark=True),
            "terminal": dict(primary="#ffffff", secondary="#ffffff", warning="#ffffff", error="#ffffff", success="#ffffff", accent="#ffffff", foreground="#ffffff", background="#000000", surface="#000000", dark=True),
        }
        for name, palette in palettes.items():
            self.register_theme(Theme(f"hans-{name}", **palette))

    def apply_theme(self, name: str, *, announce: bool = True) -> bool:
        if name not in THEME_NAMES:
            self._set_state("IDLE", f"theme error: {name}")
            return False
        self.theme = f"hans-{name}"
        self._theme_name = name
        if announce and self.is_mounted:
            self._set_state("IDLE", f"theme {name}")
        return True

    def copy_plain_text(self, text: str) -> bool:
        if not text:
            self._set_state("IDLE", "nothing to copy")
            return False
        try:
            self.copy_to_clipboard(text)
        except Exception:
            self._set_state("IDLE", "clipboard unavailable")
            return False
        self._set_state("IDLE", "copied")
        return True

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        if event.text_area.id != "composer":
            return
        rows = min(self.MAX_COMPOSER_ROWS, max(1, event.text_area.text.count("\n") + 1))
        event.text_area.styles.height = rows + 2

    def on_key(self, event: Key) -> None:
        if event.key == "ctrl+y" and isinstance(self.screen, DetailScreen):
            event.stop()
            event.prevent_default()
            self.screen.action_copy()
            return
        actions = {
            "ctrl+d": self.action_submit_or_exit,
            "ctrl+c": self.action_cancel_active,
            "ctrl+g": self.action_show_task_diff,
            "ctrl+z": self.action_undo_task,
            "ctrl+o": self.action_show_latest_output,
            "ctrl+y": self.action_copy_visible,
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

    def _footer_text(self) -> str:
        return footer_text(
            self._state,
            request_active=self._request_active,
            has_task_changes=self._has_task_changes,
        )

    def _update_footer(self) -> None:
        self.query_one("#footer", Static).update(self._footer_text())

    def _set_state(self, state: str, detail: str = "") -> None:
        shown_detail = self._compact(detail, self.MAX_TOOL_DETAIL_CHARS)
        self._state = state if not shown_detail else f"{state}  ·  {shown_detail}"
        self.query_one("#state-line", Static).update(self._state)
        self._update_footer()

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
                self._tool_outputs.pop(call_id, None)
                if self._latest_tool_call_id == call_id:
                    self._latest_tool_call_id = None
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
                    self._tool_outputs.pop(call_id, None)
                    if self._latest_tool_call_id == call_id:
                        self._latest_tool_call_id = None
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
        transcript = self.query_one("#transcript", VerticalScroll)
        position = transcript.scroll_y
        follow = position >= transcript.max_scroll_y
        widget = ToolRow(call_id)
        await transcript.mount(widget)
        self._transcript_rows.append(widget)
        self._tool_widgets[call_id] = widget
        await self._trim_transcript()
        self._follow_transcript(transcript, follow, position)
        return widget

    async def _trim_completed_tools(self) -> None:
        while len(self._completed_tool_rows) > self.MAX_TOOL_ROWS:
            call_id = self._completed_tool_rows.popleft()
            widget = self._tool_widgets.get(call_id)
            if widget is not None:
                self._forget_row(widget)
                await widget.remove()
            else:
                self._tool_outputs.pop(call_id, None)
            if self._latest_tool_call_id == call_id:
                self._latest_tool_call_id = None

    async def _render_event(self, event: HansEvent) -> None:
        if isinstance(event, UserMessageSubmitted):
            await self._append_transcript(f"YOU\n> {event.message}", "user")
        elif isinstance(event, RequestStarted):
            await self._flush_assistant_deltas()
            self._assistant_text = ""
            self._assistant_widget = None
            self._verification_failed = False
            self._has_task_changes = False
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
            output = display_bounded(event.output, self.MAX_TOOL_OUTPUT_CHARS)
            self._tool_outputs[event.call_id] = output
            self._latest_tool_call_id = event.call_id
            if self._debug_enabled():
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
            self._has_task_changes = task_summary_has_hans_changes(event.summary)
            self._update_footer()
            await self._append_transcript(
                f"CHANGES\n{format_change_summary(event.summary)}", "change"
            )
        elif isinstance(event, TaskDiff):
            self.push_screen(TaskDiffScreen(display_bounded(event.diff, self.MAX_TASK_DIFF_CHARS)))
        elif isinstance(event, TaskUndoSucceeded):
            details = []
            if event.restored_files:
                details.append("restored: " + ", ".join(event.restored_files))
            if event.removed_files:
                details.append("removed: " + ", ".join(event.removed_files))
            feedback = "; ".join(details) if details else "no HANS task changes"
            self._has_task_changes = False
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
        local = handle_local_command(prompt, self._todos)
        if local.handled:
            composer.text = ""
            if local.theme:
                self.apply_theme(local.theme)
            if local.show_theme_selector:
                self.push_screen(ThemeScreen())
            self.run_worker(self._append_transcript(local.text, "change"), exclusive=False)
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

    def action_show_latest_output(self) -> None:
        if self._request_active:
            return
        if self._latest_tool_call_id is None:
            self._set_state("IDLE", "no retained tool output")
            return
        self.open_tool_output(self._latest_tool_call_id)

    def open_tool_output(self, call_id: str) -> None:
        output = self._tool_outputs.get(call_id)
        if output is None:
            self._set_state("IDLE", "tool output is no longer retained")
            return
        name = self._tool_names.get(call_id, "tool")
        self.push_screen(DetailScreen(f"TOOL OUTPUT · {name}", output))

    def action_copy_visible(self) -> None:
        self.copy_plain_text(display_bounded(self._assistant_text, self.MAX_TOOL_OUTPUT_CHARS))

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
            self._update_footer()

    def _close_runtime(self) -> None:
        if not self._closed:
            self.runtime.close()
            self._closed = True


async def run_textual_tui(runtime: Runtime, model: str, workspace: Path) -> None:
    """Run the Textual application without exposing SDK runtime details."""
    await HansTextualApp(runtime, model, workspace).run_async()
