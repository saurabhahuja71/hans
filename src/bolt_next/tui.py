from __future__ import annotations

import asyncio
import os
import signal
import sys
import termios
from pathlib import Path

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
    ToolCompleted,
    ToolOutput,
    ToolStarted,
    UserMessageSubmitted,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)
from bolt_next.runtime import HansRuntime
from bolt_next.tui_screen import (
    Editor,
    LocalControl,
    TodoList,
    Transcript,
    copy_osc52,
    display_bounded,
    footer_text,
    format_change_summary,
    format_model_changed,
    format_model_status,
    format_reasoning_mode_changed,
    format_reasoning_mode_status,
    handle_local_command,
    is_exit_command,
    layout_rows,
    task_summary_has_hans_changes,
    visible_transcript,
)


FOOTER = footer_text("IDLE", request_active=False, has_task_changes=False)
TASK_DIFF_MAX_CHARS = 4_000
TOOL_OUTPUT_MAX_CHARS = 4_000


def debug_enabled() -> bool:
    return os.environ.get("HANS_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def workspace_label(workspace: Path) -> str:
    try:
        return "~/" + str(workspace.relative_to(Path.home()))
    except ValueError:
        return str(workspace)


def format_header(model: str, workspace: Path, connected: bool = False) -> str:
    state = "● connected" if connected else "○ disconnected"
    shown = workspace_label(workspace)
    return "\n".join(
        [
            f"HANS  model: {model}  connection: {state}",
            f"workspace: {shown}",
        ]
    )


def format_tool_call(name: str, detail: str) -> str:
    return f"  ◇ {name}  {detail}".rstrip()


def format_tool_result(name: str, success: bool, exit_code: int | None = None) -> str:
    mark = "✓" if success else "✗"
    if exit_code is not None:
        return f"  {mark} {name}  exit {exit_code}"
    return f"  {mark} {name}"


def detail_window(content: str, width: int, height: int, offset: int = 0) -> list[str]:
    """Return a bounded, width-aware slice for the curses detail overlay."""
    room = max(1, width)
    rows: list[str] = []
    for line in content.splitlines() or [""]:
        rows.extend(line[index : index + room] for index in range(0, max(1, len(line)), room))
    window_height = max(1, height)
    start = min(max(0, offset), max(0, len(rows) - window_height))
    return rows[start : start + window_height]


def turn_error_message(
    category: str | BaseException,
    message: str | None = None,
    *,
    debug: bool | None = None,
    debug_detail: str | None = None,
) -> str:
    if isinstance(category, BaseException):
        message = str(category).strip().splitlines()[0] if str(category).strip() else "request failed"
        debug_detail = debug_detail or message
        lowered = message.lower()
        category = "context" if "context" in lowered or "exceed_context" in lowered else "runtime"
    message = message or "request failed"
    if category == "configuration":
        rendered = f"\n✗ configuration failed: {message}\n"
    elif category == "authentication":
        rendered = f"\n✗ authentication failed: {message}\n"
    elif category == "context":
        rendered = "\n✗ context budget exceeded; the session is still open. Request a smaller file range.\n"
    elif category == "connection":
        rendered = f"\n✗ connection failed: {message}\n"
    elif category == "model":
        rendered = f"\n✗ model request failed: {message}\n"
    else:
        rendered = f"\n✗ {message}\n"
    if debug if debug is not None else debug_enabled():
        rendered += f"[{category}] {debug_detail or message}\n"
    return rendered


class _Display:
    """Renders HANS semantic events into a transcript or plain stdout."""

    def __init__(self, transcript: Transcript | None = None, on_connection=None, on_model=None) -> None:
        self.debug = debug_enabled()
        self.transcript = transcript
        self.on_connection = on_connection
        self.on_model = on_model
        self.state = "IDLE"
        self.request_active = False
        self.has_task_changes = False
        self._started = False
        self._verification_failed = False
        self._tool_purposes: dict[str, str] = {}
        self.tool_outputs: dict[str, str] = {}
        self.latest_tool_call_id: str | None = None
        self.max_tool_outputs = 100
        self.max_transcript_pieces = 300

    @staticmethod
    def _compact(text: str, limit: int = 240) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else f"{text[:limit]}…"

    @staticmethod
    def _debug_tool_output(output: str) -> str:
        return display_bounded(output, TOOL_OUTPUT_MAX_CHARS)

    def latest_tool_output(self) -> str | None:
        if self.latest_tool_call_id is None:
            return None
        return self.tool_outputs.get(self.latest_tool_call_id)

    def _trim_presentation(self) -> None:
        while len(self.tool_outputs) > self.max_tool_outputs:
            call_id = next(iter(self.tool_outputs))
            self.tool_outputs.pop(call_id, None)
            if self.latest_tool_call_id == call_id:
                self.latest_tool_call_id = None
        if self.transcript is not None and len(self.transcript.pieces) > self.max_transcript_pieces:
            del self.transcript.pieces[: len(self.transcript.pieces) - self.max_transcript_pieces]

    def _set_state(self, state: str, detail: str = "") -> None:
        self.state = state if not detail else f"{state} · {self._compact(detail)}"
        if self.transcript is not None:
            self.transcript.stage(self.state)

    def _tool_stage(self, name: str, purpose: str = "inspect") -> str:
        if name in {"list_directory", "search_files", "read_file"}:
            return "INVESTIGATING"
        if name in {"replace_in_file", "write_file"}:
            return "CORRECTING" if self._verification_failed else "EDITING"
        if name == "run_command":
            return "VERIFYING" if purpose == "verify" else "INVESTIGATING"
        return "EDITING"

    def event(self, event) -> None:
        if isinstance(event, UserMessageSubmitted):
            if self.transcript is not None:
                self.transcript.user(event.message)
            else:
                print(_plain_user(event.message), flush=True)
        elif isinstance(event, RequestStarted):
            self.request_active = True
            self.has_task_changes = False
            self._set_state("INVESTIGATING")
        elif isinstance(event, AssistantMessageDelta):
            self._text(event.delta)
        elif isinstance(event, ToolStarted):
            label = f"{event.name}  {self._compact(event.detail)}".rstrip()
            self._tool_purposes[event.call_id] = event.purpose
            if self.transcript is not None:
                self.transcript.tool_started(label)
                self._set_state(self._tool_stage(event.name, event.purpose))
                if self.debug:
                    self.transcript.debug(
                        f"[tool_started] call_id={event.call_id} name={event.name} detail={event.detail}"
                    )
            elif self.debug:
                print(
                    f"\n[tool_started] call_id={event.call_id} name={event.name} detail={event.detail}",
                    flush=True,
                )
            else:
                print("\n" + format_tool_call(event.name, self._compact(event.detail)), flush=True)
        elif isinstance(event, ToolOutput):
            output = self._debug_tool_output(event.output)
            self.tool_outputs[event.call_id] = output
            self.latest_tool_call_id = event.call_id
            if self.debug:
                if self.transcript is not None:
                    self.transcript.debug(f"[tool_output] call_id={event.call_id}\n{output}")
                else:
                    print(f"\n[tool_output] call_id={event.call_id}\n{output}", flush=True)
        elif isinstance(event, ToolCompleted):
            label = event.detail or event.name
            purpose = self._tool_purposes.pop(event.call_id, "inspect")
            if self.transcript is not None:
                self.transcript.tool_finished(label, ok=event.success)
            elif not self.debug:
                print(format_tool_result(event.name, event.success, event.exit_code), flush=True)
            if event.success:
                self._set_state(self._tool_stage(event.name, purpose))
            else:
                self._set_state("FAILED", f"tool {event.name}")
        elif isinstance(event, VerificationStarted):
            self._set_state("VERIFYING", event.command)
        elif isinstance(event, VerificationPassed):
            self._verification_failed = False
            if self.transcript is not None:
                self.transcript.verification(event.evidence.command, ok=True)
            self._set_state("VERIFYING", f"passed · {event.evidence.command}")
        elif isinstance(event, VerificationFailed):
            self._verification_failed = True
            if self.transcript is not None:
                self.transcript.verification(event.evidence.command, ok=False)
            self._set_state("CORRECTING", event.evidence.command)
        elif isinstance(event, RequestCompleted):
            self.request_active = False
            if self.transcript is not None:
                self.transcript.completed(event.evidence)
            self._set_state("FAILED" if event.evidence is not None and not event.evidence.success else "COMPLETE")
            if self.transcript is None and self._started:
                print(flush=True)
        elif isinstance(event, RequestCancelled):
            self.request_active = False
            if self.transcript is not None:
                self.transcript.cancelled()
            self._set_state("CANCELLED")
            if self.transcript is None:
                print("\ninterrupted", flush=True)
        elif isinstance(event, RequestFailed):
            self.request_active = False
            titles = {
                "configuration": "configuration failed",
                "authentication": "authentication failed",
                "context": "context limit exceeded",
                "connection": "connection failed",
                "model": "model request failed",
                "tool": "tool failed",
                "runtime": "runtime failed",
            }
            title = titles.get(event.category, "model request failed")
            self._set_state("FAILED", title)
            if self.transcript is not None:
                self.transcript.error(title, event.message[:160])
                if self.debug:
                    self.transcript.debug(f"[{event.category}] {event.debug_message or event.message}")
            else:
                print(
                    turn_error_message(
                        event.category, event.message, debug_detail=event.debug_message
                    ),
                    flush=True,
                )
        elif isinstance(event, TaskChangeSummary):
            self.has_task_changes = task_summary_has_hans_changes(event.summary)
            summary = format_change_summary(event.summary)
            if self.transcript is not None:
                self.transcript.change(summary)
            else:
                print(f"\nchanges:\n{summary}", flush=True)
        elif isinstance(event, TaskDiff):
            diff = display_bounded(event.diff or "(no HANS task changes)", TASK_DIFF_MAX_CHARS)
            if self.transcript is not None:
                self.transcript.diff(diff)
            else:
                print(f"\ntask diff\n{diff}", flush=True)
        elif isinstance(event, TaskUndoSucceeded):
            details = []
            if event.restored_files:
                details.append("restored: " + ", ".join(event.restored_files))
            if event.removed_files:
                details.append("removed: " + ", ".join(event.removed_files))
            label = "Undo completed" + (f" · {'; '.join(details)}" if details else " · no HANS task changes")
            if self.transcript is not None:
                self.transcript.change(f"✓ {label}", title="UNDO")
            else:
                print(f"\n✓ {label}", flush=True)
            self.has_task_changes = False
            self._set_state("IDLE", "undo complete")
        elif isinstance(event, TaskUndoRefused):
            conflicts = ", ".join(event.conflicting_files) or "task changes"
            if self.transcript is not None:
                self.transcript.error("Undo refused", f"Conflicts: {conflicts}")
            else:
                print(f"\n✗ undo refused: conflicts: {conflicts}", flush=True)
            self._set_state("IDLE", "undo refused")
        elif isinstance(event, RuntimeControlStatus):
            text = "\n".join(
                ["Permissions"]
                + [
                    f"{name:<10} {'✓ allow' if allowed else '✗ deny'}"
                    for name, allowed in (
                        ("read", event.read_allowed),
                        ("write", event.write_allowed),
                        ("execute", event.execute_allowed),
                    )
                ]
            )
            if self.transcript is not None:
                self.transcript.change(text, title="LOCAL")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, ModelStatus):
            if self.on_model is not None:
                self.on_model(event.model)
            text = format_model_status(event.model, event.current_reasoning_mode, event.models)
            if self.transcript is not None:
                self.transcript.change(text, title="LOCAL")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, ModelChanged):
            if self.on_model is not None:
                self.on_model(event.model)
            text = format_model_changed(
                event.previous_model_id,
                event.model,
                new_session_started=event.new_session_started,
                reasoning_reset=event.reasoning_reset,
            )
            if self.transcript is not None:
                self.transcript.change(text, title="LOCAL")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, ReasoningModeStatus):
            text = format_reasoning_mode_status(event.mode, event.available_modes, event.none_semantics)
            if self.transcript is not None:
                self.transcript.change(text, title="LOCAL")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, ReasoningModeChanged):
            text = format_reasoning_mode_changed(event.mode)
            if self.transcript is not None:
                self.transcript.change(text, title="LOCAL")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, PermissionPolicyChanged):
            value = "allow" if event.allowed else "deny"
            text = f"✓ {event.category} permission set to {value}."
            if self.transcript is not None:
                self.transcript.change(text, title="PERMISSIONS")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, SessionCleared):
            text = "✓ Conversation history cleared.\nWorkspace and local HANS state preserved."
            if self.transcript is not None:
                self.transcript.change(text, title="SESSION")
            else:
                print(f"\n{text}", flush=True)
        elif isinstance(event, RuntimeControlRejected):
            if self.transcript is not None:
                self.transcript.error(event.message)
            else:
                print(f"\n✗ {event.message}", flush=True)
        elif isinstance(event, ConnectionChanged) and self.on_connection is not None:
            self.on_connection(event.connected)
        elif isinstance(event, AssistantMessageComplete):
            return
        self._trim_presentation()

    def _text(self, delta: str) -> None:
        if not delta:
            return
        if self.transcript is not None:
            self.transcript.stream(delta)
            return
        if not self._started:
            print(flush=True)
            self._started = True
        print(delta, end="", flush=True)


def read_user_message(read_line) -> str | None:
    """Read one non-interactive prompt buffer until EOF.

    EOF before any line means the user is done. Embedded newlines are preserved.
    A pasted or piped block is one message because submission happens only at EOF,
    not at each newline.
    """
    lines: list[str] = []
    while True:
        try:
            line = read_line("\n> " if not lines else "… ")
        except EOFError:
            if not lines:
                return None
            return "\n".join(lines)
        lines.append(line)


async def _run_turn(runtime: HansRuntime, prompt: str, display: _Display | None = None) -> None:
    shown = display or _Display()
    async for event in runtime.submit(prompt):
        shown.event(event)


async def _dispatch_control(runtime: HansRuntime, control: LocalControl, display: _Display) -> None:
    if control.kind == "permissions_status":
        event = runtime.get_control_status()
    elif control.kind == "set_permission":
        event = runtime.set_permission(control.category or "", bool(control.allowed))
    elif control.kind == "clear_session":
        event = await runtime.clear_session_history()
    elif control.kind == "model_status":
        event = runtime.get_model_status()
    elif control.kind == "select_model":
        event = runtime.select_model(control.model_id or "")
    elif control.kind == "reasoning_mode_status":
        event = runtime.get_reasoning_mode_status()
    elif control.kind == "set_reasoning_mode":
        event = runtime.set_reasoning_mode(control.mode or "")
    else:
        return
    if event is not None:
        display.event(event)


async def serve(read_line, run_turn, handle_local=None) -> None:
    """Interactive loop. Ctrl-C returns to the prompt. Ctrl-D on empty input exits."""
    while True:
        try:
            prompt = read_user_message(read_line)
        except KeyboardInterrupt:
            print(flush=True)
            continue
        if prompt is None or is_exit_command(prompt):
            print(flush=True)
            return
        if not prompt.strip():
            continue
        if handle_local is not None and await handle_local(prompt):
            continue
        if debug_enabled():
            print("[user_turn]", flush=True)
        try:
            await run_turn(prompt)
        except KeyboardInterrupt:
            print("\ninterrupted\n", flush=True)


def _plain_user(text: str) -> str:
    rows = text.split("\n")
    return "\n".join(("> " if index == 0 else "  ") + row for index, row in enumerate(rows)) + "\n"


def _model_name() -> str:
    return os.environ.get("BOLT_MODEL", "qwen3.6-27b")


def _startup_model_name(runtime: HansRuntime) -> str:
    try:
        model = runtime.get_model_status().model
    except Exception:
        return _model_name()
    return str(getattr(model, "display_name", None) or getattr(model, "id", None) or _model_name())


def _workspace() -> Path:
    return Path(os.environ.get("BOLT_WORKSPACE") or Path.cwd()).expanduser()


async def _run_tui() -> None:
    runtime = HansRuntime(os.environ.get("BOLT_WORKSPACE"))
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        header_state = {"model": _startup_model_name(runtime), "connected": False}

        def render_header() -> None:
            print(
                format_header(header_state["model"], _workspace(), header_state["connected"]),
                flush=True,
            )

        def update_model(model: object) -> None:
            header_state["model"] = str(
                getattr(model, "display_name", None) or getattr(model, "id", None) or header_state["model"]
            )
            render_header()

        def update_connection(connected: bool) -> None:
            header_state["connected"] = connected
            render_header()

        render_header()
        print(FOOTER, flush=True)

        display = _Display(on_connection=update_connection, on_model=update_model)
        todos = TodoList()

        async def run_turn(prompt: str) -> None:
            await _run_turn(runtime, prompt, display)

        async def handle_local(prompt: str) -> bool:
            local = handle_local_command(prompt, todos)
            if not local.handled:
                return False
            if local.control is not None:
                await _dispatch_control(runtime, local.control, display)
            elif local.text:
                print(f"\n{local.text}", flush=True)
            return True

        try:
            await serve(input, run_turn, handle_local)
        finally:
            runtime.close()
        return
    if os.environ.get("HANS_TUI", "").strip().lower() == "curses":
        await _run_curses(runtime)
        return
    from bolt_next.textual_tui import run_textual_tui

    await run_textual_tui(runtime, _startup_model_name(runtime), _workspace())


async def _run_curses(runtime: HansRuntime) -> None:
    import curses

    transcript = Transcript()
    editor = Editor()
    state = {
        "connected": False,
        "model": _startup_model_name(runtime),
        "task": None,
        "cancel": False,
        "detail": None,
        "theme": "terminal",
        "todos": TodoList(),
    }
    state["display"] = _Display(
        transcript,
        lambda connected: state.__setitem__("connected", connected),
        lambda model: state.__setitem__(
            "model", str(getattr(model, "display_name", None) or getattr(model, "id", None) or state["model"])
        ),
    )

    def request_cancel(*_args) -> None:
        state["cancel"] = True
        state["display"]._set_state("CANCELLED", "cancellation requested")
        runtime.cancel_active()

    previous_int = signal.signal(signal.SIGINT, request_cancel)
    previous_stop = signal.signal(signal.SIGTSTP, signal.SIG_IGN)

    def draw(stdscr) -> None:
        height, width = stdscr.getmaxyx()
        if height < 8 or width < 20:
            return
        stdscr.erase()
        detail = state["detail"]
        if detail is not None:
            title, content, offset = detail
            stdscr.addnstr(0, 0, f"{title}  ·  Ctrl-Y copy  ·  Esc back", width - 1, curses.A_BOLD)
            stdscr.hline(1, 0, curses.ACS_HLINE, width - 1)
            for row, line in enumerate(detail_window(content, width - 1, height - 4, offset), start=2):
                stdscr.addnstr(row, 0, line, width - 1)
            stdscr.hline(height - 2, 0, curses.ACS_HLINE, width - 1)
            stdscr.addnstr(height - 1, 0, "Up/Down/Page scroll · Esc back", width - 1)
            stdscr.refresh()
            return
        header = format_header(state["model"], _workspace(), state["connected"]).splitlines()
        header_attr = curses.A_REVERSE if state["theme"] == "high-contrast" else curses.A_NORMAL
        for row, line in enumerate(header[:2]):
            stdscr.addnstr(row, 0, line, width - 1, header_attr)
        stdscr.hline(2, 0, curses.ACS_HLINE, width - 1)
        conversation, _editor_height = layout_rows(height, len(editor.lines))
        body = visible_transcript(transcript.render(width - 1), conversation)
        for offset, line in enumerate(body):
            stdscr.addnstr(3 + offset, 0, line, width - 1)
        footer_at = 3 + conversation
        stdscr.hline(footer_at, 0, curses.ACS_HLINE, width - 1)
        display = state["display"]
        status_footer = footer_text(
            display.state,
            request_active=display.request_active,
            has_task_changes=display.has_task_changes,
        )
        stdscr.addnstr(footer_at + 1, 0, status_footer, width - 1)
        for offset, line in enumerate(editor.display_lines()):
            row = footer_at + 2 + offset
            if row < height:
                stdscr.addnstr(row, 0, line, width - 1)
        stdscr.refresh()

    async def run_prompt(prompt: str) -> None:
        try:
            await _run_turn(runtime, prompt, state["display"])
        except asyncio.CancelledError:
            runtime.cancel_active()
            raise

    async def dispatch_local(local) -> None:
        if local.control is not None:
            await _dispatch_control(runtime, local.control, state["display"])
        elif local.theme:
            state["theme"] = "high-contrast" if local.theme == "high-contrast" else "terminal"
        if local.text:
            transcript.change(local.text, title="LOCAL")
            state["display"]._trim_presentation()

    def open_detail(title: str, content: str | None) -> None:
        if content is None:
            state["display"]._set_state("IDLE", "no retained tool output")
            return
        state["detail"] = (title, content, 0)

    def detail_key(name: str) -> bool:
        detail = state["detail"]
        if detail is None:
            return False
        title, content, offset = detail
        if name == "escape":
            state["detail"] = None
        elif name == "ctrl-y":
            _ok, message = copy_osc52(content, max_chars=None)
            state["display"]._set_state("IDLE", message.lower())
        elif name == "up":
            state["detail"] = (title, content, max(0, offset - 1))
        elif name == "down":
            state["detail"] = (title, content, offset + 1)
        elif name == "pageup":
            state["detail"] = (title, content, max(0, offset - 10))
        elif name == "pagedown":
            state["detail"] = (title, content, offset + 10)
        return True

    async def handle_input(name: str) -> bool:
        if state["detail"] is not None:
            if name == "ctrl-q":
                return True
            detail_key(name)
            return False
        if state["task"] is not None:
            if name == "ctrl-c":
                request_cancel()
                return False
            if name == "ctrl-q":
                request_cancel()
                return True
            submitted = editor.on_key(name)
            if submitted is None:
                return False
            local = handle_local_command(submitted, state["todos"])
            if local.handled:
                await dispatch_local(local)
            return False
        if name == "ctrl-g":
            diff = runtime.task_diff(max_chars=TASK_DIFF_MAX_CHARS)
            open_detail("DIFF", display_bounded(diff.diff, TASK_DIFF_MAX_CHARS))
            return False
        if name == "ctrl-o":
            open_detail("TOOL OUTPUT", state["display"].latest_tool_output())
            return False
        if name == "ctrl-y":
            assistant = next(
                (piece.text for piece in reversed(transcript.pieces) if piece.kind == "assistant"),
                "",
            )
            if not assistant:
                state["display"]._set_state("IDLE", "nothing to copy")
            else:
                _ok, message = copy_osc52(
                    display_bounded(assistant, TOOL_OUTPUT_MAX_CHARS), max_chars=None
                )
                state["display"]._set_state("IDLE", message.lower())
            return False
        if name == "ctrl-z":
            state["display"].event(runtime.undo_task())
            return False
        submitted = editor.on_key(name)
        if submitted is None:
            return False
        if submitted == "" or is_exit_command(submitted):
            return True
        local = handle_local_command(submitted, state["todos"])
        if local.handled:
            await dispatch_local(local)
            return False
        state["task"] = asyncio.create_task(run_prompt(submitted))
        return False

    async def loop(stdscr) -> None:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.nodelay(True)
        decoder = _InputDecoder()
        special_keys = {
            curses.KEY_UP: "up",
            curses.KEY_DOWN: "down",
            curses.KEY_PPAGE: "pageup",
            curses.KEY_NPAGE: "pagedown",
        }
        while True:
            try:
                draw(stdscr)
                if state["cancel"] and state["task"] is None:
                    editor.clear()
                    state["cancel"] = False
                if state["task"] is not None and state["task"].done():
                    state["task"] = None
                try:
                    key = stdscr.get_wch()
                except curses.error:
                    for name in decoder.flush():
                        if await handle_input(name):
                            return
                    await asyncio.sleep(0.04)
                    continue
            except KeyboardInterrupt:
                request_cancel()
                if state["task"] is None:
                    editor.clear()
                state["cancel"] = False
                continue
            if key == curses.KEY_RESIZE:
                continue
            names = [special_keys[key]] if key in special_keys else decoder.feed(key)
            for name in names:
                if await handle_input(name):
                    return

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    tty_attr = termios.tcgetattr(sys.stdin)
    raw_attr = termios.tcgetattr(sys.stdin)
    raw_attr[0] = raw_attr[0] & ~(termios.IXON | termios.IXOFF)
    raw_attr[3] = raw_attr[3] & ~termios.ISIG
    termios.tcsetattr(sys.stdin, termios.TCSANOW, raw_attr)
    _set_enhanced_input(True)
    try:
        await loop(stdscr)
    finally:
        _set_enhanced_input(False)
        termios.tcsetattr(sys.stdin, termios.TCSANOW, tty_attr)
        curses.nocbreak()
        curses.echo()
        curses.endwin()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTSTP, previous_stop)
        runtime.close()


class _InputDecoder:
    _PASTE_START = "\x1b[200~"
    _PASTE_END = "\x1b[201~"
    _SHIFT_ENTER = ("\x1b[13;2u", "\x1b[27;2;13~", "\x1b\r")

    def __init__(self) -> None:
        self._pending = ""
        self._pasting = False

    def feed(self, key) -> list[str]:
        if not isinstance(key, str):
            name = _key_name(key)
            return [] if name is None else [name]
        self._pending += key
        events: list[str] = []
        while self._pending:
            if self._pasting:
                end = self._pending.find(self._PASTE_END)
                if end < 0:
                    keep = len(self._PASTE_END) - 1
                    if len(self._pending) <= keep:
                        break
                    events.append("char:" + self._pending[:-keep])
                    self._pending = self._pending[-keep:]
                    break
                if end:
                    events.append("char:" + self._pending[:end])
                self._pending = self._pending[end + len(self._PASTE_END) :]
                self._pasting = False
                continue
            if self._pending.startswith(self._PASTE_START):
                self._pending = self._pending[len(self._PASTE_START) :]
                self._pasting = True
                continue
            shift = next((value for value in self._SHIFT_ENTER if self._pending.startswith(value)), None)
            if shift is not None:
                self._pending = self._pending[len(shift) :]
                events.append("shift-enter")
                continue
            protocols = (self._PASTE_START, *self._SHIFT_ENTER)
            if any(value.startswith(self._pending) for value in protocols):
                break
            if self._pending.startswith("\x1b["):
                if "@" <= self._pending[-1] <= "~":
                    self._pending = ""
                    continue
                break
            name = _key_name(self._pending[0])
            self._pending = self._pending[1:]
            if name is not None:
                events.append(name)
        return events

    def flush(self) -> list[str]:
        if self._pending == "\x1b":
            self._pending = ""
            return ["escape"]
        return []


def _set_enhanced_input(enabled: bool) -> None:
    sequence = "\x1b[?2004h\x1b[>1u\x1b[>4;2m" if enabled else "\x1b[?2004l\x1b[<u\x1b[>4;0m"
    os.write(sys.stdout.fileno(), sequence.encode())


def _key_name(key) -> str | None:
    if key in {3, "\x03"}:
        return "ctrl-c"
    if key in {4, "\x04"}:
        return "ctrl-d"
    if key in {7, "\x07"}:
        return "ctrl-g"
    if key in {15, "\x0f"}:
        return "ctrl-o"
    if key in {17, "\x11"}:
        return "ctrl-q"
    if key in {25, "\x19"}:
        return "ctrl-y"
    if key in {26, "\x1a"}:
        return "ctrl-z"
    if key in {27, "\x1b"}:
        return "escape"
    if key in {"\n", "\r", 10}:
        return "enter"
    if key in {"\x7f", "\b", 127, 263}:
        return "backspace"
    if key in {259, 258, 339, 338}:
        return {259: "up", 258: "down", 339: "pageup", 338: "pagedown"}[key]
    if isinstance(key, str) and key.isprintable():
        return "char:" + key
    return None


def run_tui() -> None:
    try:
        asyncio.run(_run_tui())
    except KeyboardInterrupt:
        print("\ninterrupted")
