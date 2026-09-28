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
from bolt_next.runtime import HansRuntime
from bolt_next.tui_screen import (
    Editor,
    Transcript,
    footer_text,
    format_change_summary,
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

    def __init__(self, transcript: Transcript | None = None, on_connection=None) -> None:
        self.debug = debug_enabled()
        self.transcript = transcript
        self.on_connection = on_connection
        self.state = "IDLE"
        self.request_active = False
        self.has_task_changes = False
        self._started = False
        self._verification_failed = False
        self._tool_purposes: dict[str, str] = {}

    @staticmethod
    def _compact(text: str, limit: int = 240) -> str:
        text = " ".join(text.split())
        return text if len(text) <= limit else f"{text[:limit]}…"

    @staticmethod
    def _debug_tool_output(output: str) -> str:
        if len(output) <= TOOL_OUTPUT_MAX_CHARS:
            return output
        omitted = len(output) - TOOL_OUTPUT_MAX_CHARS
        return f"{output[:TOOL_OUTPUT_MAX_CHARS]}\n… display truncated ({omitted} characters omitted)"

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
            if self.debug:
                output = self._debug_tool_output(event.output)
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
            diff = (event.diff or "(no HANS task changes)")[:TASK_DIFF_MAX_CHARS]
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
        elif isinstance(event, ConnectionChanged) and self.on_connection is not None:
            self.on_connection(event.connected)
        elif isinstance(event, AssistantMessageComplete):
            return

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


async def serve(read_line, run_turn) -> None:
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


def _workspace() -> Path:
    return Path(os.environ.get("BOLT_WORKSPACE") or Path.cwd()).expanduser()


async def _run_tui() -> None:
    runtime = HansRuntime(os.environ.get("BOLT_WORKSPACE"))
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(format_header(_model_name(), _workspace(), False), flush=True)
        print(FOOTER, flush=True)

        async def run_turn(prompt: str) -> None:
            await _run_turn(runtime, prompt)

        try:
            await serve(input, run_turn)
        finally:
            runtime.close()
        return
    if os.environ.get("HANS_TUI", "").strip().lower() == "curses":
        await _run_curses(runtime)
        return
    from bolt_next.textual_tui import run_textual_tui

    await run_textual_tui(runtime, _model_name(), _workspace())


async def _run_curses(runtime: HansRuntime) -> None:
    import curses

    transcript = Transcript()
    editor = Editor()
    state = {"connected": False, "task": None, "cancel": False}
    state["display"] = _Display(transcript, lambda connected: state.__setitem__("connected", connected))

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
        header = format_header(_model_name(), _workspace(), state["connected"]).splitlines()
        for row, line in enumerate(header[:2]):
            stdscr.addnstr(row, 0, line, width - 1)
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

    async def loop(stdscr) -> None:
        try:
            curses.curs_set(1)
        except curses.error:
            pass
        stdscr.keypad(True)
        stdscr.nodelay(True)
        decoder = _InputDecoder()
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
            for name in decoder.feed(key):
                if state["task"] is not None:
                    if name == "ctrl-c":
                        request_cancel()
                    elif name == "ctrl-q":
                        request_cancel()
                        return
                    continue
                if name == "ctrl-g":
                    state["display"].event(runtime.task_diff(max_chars=TASK_DIFF_MAX_CHARS))
                    continue
                if name == "ctrl-z":
                    state["display"].event(runtime.undo_task())
                    continue
                submitted = editor.on_key(name)
                if submitted is None:
                    continue
                if submitted == "" or is_exit_command(submitted):
                    return
                state["task"] = asyncio.create_task(run_prompt(submitted))

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
    if key in {26, "\x1a"}:
        return "ctrl-z"
    if key in {17, "\x11"}:
        return "ctrl-q"
    if key in {"\n", "\r", 10}:
        return "enter"
    if key in {"\x7f", "\b", 127, 263}:
        return "backspace"
    if isinstance(key, str) and key.isprintable():
        return "char:" + key
    return None


def run_tui() -> None:
    try:
        asyncio.run(_run_tui())
    except KeyboardInterrupt:
        print("\ninterrupted")
