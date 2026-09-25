from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import termios
from pathlib import Path
from urllib.parse import urlparse

from agents import Runner, SQLiteSession, set_tracing_disabled
from agents.run_config import RunConfig

from bolt_next.agent import create_agent
from bolt_next.context_budget import fit_model_input
from bolt_next.tui_screen import Editor, Transcript, is_exit_command, layout_rows, visible_transcript


FOOTER = "Enter send · Ctrl-C cancel · Ctrl-Q exit"


def debug_enabled() -> bool:
    return os.environ.get("HANS_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def workspace_label(workspace: Path) -> str:
    try:
        return "~/" + str(workspace.relative_to(Path.home()))
    except ValueError:
        return str(workspace)


def format_header(model: str, workspace: Path, connected: bool = True) -> str:
    state = "● connected" if connected else "○ disconnected"
    shown = workspace_label(workspace)
    return "\n".join(
        [
            f"HANS  model: {model}  connection: {state}",
            f"workspace: {shown}",
        ]
    )


def tool_detail(name: str | None, arguments) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    if name == "read_file":
        detail = str(arguments.get("path") or "")
        start = arguments.get("start_line") or 0
        end = arguments.get("end_line") or 0
        if start and end:
            detail = f"{detail}:{start}-{end}"
        elif start and int(start) > 1:
            detail = f"{detail}:{start}"
        return detail
    if name == "write_file":
        return str(arguments.get("path") or "")
    if name == "run_command":
        return str(arguments.get("command") or "")
    return ""


def format_tool_call(name: str | None, arguments) -> str:
    detail = tool_detail(name, arguments)
    return f"  ◇ {name}  {detail}".rstrip()


def format_tool_result(name: str | None, output: str) -> str:
    code = None
    failed = output.startswith("Error:")
    for line in output.splitlines():
        if line.startswith("exit_code="):
            code = line.split("=", 1)[1].strip()
            failed = code != "0"
    mark = "✗" if failed else "✓"
    if code is not None:
        return f"  {mark} {name}  exit {code}"
    if failed:
        first = output.splitlines()[0][:120]
        return f"  {mark} {name}  {first}"
    return f"  {mark} {name}"


def turn_error_message(exc: BaseException, *, debug: bool | None = None) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    if "context" in text.lower() or "exceed_context" in text:
        message = "context budget exceeded; the session is still open. Request a smaller file range."
    elif "connection" in text.lower() or "tunnel" in text.lower():
        message = f"connection failed: {text}"
    else:
        message = text
    rendered = f"\n✗ {message}\n"
    if debug if debug is not None else debug_enabled():
        rendered += f"{exc.__class__.__name__}: {exc}\n"
    return rendered


def run_config() -> RunConfig:
    return RunConfig(call_model_input_filter=fit_model_input, tracing_disabled=True)


class _Display:
    """Renders one SDK stream into either the transcript or, for pipes, stdout."""

    def __init__(self, transcript: Transcript | None = None) -> None:
        self.debug = debug_enabled()
        self.transcript = transcript
        self._last_name: str | None = None
        self._last_detail = ""
        self._started = False

    def event(self, event) -> None:
        if event.type == "raw_response_event":
            data = event.data
            if getattr(data, "type", None) in {"response.output_text.delta", "output_text.delta"}:
                self._text(getattr(data, "delta", "") or "")
            return
        if event.type != "run_item_stream_event":
            return
        if event.name == "tool_called":
            item = event.item
            raw = getattr(item, "raw_item", None)
            arguments = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
            self._last_name = getattr(item, "tool_name", None)
            self._last_detail = tool_detail(self._last_name, arguments)
            label = f"{self._last_name}  {self._last_detail}".rstrip()
            if self.transcript is not None:
                self.transcript.tool_started(label)
                if self.debug:
                    self.transcript.debug(f"[tool_called] name={self._last_name} arguments={arguments}")
            elif self.debug:
                print(f"\n[tool_called] name={self._last_name} arguments={arguments}", flush=True)
            else:
                print("\n" + format_tool_call(self._last_name, arguments), flush=True)
        elif event.name == "tool_output":
            output = str(event.item.output)
            failed = output.startswith("Error:")
            for line in output.splitlines():
                if line.startswith("exit_code="):
                    failed = line.split("=", 1)[1].strip() != "0"
            label = self._last_detail or self._last_name or "tool"
            if self.transcript is not None:
                self.transcript.tool_finished(label, ok=not failed)
                if self.debug:
                    self.transcript.debug(f"[tool_output]\n{output}")
            elif self.debug:
                print(f"\n[tool_output]\n{output}", flush=True)
            else:
                print(format_tool_result(self._last_name, output), flush=True)

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
    """Read one prompt. Enter inserts a line; Ctrl-D submits the whole buffer.

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


async def _run_turn(agent, session: SQLiteSession, prompt: str, display: _Display | None = None) -> None:
    result = Runner.run_streamed(agent, prompt, session=session, run_config=run_config())
    shown = display or _Display()
    try:
        async for event in result.stream_events():
            shown.event(event)
        if display is None:
            print(flush=True)
        elif display.transcript is not None:
            display.transcript.finish()
    except asyncio.CancelledError:
        result.cancel()
        if display is not None and display.transcript is not None:
            display.transcript.cancelled()
        raise
    except KeyboardInterrupt:
        result.cancel()
        raise


async def serve(read_line, run_turn) -> None:
    """Interactive loop. Ctrl-C returns to the prompt. Ctrl-D on empty input exits."""
    while True:
        try:
            prompt = read_user_message(read_line)
        except KeyboardInterrupt:
            print(flush=True)
            continue
        if prompt is None:
            print(flush=True)
            return
        if is_exit_command(prompt):
            return
        if not prompt.strip():
            continue
        if debug_enabled():
            print("[user_turn]", flush=True)
        print(_plain_user(prompt), flush=True)
        try:
            await run_turn(prompt)
        except KeyboardInterrupt:
            print("\ninterrupted\n", flush=True)
        except Exception as exc:
            print(turn_error_message(exc), flush=True)


def _plain_user(text: str) -> str:
    rows = text.split("\n")
    return "\n".join(("> " if index == 0 else "  ") + row for index, row in enumerate(rows)) + "\n"


def _model_name() -> str:
    return os.environ.get("BOLT_MODEL", "qwen3.6-27b")


def _workspace() -> Path:
    return Path(os.environ.get("BOLT_WORKSPACE") or Path.cwd()).expanduser()


async def _run_tui() -> None:
    set_tracing_disabled(True)
    connected = bool(urlparse(os.environ.get("BOLT_MODEL_BASE_URL", "")).hostname)
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        print(format_header(_model_name(), _workspace(), connected), flush=True)
        print(FOOTER, flush=True)
        agent = create_agent(os.environ.get("BOLT_WORKSPACE"))
        session = SQLiteSession("hans-tui")

        async def run_turn(prompt: str) -> None:
            await _run_turn(agent, session, prompt)

        try:
            await serve(input, run_turn)
        finally:
            session.close()
        return
    await _run_curses(connected)


async def _run_curses(connected: bool) -> None:
    import curses

    agent = create_agent(os.environ.get("BOLT_WORKSPACE"))
    session = SQLiteSession("hans-tui")
    transcript = Transcript()
    editor = Editor()
    state = {"connected": connected, "task": None, "cancel": False}

    def request_cancel(*_args) -> None:
        state["cancel"] = True
        task = state["task"]
        if task is not None and not task.done():
            task.cancel()

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
        conversation, editor_height = layout_rows(height, len(editor.lines))
        body = visible_transcript(transcript.render(width - 1), conversation)
        for offset, line in enumerate(body):
            stdscr.addnstr(3 + offset, 0, line, width - 1)
        footer_at = 3 + conversation
        stdscr.hline(footer_at, 0, curses.ACS_HLINE, width - 1)
        stdscr.addnstr(footer_at + 1, 0, FOOTER, width - 1)
        for offset, line in enumerate(editor.display_lines()):
            row = footer_at + 2 + offset
            if row < height:
                stdscr.addnstr(row, 0, line, width - 1)
        stdscr.refresh()

    async def run_prompt(prompt: str) -> None:
        transcript.user(prompt)
        transcript.thinking()
        display = _Display(transcript)
        try:
            await _run_turn(agent, session, prompt, display)
            state["connected"] = True
        except asyncio.CancelledError:
            state["cancel"] = False
        except Exception as exc:
            text = str(exc)
            if "context" in text.lower():
                transcript.error("context limit exceeded", "request was prevented by HANS context budget")
            elif "connection" in text.lower() or "tunnel" in text.lower():
                state["connected"] = False
                transcript.error("model request failed", text.splitlines()[0][:160])
            else:
                detail = text.splitlines()[0][:160]
                transcript.error("model request failed", detail)
            if debug_enabled():
                transcript.debug(f"{exc.__class__.__name__}: {exc}")

    async def loop(stdscr) -> None:
        curses.curs_set(1)
        stdscr.keypad(True)
        stdscr.nodelay(True)
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
            if state["task"] is not None:
                if key in {3, "\x03"}:
                    request_cancel()
                elif key in {17, "\x11"}:
                    request_cancel()
                    return
                continue
            name = _key_name(key)
            if name is None:
                continue
            submitted = editor.on_key(name)
            if submitted is None:
                continue
            if submitted == "":
                return
            if is_exit_command(submitted):
                return
            state["task"] = asyncio.create_task(run_prompt(submitted))

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()
    tty_attr = termios.tcgetattr(sys.stdin)
    raw_attr = termios.tcgetattr(sys.stdin)
    raw_attr[0] = raw_attr[0] & ~(termios.IXON | termios.IXOFF)
    termios.tcsetattr(sys.stdin, termios.TCSANOW, raw_attr)
    try:
        await loop(stdscr)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSANOW, tty_attr)
        curses.nocbreak()
        curses.echo()
        curses.endwin()
        signal.signal(signal.SIGINT, previous_int)
        signal.signal(signal.SIGTSTP, previous_stop)
        session.close()


def _key_name(key) -> str | None:
    if key in {3, "\x03"}:
        return "ctrl-c"
    if key in {4, "\x04"}:
        return "ctrl-d"
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
