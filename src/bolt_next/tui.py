from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from urllib.parse import urlparse

from agents import Runner, SQLiteSession, set_tracing_disabled
from agents.run_config import RunConfig

from bolt_next.agent import create_agent
from bolt_next.context_budget import fit_model_input


def debug_enabled() -> bool:
    return os.environ.get("HANS_DEBUG", "").strip().lower() in {"1", "true", "yes"}


def format_header(model: str, workspace: Path, host: str) -> str:
    try:
        shown = "~/" + str(workspace.relative_to(Path.home()))
    except ValueError:
        shown = str(workspace)
    gap = max(2, 40 - len("HANS") - len(model))
    lines = ["HANS" + (" " * gap) + model, shown if not host else f"{shown}  {host}"]
    return "\n".join(lines)


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
    def __init__(self) -> None:
        self.debug = debug_enabled()
        self._last_name: str | None = None

    def event(self, event) -> None:
        if event.type == "raw_response_event":
            data = event.data
            if getattr(data, "type", None) in {"response.output_text.delta", "output_text.delta"}:
                print(getattr(data, "delta", ""), end="", flush=True)
            return
        if event.type != "run_item_stream_event":
            return
        if event.name == "tool_called":
            item = event.item
            raw = getattr(item, "raw_item", None)
            arguments = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
            self._last_name = getattr(item, "tool_name", None)
            if self.debug:
                print(f"\n[tool_called] name={self._last_name} arguments={arguments}", flush=True)
            else:
                print("\n" + format_tool_call(self._last_name, arguments), flush=True)
        elif event.name == "tool_output":
            output = str(event.item.output)
            if self.debug:
                print(f"\n[tool_output]\n{output}", flush=True)
            else:
                print(format_tool_result(self._last_name, output), flush=True)


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


async def _run_turn(agent, session: SQLiteSession, prompt: str) -> None:
    result = Runner.run_streamed(agent, prompt, session=session, run_config=run_config())
    display = _Display()
    try:
        async for event in result.stream_events():
            display.event(event)
        print(flush=True)
    except asyncio.CancelledError:
        result.cancel()
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
        if prompt.strip().lower() in {"exit", "quit"}:
            return
        if not prompt.strip():
            continue
        if debug_enabled():
            print("[user_turn]", flush=True)
        try:
            await run_turn(prompt)
        except KeyboardInterrupt:
            print("\ninterrupted\n", flush=True)
        except Exception as exc:
            print(turn_error_message(exc), flush=True)


async def _run_tui() -> None:
    set_tracing_disabled(True)
    workspace = Path(os.environ.get("BOLT_WORKSPACE") or Path.cwd()).expanduser()
    host = urlparse(os.environ.get("BOLT_MODEL_BASE_URL", "")).hostname or ""
    print(
        format_header(os.environ.get("BOLT_MODEL", "qwen3.6-27b"), workspace, host),
        flush=True,
    )
    print("Enter newline · Ctrl-D send · Ctrl-C interrupts · Ctrl-D on empty exits", flush=True)
    agent = create_agent(os.environ.get("BOLT_WORKSPACE"))
    session = SQLiteSession("hans-tui")

    async def run_turn(prompt: str) -> None:
        await _run_turn(agent, session, prompt)

    try:
        await serve(input, run_turn)
    finally:
        session.close()


def run_tui() -> None:
    try:
        asyncio.run(_run_tui())
    except KeyboardInterrupt:
        print("\ninterrupted")
