from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse

from agents import Runner, SQLiteSession

from bolt_next.agent import create_agent


def _print_stream_event(event) -> None:
    if event.type == "raw_response_event":
        data = event.data
        if getattr(data, "type", None) in {"response.output_text.delta", "output_text.delta"}:
            print(getattr(data, "delta", ""), end="", flush=True)
    elif event.type == "run_item_stream_event" and event.name == "tool_called":
        item = event.item
        raw = getattr(item, "raw_item", None)
        arguments = raw.get("arguments") if isinstance(raw, dict) else getattr(raw, "arguments", None)
        print(
            f"\n[tool_called] name={getattr(item, 'tool_name', None)} arguments={arguments}",
            flush=True,
        )
    elif event.type == "run_item_stream_event" and event.name == "tool_output":
        print(f"\n[tool_output]\n{event.item.output}", flush=True)


async def _run_turn(agent, session: SQLiteSession, prompt: str) -> None:
    result = Runner.run_streamed(agent, prompt, session=session)
    try:
        async for event in result.stream_events():
            _print_stream_event(event)
        if result.final_output:
            print("\n", flush=True)
    except asyncio.CancelledError:
        result.cancel()
        raise


async def _run_tui() -> None:
    print("╭──────────────────────────────────────╮")
    print("│                Hans                 │")
    print("│       OpenAI Agents SDK runtime      │")
    print("╰──────────────────────────────────────╯")
    base = urlparse(os.environ.get("BOLT_MODEL_BASE_URL", ""))
    print(
        f"model={os.environ.get('BOLT_MODEL', 'qwen3.6-27b')} "
        f"endpoint={base.scheme}://{base.hostname}{base.path}",
        flush=True,
    )
    agent = create_agent(os.environ.get("BOLT_WORKSPACE"))
    session = SQLiteSession("hans-tui")
    try:
        while True:
            try:
                prompt = input("\n> ")
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if prompt.strip().lower() in {"exit", "quit"}:
                return
            if not prompt.strip():
                continue
            try:
                await _run_turn(agent, session, prompt.strip())
            except KeyboardInterrupt:
                print("\nInterrupted.\n", flush=True)
            except Exception as exc:
                print(f"\nError: {exc}\n", flush=True)
    finally:
        session.close()


def run_tui() -> None:
    try:
        asyncio.run(_run_tui())
    except KeyboardInterrupt:
        print("\nInterrupted.")
