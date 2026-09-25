import asyncio
from pathlib import Path

from bolt_next.tui import (
    format_header,
    format_tool_call,
    format_tool_result,
    read_user_message,
    run_config,
    serve,
    turn_error_message,
)


class _Lines:
    def __init__(self, lines: list[object]) -> None:
        self._lines = iter(lines)
        self._eof = False

    def __call__(self, _prompt: str) -> str:
        if self._eof:
            raise EOFError
        item = next(self._lines, None)
        if item is None:
            self._eof = True
            raise EOFError
        return str(item)


def test_multiline_text_is_one_message() -> None:
    message = read_user_message(
        _Lines(
            [
                "Inspect this repository.",
                "",
                "Run the tests.",
                None,
            ]
        )
    )
    assert message == "Inspect this repository.\n\nRun the tests."


def test_embedded_newlines_are_preserved() -> None:
    message = read_user_message(_Lines(["alpha", "beta", "gamma", None]))
    assert message is not None
    assert message.split("\n") == ["alpha", "beta", "gamma"]


def test_one_pasted_block_is_one_message_and_then_stop() -> None:
    read_line = _Lines(
        [
            "Inspect this repository.",
            "Find the relevant Go implementation and tests.",
            "Diagnose the problem.",
            None,
        ]
    )
    first = read_user_message(read_line)
    second = read_user_message(read_line)
    assert first == (
        "Inspect this repository.\n"
        "Find the relevant Go implementation and tests.\n"
        "Diagnose the problem."
    )
    assert second is None


def test_lines_are_not_separate_messages() -> None:
    read_line = _Lines(["one", "two", None])
    messages = []
    while True:
        message = read_user_message(read_line)
        if message is None:
            break
        messages.append(message)
    assert messages == ["one\ntwo"]


def test_header_is_compact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workspace = tmp_path / "covered_call_bot"
    text = format_header("qwen3.6-27b", workspace, "ranger-girls-gel-gain.trycloudflare.com")
    assert text.startswith("HANS")
    assert "qwen3.6-27b" in text
    assert "~/covered_call_bot" in text
    assert "https://" not in text
    assert "/v1" not in text


def test_normal_tool_activity_is_concise() -> None:
    call = format_tool_call("read_file", {"path": "invoice/money.go"})
    done = format_tool_result("run_command", "exit_code=0\nstdout:\nok\n")
    failed = format_tool_result("run_command", "exit_code=1\nstdout:\nFAIL\n")
    assert call == "  ◇ read_file  invoice/money.go"
    assert "[tool_called]" not in call
    assert "ok" not in done
    assert done == "  ✓ run_command  exit 0"
    assert failed == "  ✗ run_command  exit 1"


def test_debug_marker_is_not_in_normal_error() -> None:
    message = turn_error_message(RuntimeError("Connection error."), debug=False)
    assert message.startswith("\n✗ ")
    assert "Traceback" not in message
    assert "RuntimeError" not in message
    debug = turn_error_message(RuntimeError("Connection error."), debug=True)
    assert "RuntimeError" in debug


def test_tracing_is_disabled_without_openai_key() -> None:
    assert run_config().tracing_disabled is True


def test_ctrl_c_during_prompt_and_turn_stays_in_session() -> None:
    reads = iter([KeyboardInterrupt(), "hi", None, None])

    def read_line(_prompt: str) -> str:
        item = next(reads)
        if item is None:
            raise EOFError
        if isinstance(item, BaseException):
            raise item
        return str(item)

    seen: list[str] = []

    async def run_turn(prompt: str) -> None:
        seen.append(prompt)
        raise KeyboardInterrupt

    asyncio.run(serve(read_line, run_turn))
    assert seen == ["hi"]
