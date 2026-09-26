import asyncio
from pathlib import Path

from bolt_next.events import (
    AssistantMessageDelta,
    RequestCompleted,
    ToolCompleted,
    ToolOutput,
    ToolStarted,
    VerificationEvidence,
    VerificationFailed,
    VerificationStarted,
)
from bolt_next.runtime import run_config
from bolt_next.tui import (
    FOOTER,
    _Display,
    format_header,
    format_tool_call,
    format_tool_result,
    read_user_message,
    serve,
    turn_error_message,
)
from bolt_next.tui_screen import Editor, Transcript, is_exit_command, layout_rows, visible_transcript


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
    message = read_user_message(_Lines(["Inspect this repository.", "", "Run the tests.", None]))
    assert message == "Inspect this repository.\n\nRun the tests."


def test_embedded_newlines_are_preserved() -> None:
    message = read_user_message(_Lines(["alpha", "beta", "gamma", None]))
    assert message is not None
    assert message.split("\n") == ["alpha", "beta", "gamma"]


def test_one_pasted_block_is_one_message_and_then_stop() -> None:
    read_line = _Lines(["Inspect this repository.", "Find the relevant Go implementation.", None])
    assert read_user_message(read_line) == "Inspect this repository.\nFind the relevant Go implementation."
    assert read_user_message(read_line) is None


def test_footer_lists_the_real_controls() -> None:
    assert FOOTER == "Enter send · Ctrl-D send · Ctrl-C cancel · Ctrl-Q exit"


def test_header_is_compact(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    workspace = tmp_path / "covered_call_bot"
    text = format_header("qwen3.6-27b", workspace, True)
    assert text.startswith("HANS")
    assert "qwen3.6-27b" in text
    assert "workspace: ~/covered_call_bot" in text
    assert "● connected" in text
    assert "https://" not in text
    assert "trycloudflare" not in text
    assert "○ disconnected" in format_header("qwen3.6-27b", workspace, False)


def test_normal_tool_activity_is_concise() -> None:
    call = format_tool_call("read_file", "invoice/money.go")
    done = format_tool_result("run_command", True, 0)
    failed = format_tool_result("run_command", False, 1)
    assert call == "  ◇ read_file  invoice/money.go"
    assert "[tool_started]" not in call
    assert done == "  ✓ run_command  exit 0"
    assert failed == "  ✗ run_command  exit 1"


def test_debug_marker_is_not_in_normal_error() -> None:
    message = turn_error_message("connection", "Connection error.", debug=False)
    assert message.startswith("\n✗ ")
    assert "Traceback" not in message
    assert "[connection]" not in message
    debug = turn_error_message("connection", "Connection error.", debug=True)
    assert "[connection]" in debug


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


def test_user_and_assistant_are_separate_lines() -> None:
    transcript = Transcript()
    transcript.user("hi")
    transcript.thinking()
    transcript.stream("Hi!")
    transcript.stream(" How can I help?")
    transcript.finish()
    lines = transcript.render(80)
    user_at = lines.index("> hi")
    assistant_at = lines.index("Hi! How can I help?")
    assert assistant_at > user_at
    assert not any(line.startswith("> hiHi") for line in lines)
    assert lines.count("Hi! How can I help?") == 1


def test_streaming_does_not_add_a_newline_per_chunk() -> None:
    transcript = Transcript()
    transcript.stream("Hello")
    transcript.stream(" there")
    assert transcript.render(80) == ["Hello there"]


def test_enter_submits_ctrl_d_remains_compatible_and_ctrl_q_exits() -> None:
    editor = Editor()
    editor.on_key("char:line one")
    assert editor.on_key("enter") == "line one"
    assert editor.on_key("enter") == ""
    editor.lines = ["line one", "line two"]
    assert editor.on_key("ctrl-d") == "line one\nline two"
    editor.on_key("char:keep")
    assert editor.on_key("ctrl-q") == ""
    assert editor.lines == [""]


def test_exit_and_quit_are_not_model_prompts() -> None:
    assert is_exit_command("exit")
    assert is_exit_command(" quit ")
    assert not is_exit_command("exit the file")


def test_exit_command_does_not_call_the_runtime() -> None:
    seen: list[str] = []

    async def run_turn(prompt: str) -> None:
        seen.append(prompt)

    class _Once:
        def __init__(self) -> None:
            self.sent = False

        def __call__(self, _prompt: str) -> str:
            if not self.sent:
                self.sent = True
                return "exit"
            raise EOFError

    asyncio.run(serve(_Once(), run_turn))
    assert seen == []


def test_ctrl_c_clears_the_editor_without_submitting() -> None:
    editor = Editor()
    editor.on_key("char:partial")
    assert editor.on_key("ctrl-c") is None
    assert editor.lines == [""]


def test_display_consumes_semantic_tool_and_verification_events() -> None:
    transcript = Transcript()
    display = _Display(transcript)
    command = "pytest -q"
    evidence = VerificationEvidence(command, 1, True, True, False, "2026-09-26T00:00:00+00:00")

    display.event(ToolStarted("verify-1", "run_command", command))
    display.event(VerificationStarted("verify-1", command))
    display.event(ToolOutput("verify-1", "authoritative tool output"))
    display.event(ToolCompleted("verify-1", "run_command", command, False, 1))
    display.event(VerificationFailed("verify-1", evidence))

    assert transcript.pieces[-1].text == "verification failed"
    display.event(ToolStarted("write-1", "write_file", "main.py"))
    assert transcript.pieces[-1].text == "correcting"
    display.event(ToolOutput("write-1", "Wrote main.py (5 bytes)"))
    display.event(ToolCompleted("write-1", "write_file", "main.py", True))
    display.event(AssistantMessageDelta("Fixed it."))
    display.event(RequestCompleted(None))
    assert transcript.pieces[-1].text == "completed (verification not established)"


def test_verification_status_and_completion_require_tool_evidence() -> None:
    failed = VerificationEvidence("pytest -q", 1, True, True, False, "2026-09-26T00:00:00+00:00")
    passed = VerificationEvidence("pytest -q", 0, True, True, True, "2026-09-26T00:00:01+00:00")
    transcript = Transcript()
    transcript.tool_started("run_command  pytest -q")
    transcript.stage("verifying")
    transcript.tool_finished("pytest -q", ok=False)
    transcript.stage("verification failed")
    transcript.completed(failed)
    failed_lines = transcript.render(80)
    assert "  ✗ pytest -q" in failed_lines
    assert "  ✓ completed (verification failed)" in failed_lines
    unverified = Transcript()
    unverified.completed(None)
    assert unverified.render(80) == ["  ✓ completed (verification not established)"]
    verified = Transcript()
    verified.completed(passed)
    assert verified.render(80) == ["  ✓ completed"]


def test_errors_stay_in_the_transcript() -> None:
    transcript = Transcript()
    transcript.error("model request failed", "connection refused")
    rendered = "\n".join(transcript.render(80))
    assert "✗ model request failed" in rendered
    assert "connection refused" in rendered
    assert "Traceback" not in rendered


def test_resize_reflows_without_duplicating_the_message() -> None:
    transcript = Transcript()
    transcript.user("Investigate the failing test")
    transcript.stream("The discount is applied before tax, which changes the total.")
    wide = transcript.render(80)
    narrow = transcript.render(24)
    assert sum(line.startswith("> ") for line in wide) == 1
    assert sum(line.startswith("> ") for line in narrow) == 1
    assert "Investigate" in " ".join(narrow)
    assert len(visible_transcript(narrow, 3)) == 3
    conversation, editor_height = layout_rows(24, 2)
    assert conversation >= 1
    assert editor_height == 3
