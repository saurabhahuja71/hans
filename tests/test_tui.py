import asyncio
from pathlib import Path

import pytest

from bolt_next.events import (
    AssistantMessageDelta,
    ModelChanged,
    ModelStatus,
    PermissionPolicyChanged,
    ReasoningModeChanged,
    ReasoningModeStatus,
    RequestCancelled,
    RuntimeControlRejected,
    RuntimeControlStatus,
    SessionCleared,
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
    VerificationEvidence,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)
from bolt_next.model_catalog import ModelInfo
from bolt_next.runtime import run_config
from bolt_next.tui import (
    FOOTER,
    _Display,
    _InputDecoder,
    _dispatch_control,
    detail_window,
    format_header,
    format_tool_call,
    format_tool_result,
    read_user_message,
    serve,
    turn_error_message,
)
from bolt_next.tui_screen import (
    Editor,
    TodoList,
    Transcript,
    copy_osc52,
    display_bounded,
    footer_text,
    handle_local_command,
    is_exit_command,
    layout_rows,
    task_summary_has_hans_changes,
    visible_transcript,
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
    message = read_user_message(_Lines(["Inspect this repository.", "", "Run the tests.", None]))
    assert message == "Inspect this repository.\n\nRun the tests."


def test_footer_and_compact_header_list_real_controls(tmp_path: Path, monkeypatch) -> None:
    assert FOOTER == "Enter send · Shift+Enter newline · Ctrl-D send / empty exit · Ctrl-Q quit"
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    text = format_header("qwen3.6-27b", tmp_path / "project", True)
    assert text.startswith("HANS")
    assert "qwen3.6-27b" in text
    assert "workspace: ~/project" in text
    assert "● connected" in text
    assert "https://" not in text


@pytest.mark.parametrize(
    ("state", "request_active", "has_task_changes", "expected"),
    (
        ("IDLE", False, False, "Enter send · Shift+Enter newline · Ctrl-D send / empty exit · Ctrl-Q quit"),
        ("INVESTIGATING", True, False, "◉ INVESTIGATING · Ctrl-C cancel · Ctrl-Q quit"),
        ("VERIFYING · pytest -q", True, False, "◉ VERIFYING · Ctrl-C cancel · Ctrl-Q quit"),
        ("COMPLETE", False, True, "✓ COMPLETE · Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"),
        ("COMPLETE", False, False, "✓ COMPLETE · Enter new task · Ctrl-Q quit"),
        ("CANCELLED", False, False, "⏸ CANCELLED · Enter new task · Ctrl-Q quit"),
        ("FAILED", False, False, "✗ FAILED · Enter retry/new task · Ctrl-Q quit"),
        ("IDLE", False, True, "Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"),
    ),
)
def test_footer_text_uses_semantic_task_context(
    state: str, request_active: bool, has_task_changes: bool, expected: str
) -> None:
    assert (
        footer_text(state, request_active=request_active, has_task_changes=has_task_changes)
        == expected
    )


@pytest.mark.parametrize(
    ("summary", "expected"),
    (
        ("preexisting_git_worktree_changes: notes.txt", False),
        ("changed_files: none\ncreated_files: none", False),
        ("changed_files: modified.py", True),
        ("created_files: created.py", True),
    ),
)
def test_task_summary_has_hans_changes_distinguishes_task_and_preexisting_work(
    summary: str, expected: bool
) -> None:
    assert task_summary_has_hans_changes(summary) is expected


def test_normal_tool_activity_is_concise_and_raw_errors_are_debug_only() -> None:
    assert format_tool_call("read_file", "invoice/money.go") == "  ◇ read_file  invoice/money.go"
    assert format_tool_result("run_command", True, 0) == "  ✓ run_command  exit 0"
    assert format_tool_result("run_command", False, 1) == "  ✗ run_command  exit 1"
    normal = turn_error_message("connection", "Connection error.", debug=False, debug_detail="raw provider response")
    assert "raw provider response" not in normal
    debug = turn_error_message("connection", "Connection error.", debug=True, debug_detail="raw provider response")
    assert "[connection] raw provider response" in debug


@pytest.mark.parametrize("category", ("configuration", "authentication", "model"))
def test_terminal_uses_semantic_failure_titles(category: str) -> None:
    rendered = turn_error_message(category, "details", debug=False)
    assert f"✗ {category}" in rendered or "✗ model request failed" in rendered


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


def test_editor_submission_newlines_cancellation_and_exit() -> None:
    editor = Editor()
    assert editor.on_key("char:line one") is None
    assert editor.on_key("shift-enter") is None
    assert editor.on_key("char:line two") is None
    assert editor.on_key("enter") == "line one\nline two"
    assert editor.lines == [""]
    assert editor.on_key("ctrl-d") == ""
    editor.on_key("char:submit with ctrl-d")
    assert editor.on_key("ctrl-d") == "submit with ctrl-d"
    assert editor.lines == [""]
    editor.on_key("char:partial")
    assert editor.on_key("ctrl-c") is None
    assert editor.lines == [""]
    editor.on_key("char:keep")
    assert editor.on_key("ctrl-q") == ""
    assert editor.lines == [""]


def test_terminal_key_decoder_preserves_controls_and_bracketed_paste() -> None:
    decoder = _InputDecoder()
    assert decoder.feed("\n") == ["enter"]
    assert decoder.feed("\x1b[13;2u") == ["shift-enter"]
    assert decoder.feed("\x03") == ["ctrl-c"]
    assert decoder.feed("\x04") == ["ctrl-d"]
    assert decoder.feed("\x07") == ["ctrl-g"]
    assert decoder.feed("\x1a") == ["ctrl-z"]
    assert decoder.feed("\x11") == ["ctrl-q"]

    message = "line one\nline two"
    pasted = _InputDecoder()
    events = [event for key in "\x1b[200~" + message + "\x1b[201~" for event in pasted.feed(key)]
    assert "".join(event[5:] for event in events) == message


def test_exit_and_quit_are_not_model_prompts() -> None:
    assert is_exit_command("exit")
    assert is_exit_command(" quit ")
    assert not is_exit_command("exit the file")


def test_curses_display_uses_semantic_lifecycle_hierarchy_and_tool_privacy(monkeypatch) -> None:
    monkeypatch.delenv("HANS_DEBUG", raising=False)
    transcript = Transcript()
    display = _Display(transcript)
    passed = VerificationEvidence("pytest -q", 0, True, True, True, "2026-09-28T00:00:00+00:00")
    failed = VerificationEvidence("pytest -q", 1, True, True, False, "2026-09-28T00:00:01+00:00")

    display.event(UserMessageSubmitted("fix it"))
    display.event(RequestStarted("fix it"))
    assert display.state == "INVESTIGATING"
    display.event(AssistantMessageDelta("I found it."))
    display.event(ToolStarted("read-1", "read_file", "a.py"))
    assert display.state == "INVESTIGATING"
    display.event(ToolOutput("read-1", "private source bytes"))
    display.event(ToolCompleted("read-1", "read_file", "a.py", True))
    display.event(ToolStarted("write-1", "write_file", "a.py"))
    assert display.state == "EDITING"
    display.event(VerificationStarted("verify-1", "pytest -q"))
    assert display.state.startswith("VERIFYING")
    display.event(VerificationFailed("verify-1", failed))
    assert display.state.startswith("CORRECTING")
    display.event(VerificationPassed("verify-2", passed))
    display.event(RequestCompleted(passed))
    assert display.state == "COMPLETE"

    rendered = "\n".join(transcript.render(120))
    assert "YOU\n> fix it" in rendered
    assert "HANS\nI found it." in rendered
    assert "TOOL read_file  a.py" in rendered
    assert "read-1" not in rendered
    assert "private source bytes" not in rendered
    assert "VERIFICATION" in rendered
    assert "✓ pytest -q passed" in rendered
    assert "FINAL RESULT" in rendered
    assert "Verified: pytest -q" in rendered


def test_curses_debug_tool_output_is_explicit_and_bounded(monkeypatch) -> None:
    monkeypatch.setenv("HANS_DEBUG", "1")
    transcript = Transcript()
    display = _Display(transcript)
    display.event(ToolOutput("call", "authoritative tool output"))
    assert "authoritative tool output" in "\n".join(transcript.render(120))

    display.event(ToolOutput("call", "x" * 4_001))
    rendered = "\n".join(transcript.render(8_000))
    assert "display truncated (1 characters omitted)" in rendered
    assert "x" * 4_001 not in rendered


def test_curses_changes_diff_undo_and_errors_are_task_scoped() -> None:
    transcript = Transcript()
    display = _Display(transcript)
    display.event(
        TaskChangeSummary(
            "changed_files: existing.py, new.py\n"
            "created_files: new.py\n"
            "preexisting_git_worktree_changes: notes.txt\n"
            "git: available"
        )
    )
    display.event(TaskDiff("--- a/existing.py\n+++ b/existing.py\n+updated"))
    display.event(TaskDiff("x" * 4_001))
    assert transcript.pieces[-1].text.startswith("x" * 4_000)
    assert "display truncated (1 characters omitted)" in transcript.pieces[-1].text
    display.event(TaskUndoSucceeded(("existing.py",), ("new.py",)))
    assert display.state.startswith("IDLE")
    display.event(TaskUndoRefused(("changed.py",)))
    display.event(RequestFailed("connection", "model endpoint unavailable"))

    rendered = "\n".join(transcript.render(120))
    assert "CHANGES\n  M existing.py  HANS\n  A new.py  HANS" in rendered
    assert "Existing worktree changes preserved: notes.txt" in rendered
    assert "TASK DIFF" in rendered
    assert "+++ b/existing.py" in rendered
    assert "UNDO\n  ✓ Undo completed · restored: existing.py; removed: new.py" in rendered
    assert "ERROR\n  ✗ Undo refused" in rendered
    assert "Conflicts: changed.py" in rendered
    assert "✗ connection failed" in rendered
    assert display.state.startswith("FAILED")


def test_curses_display_footer_context_tracks_task_lifecycle() -> None:
    display = _Display(Transcript())
    display.has_task_changes = True

    display.event(RequestStarted("change a file"))
    assert display.request_active is True
    assert display.has_task_changes is False

    display.event(TaskChangeSummary("changed_files: changed.py"))
    assert display.has_task_changes is True
    display.event(RequestCompleted(None))
    assert display.request_active is False
    assert (
        footer_text(
            display.state,
            request_active=display.request_active,
            has_task_changes=display.has_task_changes,
        )
        == "✓ COMPLETE · Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"
    )

    display.event(TaskUndoSucceeded(("changed.py",), ()))
    assert display.has_task_changes is False
    assert (
        footer_text(
            display.state,
            request_active=display.request_active,
            has_task_changes=display.has_task_changes,
        )
        == "Enter send · Shift+Enter newline · Ctrl-D send / empty exit · Ctrl-Q quit"
    )

    display.event(TaskChangeSummary("created_files: new.py"))
    display.event(TaskUndoRefused(("new.py",)))
    assert display.has_task_changes is True
    assert (
        footer_text(
            display.state,
            request_active=display.request_active,
            has_task_changes=display.has_task_changes,
        )
        == "Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"
    )


def test_cancel_and_completion_claims_follow_semantic_events() -> None:
    transcript = Transcript()
    display = _Display(transcript)
    display.event(RequestCompleted(None))
    assert "Verification not established" in "\n".join(transcript.render(80))
    display.event(RequestCancelled())
    rendered = "\n".join(transcript.render(80))
    assert "CANCELLED" in rendered
    assert display.state == "CANCELLED"


def test_transcript_resize_reflows_hierarchy_without_duplicate_messages() -> None:
    transcript = Transcript()
    transcript.user("Investigate the failing test")
    transcript.stream("The discount is applied before tax, which changes the total.")
    wide = transcript.render(80)
    narrow = transcript.render(24)
    assert wide.count("YOU") == 1
    assert narrow.count("YOU") == 1
    assert wide.count("HANS") == 1
    assert narrow.count("HANS") == 1
    assert "Investigate" in " ".join(narrow)
    assert len(visible_transcript(narrow, 3)) == 3
    conversation, editor_height = layout_rows(24, 2)
    assert conversation >= 1
    assert editor_height == 3


def test_curses_failed_verification_is_not_claimed_complete() -> None:
    transcript = Transcript()
    display = _Display(transcript)
    failed = VerificationEvidence("pytest -q", 1, True, True, False, "2026-09-28T00:00:00+00:00")
    display.event(RequestCompleted(failed))

    rendered = "\n".join(transcript.render(120))
    assert display.state == "FAILED"
    assert "✗ VERIFICATION FAILED" in rendered
    assert "pytest -q" in rendered
    assert "HANS did not claim completion." in rendered


def test_local_todos_commands_and_bounded_presentation_are_ui_only() -> None:
    todos = TodoList(max_items=2, max_text_chars=12)
    assert handle_local_command("ordinary prompt", todos).handled is False
    assert handle_local_command("/unknown", todos).text == "Unknown command: /unknown"
    assert handle_local_command("/todo add  write   tests ", todos).text == "TODO added #1: write tests"
    assert handle_local_command("/todo add review", todos).text == "TODO added #2: review"
    assert handle_local_command("/todo add one too many", todos).text == "TODO error: list is limited to 2 items"
    assert handle_local_command("/todo done 1", todos).text == "TODO completed #1: write tests"
    assert handle_local_command("/todo list", todos).text == "TODO\nx #1 write tests\n  #2 review"
    assert handle_local_command("/todo remove no", todos).text == "TODO error: id must be a number"
    assert handle_local_command("/todo remove 2", todos).text == "TODO removed #2"
    assert handle_local_command("/todo clear", todos).text == "TODO cleared (1 item)"
    assert handle_local_command("/theme terminal", todos).theme == "terminal"
    assert handle_local_command("/theme sepia", todos).text == (
        "Theme error: choose one of dark, light, high-contrast, terminal"
    )
    model_status = handle_local_command("/models", todos)
    mode_status = handle_local_command("/mode", todos)
    mode_change = handle_local_command("/mode  HIGH ", todos)
    assert model_status.control is not None and model_status.control.kind == "model_status"
    assert mode_status.control is not None and mode_status.control.kind == "reasoning_mode_status"
    assert mode_change.control is not None
    assert (mode_change.control.kind, mode_change.control.mode) == ("set_reasoning_mode", "high")
    model_change = handle_local_command("/models use LARGE", todos)
    assert model_change.control is not None
    assert (model_change.control.kind, model_change.control.model_id) == ("select_model", "large")
    assert handle_local_command("/models use", todos).text == "Usage: /models [use <model>]"
    assert handle_local_command("/models use large extra", todos).text == "Usage: /models [use <model>]"
    assert handle_local_command("/models list", todos).text == "Usage: /models [use <model>]"
    assert handle_local_command("/mode high extra", todos).text == "Usage: /mode [mode]"
    assert display_bounded("abcdef", 3) == "abc\n… display truncated (3 characters omitted)"
    assert display_bounded("abc", 3) == "abc"


def test_runtime_control_commands_are_parsed_rendered_and_dispatched_without_a_model_turn() -> None:
    todos = TodoList()
    status = handle_local_command("/permissions", todos)
    changed = handle_local_command("/permissions WRITE deny", todos)
    clear = handle_local_command("/clear", todos)
    model_status = handle_local_command("/models", todos)
    reasoning_status = handle_local_command("/mode", todos)
    reasoning_changed = handle_local_command("/mode high", todos)
    invalid_reasoning_mode = handle_local_command("/mode unsupported", todos)
    model_changed = handle_local_command("/models use LARGE", todos)
    unknown_model = handle_local_command("/models use missing", todos)

    assert status.control is not None and status.control.kind == "permissions_status"
    assert changed.control is not None
    assert (changed.control.kind, changed.control.category, changed.control.allowed) == ("set_permission", "write", False)
    assert clear.control is not None and clear.control.kind == "clear_session"
    assert model_status.control is not None and model_status.control.kind == "model_status"
    assert reasoning_status.control is not None and reasoning_status.control.kind == "reasoning_mode_status"
    assert reasoning_changed.control is not None
    assert (reasoning_changed.control.kind, reasoning_changed.control.mode) == ("set_reasoning_mode", "high")
    assert invalid_reasoning_mode.control is not None
    assert model_changed.control is not None
    assert (model_changed.control.kind, model_changed.control.model_id) == ("select_model", "large")
    assert unknown_model.control is not None
    assert (unknown_model.control.kind, unknown_model.control.model_id) == ("select_model", "missing")
    assert handle_local_command("/permissions foo allow", todos).text == "Unknown permission: foo"
    assert handle_local_command("/permissions write maybe", todos).text == "Expected allow or deny."
    assert handle_local_command("/permissions write", todos).text == (
        "Usage: /permissions [read|write|execute] [allow|deny]"
    )
    assert handle_local_command("/clear now", todos).text == "Usage: /clear"

    class Runtime:
        def __init__(self) -> None:
            self.permissions = {"read": True, "write": True, "execute": True}
            self.calls: list[tuple[object, ...]] = []
            self.models = (
                ModelInfo(
                    id="configured-model",
                    display_name="Configured Model",
                    endpoint_profile="configured endpoint",
                    context_tokens=128_000,
                    supported_reasoning_modes=("none", "high"),
                    none_semantics="omit",
                ),
                ModelInfo(
                    id="large",
                    display_name="Large Model",
                    endpoint_profile="configured endpoint",
                    context_tokens=256_000,
                    supported_reasoning_modes=("none", "low"),
                    none_semantics="literal",
                ),
            )
            self.model = self.models[0]
            self.reasoning_mode: str | None = None

        def get_control_status(self) -> RuntimeControlStatus:
            self.calls.append(("status",))
            return RuntimeControlStatus(**{f"{name}_allowed": allowed for name, allowed in self.permissions.items()})

        def set_permission(self, category: str, allowed: bool) -> PermissionPolicyChanged:
            self.calls.append(("permission", category, allowed))
            self.permissions[category] = allowed
            return PermissionPolicyChanged(category, allowed)

        async def clear_session_history(self) -> SessionCleared:
            self.calls.append(("clear",))
            return SessionCleared()

        def get_model_status(self) -> ModelStatus:
            self.calls.append(("model_status",))
            return ModelStatus(self.model, self.reasoning_mode, self.models)

        def select_model(self, model_id: str) -> ModelChanged | RuntimeControlRejected | None:
            self.calls.append(("select_model", model_id))
            selected = next((model for model in self.models if model.id == model_id), None)
            if selected is None:
                return RuntimeControlRejected("Model selection is unavailable.")
            if selected.id == self.model.id:
                return None
            previous_model_id = self.model.id
            self.model = selected
            self.reasoning_mode = None
            return ModelChanged(previous_model_id, selected, new_session_started=True, reasoning_reset=True)

        def get_reasoning_mode_status(self) -> ReasoningModeStatus:
            self.calls.append(("reasoning_mode_status",))
            return ReasoningModeStatus(
                self.reasoning_mode, self.model.supported_reasoning_modes, self.model.none_semantics
            )

        def set_reasoning_mode(self, mode: str) -> ReasoningModeChanged | RuntimeControlRejected:
            self.calls.append(("reasoning_mode", mode))
            if mode not in self.model.supported_reasoning_modes:
                return RuntimeControlRejected(f"Unsupported reasoning mode: {mode}")
            self.reasoning_mode = mode
            return ReasoningModeChanged(mode)

    runtime = Runtime()
    transcript = Transcript()
    shown_models: list[str] = []
    display = _Display(
        transcript,
        on_model=lambda model: shown_models.append(str(getattr(model, "display_name", ""))),
    )

    async def scenario() -> None:
        for local in (
            status,
            changed,
            clear,
            model_status,
            reasoning_status,
            reasoning_changed,
            invalid_reasoning_mode,
            model_changed,
            unknown_model,
        ):
            assert local.control is not None
            await _dispatch_control(runtime, local.control, display)

    asyncio.run(scenario())
    rendered = "\n".join(transcript.render(120))
    assert runtime.calls == [
        ("status",),
        ("permission", "write", False),
        ("clear",),
        ("model_status",),
        ("reasoning_mode_status",),
        ("reasoning_mode", "high"),
        ("reasoning_mode", "unsupported"),
        ("select_model", "large"),
        ("select_model", "missing"),
    ]
    assert shown_models == ["Configured Model", "Large Model"]
    assert "model: Large Model" in format_header(shown_models[-1], Path.cwd())
    assert "Permissions" in rendered
    assert "✓ write permission set to deny." in rendered
    assert "✓ Conversation history cleared." in rendered
    assert "LOCAL\n  MODELS\n  * Active: Configured Model (configured-model)" in rendered
    assert "Context: 128,000 tokens" in rendered
    assert "Reasoning support: none, high" in rendered
    assert "Current reasoning: configured default" in rendered
    assert "Large Model (large)" in rendered
    assert "Context: 256,000 tokens" in rendered
    assert "✓ Switched from configured-model to Large Model (large)." in rendered
    assert "✓ New conversation started." in rendered
    assert "✓ Reasoning reset to configured default." in rendered
    assert "https://" not in rendered
    assert "LOCAL\n  REASONING MODE\n  Current: configured default\n  Supported: none, high" in rendered
    assert "LOCAL\n  REASONING\n  ✓ Mode set to high." in rendered
    assert "Unsupported reasoning mode: unsupported" in rendered
    assert "Model selection is unavailable." in rendered

    display.event(RuntimeControlRejected("Cannot clear the session while HANS is busy."))
    assert "Cannot clear the session while HANS is busy." in "\n".join(transcript.render(120))


def test_non_tty_local_commands_do_not_run_model_turns() -> None:
    async def submit_once(message: str) -> list[str]:
        seen: list[str] = []

        async def run_turn(prompt: str) -> None:
            seen.append(prompt)

        async def handle_local(prompt: str) -> bool:
            return handle_local_command(prompt, TodoList()).handled

        await serve(_Lines([message, None]), run_turn, handle_local)
        return seen

    assert asyncio.run(submit_once("/permissions")) == []
    assert asyncio.run(submit_once("/clear")) == []
    assert asyncio.run(submit_once("/models")) == []
    assert asyncio.run(submit_once("/models use any-model")) == []
    assert asyncio.run(submit_once("/mode high")) == []
    assert asyncio.run(submit_once("/unknown")) == []
    assert asyncio.run(submit_once("ordinary prompt")) == ["ordinary prompt"]


def test_osc52_copy_is_plain_text_bounded_and_failure_safe() -> None:
    captured: list[bytes] = []
    ok, message = copy_osc52("a\nb", captured.append)
    assert (ok, message) == (True, "Copied")
    assert captured == [b"\x1b]52;c;YQpi\x07"]

    ok, message = copy_osc52("abcdef", captured.append, max_chars=3)
    assert (ok, message) == (True, "Copied")
    assert b"YWJjCuKApiBkaXNwbGF5IHRydW5jYXRlZCAoMyBjaGFyYWN0ZXJzIG9taXR0ZWQp" in captured[-1]

    def fail(_data: bytes) -> None:
        raise RuntimeError("no clipboard")

    assert copy_osc52("text", fail) == (False, "Clipboard unavailable")


def test_detail_window_decoder_and_retention_bounds() -> None:
    assert detail_window("abcdef\nx", 3, 2) == ["abc", "def"]
    assert detail_window("abcdef\nx", 3, 2, 1) == ["def", "x"]
    assert detail_window("abcdef\nx", 3, 2, 100) == ["def", "x"]

    decoder = _InputDecoder()
    assert decoder.feed("\x0f\x19\x1b") == ["ctrl-o", "ctrl-y"]
    assert decoder.flush() == ["escape"]
    assert decoder.feed(259) == ["up"]
    assert decoder.feed(258) == ["down"]
    assert decoder.feed(339) == ["pageup"]
    assert decoder.feed(338) == ["pagedown"]

    transcript = Transcript()
    display = _Display(transcript)
    display.max_tool_outputs = 1
    display.max_transcript_pieces = 2
    for number in range(3):
        display.event(UserMessageSubmitted(f"message {number}"))
    display.event(ToolOutput("old", "old output"))
    display.event(ToolOutput("new", "new output"))
    assert display.tool_outputs == {"new": "new output"}
    assert display.latest_tool_output() == "new output"
    assert [piece.text for piece in transcript.pieces] == ["message 1", "message 2"]
