import json
from pathlib import Path

import pytest
from agents import Agent, Runner, SQLiteSession
from agents.run_config import CallModelData, ModelInputData, RunConfig
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

from bolt_next.context_budget import (
    context_token_limit,
    estimate_tokens,
    fit_model_input,
    request_tokens,
)
from bolt_next.tui import turn_error_message
from bolt_next.workspace import make_read_file_tool, make_run_command_tool, make_write_file_tool
from tests.test_sdk_runtime import make_agent, run
from tests.test_workspace import invoke


def test_small_file_is_returned_unchanged(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    result = invoke(make_read_file_tool(tmp_path), '{"path":"note.txt"}')
    assert result == "hello"
    assert "returned_range" not in result


def test_large_file_is_a_bounded_authoritative_range(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "16384")
    lines = [f"LINE-{index}-" + ("a" * 40) for index in range(1, 4001)]
    lines[-1] = "UNIQUE_LAST_LINE"
    (tmp_path / "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = invoke(make_read_file_tool(tmp_path), '{"path":"big.py"}')
    assert "UNIQUE_LAST_LINE" not in result
    assert "returned_range:" in result
    assert "remaining_ranges:" in result
    assert "request_next:" in result
    assert estimate_tokens(result) <= context_token_limit()
    body = result.split("---\n", 1)[1].splitlines()
    assert body[0] == lines[0]
    assert all(line in lines for line in body)


def test_large_file_can_be_read_incrementally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "16384")
    lines = [f"LINE-{index}" for index in range(1, 4001)]
    (tmp_path / "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tool = make_read_file_tool(tmp_path)
    first = invoke(tool, '{"path":"big.py"}')
    next_start = int(first.split("request_next: read_file path=big.py start_line=", 1)[1].splitlines()[0])
    second = invoke(tool, json.dumps({"path": "big.py", "start_line": next_start}))
    assert f"returned_range: {next_start}-" in second
    body = second.split("---\n", 1)[1].splitlines()
    assert body[0] == lines[next_start - 1]
    assert lines[0] not in body


def test_explicit_oversized_range_returns_no_partial_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "16384")
    lines = [f"SECRET-{index}-" + ("b" * 80) for index in range(1, 4001)]
    (tmp_path / "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = invoke(
        make_read_file_tool(tmp_path),
        '{"path":"big.py","start_line":1,"end_line":4000}',
    )
    assert "No partial source was returned" in result
    assert "SECRET-1-" not in result


def test_filter_keeps_request_inside_budget() -> None:
    huge = "SOURCE " * 20000
    items = [
        {"role": "user", "content": "inspect the repository"},
        {"type": "function_call_output", "call_id": "a", "output": "go.mod\nmodule example"},
        {"type": "function_call_output", "call_id": "b", "output": "underlyings.txt\nBEL\nMANKIND"},
        {"type": "function_call_output", "call_id": "c", "output": huge},
    ]
    assert request_tokens("instructions", items) > 16384
    fitted = fit_model_input(
        CallModelData(
            model_data=ModelInputData(input=items, instructions="instructions"),
            agent=Agent(name="Hans"),
            context=None,
        )
    )
    assert request_tokens(fitted.instructions, fitted.input) <= 16384 * 3 // 4
    assert "go.mod" in json.dumps(fitted.input)
    assert "MANKIND" in json.dumps(fitted.input)
    assert huge not in json.dumps(fitted.input)
    assert "not a summary" in json.dumps(fitted.input)
    assert items[3]["output"] == huge


def test_context_error_does_not_end_the_session() -> None:
    message = turn_error_message(
        RuntimeError("request (34229 tokens) exceeds the available context size (16384 tokens)")
    )
    assert "session is still open" in message


def test_runner_does_not_send_the_whole_large_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "16384")
    lines = [f"LINE-{index}-" + ("c" * 40) for index in range(1, 4001)]
    lines[-1] = "UNIQUE_LAST_LINE"
    (tmp_path / "big.py").write_text("\n".join(lines) + "\n", encoding="utf-8")
    model = ScriptedModel(
        [
            ModelStep(output=[function_call("read_file", {"path": "big.py"}, call_id="read-1")]),
            ModelStep(output=[assistant_message("continued after the range")]),
        ]
    )
    agent = make_agent(model, tmp_path)
    session = SQLiteSession("context-budget")
    result = run(
        Runner.run(
            agent,
            "Inspect big.py",
            session=session,
            run_config=RunConfig(call_model_input_filter=fit_model_input),
        )
    )
    sent = repr(model.calls[1].input)
    assert "UNIQUE_LAST_LINE" not in sent
    assert "returned_range" in sent
    assert "LINE-1-" in sent
    assert request_tokens("", model.calls[1].input) <= 16384
    assert result.final_output == "continued after the range"
    assert (tmp_path / "big.py").read_text(encoding="utf-8").endswith("UNIQUE_LAST_LINE\n")
    session.close()
