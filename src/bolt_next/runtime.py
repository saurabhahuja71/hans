"""Agents SDK integration and translation into HANS semantic events."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Iterable

from agents import Runner, SQLiteSession, set_tracing_disabled
from agents.run_config import RunConfig

from bolt_next.agent import create_agent
from bolt_next.context_budget import fit_model_input
from bolt_next.events import (
    AssistantMessageComplete,
    AssistantMessageDelta,
    ConnectionChanged,
    HansEvent,
    RequestCancelled,
    RequestCompleted,
    RequestFailed,
    RequestStarted,
    ToolCompleted,
    ToolOutput,
    ToolStarted,
    UserMessageSubmitted,
    VerificationEvidence,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)


@dataclass(frozen=True, slots=True)
class _ToolCall:
    name: str
    detail: str


def run_config() -> RunConfig:
    return RunConfig(call_model_input_filter=fit_model_input, tracing_disabled=True)


def _tool_detail(name: str, arguments: Any) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        return ""
    if name == "read_file":
        detail = str(arguments.get("path") or "")
        start = arguments.get("start_line") or 0
        end = arguments.get("end_line") or 0
        if start and end:
            return f"{detail}:{start}-{end}"
        if start and int(start) > 1:
            return f"{detail}:{start}"
        return detail
    if name == "write_file":
        return str(arguments.get("path") or "")
    if name == "run_command":
        return str(arguments.get("command") or "")
    return ""


def _call_arguments(raw_item: Any) -> Any:
    if isinstance(raw_item, dict):
        return raw_item.get("arguments")
    return getattr(raw_item, "arguments", None)


def _exit_code(output: str) -> int | None:
    for line in output.splitlines():
        if line.startswith("exit_code="):
            try:
                return int(line.split("=", 1)[1].strip())
            except ValueError:
                return None
    return None


def _verification_evidence(command: str, output: str) -> VerificationEvidence:
    exit_code = _exit_code(output)
    return VerificationEvidence(
        command=command,
        exit_code=exit_code,
        stdout_available="stdout:\n" in output,
        stderr_available="stderr:\n" in output,
        success=exit_code == 0 and not output.startswith("Error:"),
        timestamp=datetime.now(UTC).isoformat(),
    )


def _failure_details(exc: BaseException) -> tuple[str, str]:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else "request failed"
    lowered = text.lower()
    if "context" in lowered or "exceed_context" in lowered:
        return "context", "context budget exceeded; the session is still open. Request a smaller file range."
    if "connection" in lowered or "tunnel" in lowered:
        return "connection", "connection to the configured model endpoint failed"
    return "runtime", "model request failed"


class HansRuntime:
    """Runs the normal Agents SDK flow and exposes only HANS events to callers."""

    def __init__(
        self,
        workspace: str | None = None,
        *,
        agent: Any | None = None,
        session: Any | None = None,
        runner: Any = Runner,
    ) -> None:
        set_tracing_disabled(True)
        self._agent = agent if agent is not None else create_agent(workspace)
        self._session = session if session is not None else SQLiteSession("hans-tui")
        self._owns_session = session is None
        self._runner = runner
        self._active_result: Any | None = None
        self._cancel_requested = False
        self._tool_calls: dict[str, _ToolCall] = {}
        self._evidence: VerificationEvidence | None = None
        self._assistant_text: list[str] = []

    def cancel_active(self) -> None:
        self._cancel_requested = True
        if self._active_result is not None:
            self._active_result.cancel()

    def close(self) -> None:
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    async def submit(self, message: str) -> AsyncIterator[HansEvent]:
        if self._active_result is not None:
            raise RuntimeError("a request is already active")
        self._cancel_requested = False
        self._tool_calls = {}
        self._evidence = None
        self._assistant_text = []
        yield UserMessageSubmitted(message)
        yield RequestStarted(message)
        try:
            result = self._runner.run_streamed(
                self._agent,
                message,
                session=self._session,
                run_config=run_config(),
            )
            self._active_result = result
            async for stream_event in result.stream_events():
                for event in self.translate_stream_event(stream_event):
                    yield event
            if self._cancel_requested:
                yield RequestCancelled()
            else:
                yield AssistantMessageComplete("".join(self._assistant_text))
                yield RequestCompleted(self._evidence)
                yield ConnectionChanged(True)
        except asyncio.CancelledError:
            self.cancel_active()
            yield RequestCancelled()
        except Exception as exc:
            if self._cancel_requested:
                yield RequestCancelled()
            else:
                category, detail = _failure_details(exc)
                yield RequestFailed(category, detail)
                yield ConnectionChanged(False)
        finally:
            self._active_result = None

    def translate_stream_event(self, stream_event: Any) -> Iterable[HansEvent]:
        if getattr(stream_event, "type", None) == "raw_response_event":
            data = getattr(stream_event, "data", None)
            if getattr(data, "type", None) in {"response.output_text.delta", "output_text.delta"}:
                delta = getattr(data, "delta", "") or ""
                if delta:
                    self._assistant_text.append(str(delta))
                    return (AssistantMessageDelta(str(delta)),)
            return ()
        if getattr(stream_event, "type", None) != "run_item_stream_event":
            return ()
        item = getattr(stream_event, "item", None)
        name = getattr(stream_event, "name", None)
        if name == "tool_called":
            call_id = getattr(item, "call_id", None)
            tool_name = getattr(item, "tool_name", None)
            if not isinstance(call_id, str) or not call_id or not isinstance(tool_name, str) or not tool_name:
                return ()
            detail = _tool_detail(tool_name, _call_arguments(getattr(item, "raw_item", None)))
            self._tool_calls[call_id] = _ToolCall(tool_name, detail)
            started: list[HansEvent] = [ToolStarted(call_id, tool_name, detail)]
            if tool_name == "run_command":
                started.append(VerificationStarted(call_id, detail))
            return tuple(started)
        if name != "tool_output":
            return ()
        call_id = getattr(item, "call_id", None)
        if not isinstance(call_id, str) or not call_id:
            return ()
        output = getattr(item, "output", "")
        rendered_output = output if isinstance(output, str) else str(output)
        events: list[HansEvent] = [ToolOutput(call_id, rendered_output)]
        call = self._tool_calls.pop(call_id, None)
        if call is None:
            return tuple(events)
        if call.name == "run_command":
            evidence = _verification_evidence(call.detail, rendered_output)
            self._evidence = evidence
            events.append(
                ToolCompleted(call_id, call.name, call.detail, evidence.success, evidence.exit_code)
            )
            if evidence.success:
                events.append(VerificationPassed(call_id, evidence))
            else:
                events.append(VerificationFailed(call_id, evidence))
            return tuple(events)
        success = not rendered_output.startswith("Error:")
        if call.name == "write_file" and success:
            self._evidence = None
        events.append(ToolCompleted(call_id, call.name, call.detail, success))
        return tuple(events)
