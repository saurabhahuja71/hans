"""Agents SDK integration and translation into HANS semantic events."""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass, is_dataclass, replace
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Callable, Iterable
from uuid import uuid4

from agents import Runner, SQLiteSession, set_tracing_disabled
from agents.model_settings import Reasoning
from agents.run_config import RunConfig

from bolt_next.agent import apply_tool_permission_policy, create_agent
from bolt_next.context_budget import fit_model_input, make_fit_model_input
from bolt_next.errors import ConfigurationError, FailureCategory
from bolt_next.model_catalog import (
    ConfiguredModelProfile,
    ModelInfo,
    configured_model_profile,
    configured_model_profiles,
)
from bolt_next.workspace import TaskMutationJournal, resolve_workspace
from bolt_next.events import (
    AssistantMessageComplete,
    AssistantMessageDelta,
    ConnectionChanged,
    HansEvent,
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
    VerificationEvidence,
    VerificationFailed,
    VerificationPassed,
    VerificationStarted,
)


@dataclass(frozen=True, slots=True)
class _ToolCall:
    name: str
    detail: str
    purpose: str = "inspect"


def run_config(context_filter: Callable[[Any], Any] = fit_model_input) -> RunConfig:
    return RunConfig(call_model_input_filter=context_filter, tracing_disabled=True)


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
    if name in {"list_directory", "write_file", "replace_in_file"}:
        return str(arguments.get("path") or "")
    if name == "search_files":
        return f"{arguments.get('path') or '.'}: {arguments.get('query') or ''}".rstrip()
    if name == "run_command":
        return str(arguments.get("command") or "")
    return ""


def _call_arguments(raw_item: Any) -> Any:
    if isinstance(raw_item, dict):
        return raw_item.get("arguments")
    return getattr(raw_item, "arguments", None)


def _tool_purpose(name: str, arguments: Any) -> str:
    if name != "run_command":
        return "inspect"
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return "inspect"
    if not isinstance(arguments, dict):
        return "inspect"
    purpose = arguments.get("purpose")
    return purpose if purpose in {"inspect", "verify"} else "inspect"


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


_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[:=]\s*[\"']?)([^\s,;\"']+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)


def _debug_detail(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else "request failed"
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            text = pattern.sub(r"\1[REDACTED]", text)
        else:
            text = pattern.sub("[REDACTED]", text)
    return text


def _status_code(exc: BaseException) -> int | None:
    for name in ("status_code", "status"):
        value = getattr(exc, name, None)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _failure_details(exc: BaseException) -> tuple[FailureCategory, str, str]:
    text = _debug_detail(exc)
    lowered = text.lower()
    type_name = type(exc).__name__.lower()
    status = _status_code(exc)
    if isinstance(exc, ConfigurationError) or "bolt_model_" in lowered:
        return FailureCategory.CONFIGURATION, text, text
    if status in {401, 403} or "auth" in type_name or "unauthorized" in lowered or "forbidden" in lowered:
        return FailureCategory.AUTHENTICATION, "authentication with the configured model endpoint failed", text
    if "context" in lowered or "token limit" in lowered or "exceed_context" in lowered:
        return (
            FailureCategory.CONTEXT,
            "context budget exceeded; the session is still open. Request a smaller file range.",
            text,
        )
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)) or "timeout" in type_name or "connection" in type_name:
        return FailureCategory.CONNECTION, "connection to the configured model endpoint failed", text
    if "connection" in lowered or "tunnel" in lowered or "timed out" in lowered:
        return FailureCategory.CONNECTION, "connection to the configured model endpoint failed", text
    if "tool" in type_name or "tool" in lowered:
        return FailureCategory.TOOL, "a tool operation failed", text
    if status is not None or "request" in type_name or "model" in type_name:
        return FailureCategory.MODEL, "the model request failed", text
    return FailureCategory.RUNTIME, "an internal runtime error occurred", text


class HansRuntime:
    """Runs the normal Agents SDK flow and exposes only HANS events to callers."""

    def __init__(
        self,
        workspace: str | None = None,
        *,
        agent: Any | None = None,
        session: Any | None = None,
        runner: Any = Runner,
        agent_factory: Callable[..., Any] | None = None,
        session_factory: Callable[[str], Any] | None = None,
    ) -> None:
        set_tracing_disabled(True)
        # Build the agent on the first request so the terminal UI can still
        # open when model configuration is missing. The resulting startup
        # error is then rendered as a normal request failure instead of a
        # Python traceback before the user sees HANS.
        self._agent = agent
        self._workspace = resolve_workspace(workspace or os.environ.get("BOLT_WORKSPACE"))
        self._journal = TaskMutationJournal(self._workspace)
        self._agent_factory = agent_factory or create_agent
        self._session_factory = session_factory or SQLiteSession
        self._session = session if session is not None else self._session_factory("hans-tui")
        self._owns_session = session is None
        self._runner = runner
        self._permissions = {"read": True, "write": True, "execute": True}
        self._request_active = False
        self._active_result: Any | None = None
        self._stream_waiter: asyncio.Future[Any] | None = None
        self._cancel_requested = False
        self._tool_calls: dict[str, _ToolCall] = {}
        self._evidence: VerificationEvidence | None = None
        self._assistant_text: list[str] = []
        self._profile: ConfiguredModelProfile | None = None
        self._model_info: ModelInfo | None = None
        self._context_filter: Callable[[Any], Any] = fit_model_input
        self._reasoning_mode_override: str | None = None
        self._base_model_settings: Any | None = None
        if self._agent is not None:
            apply_tool_permission_policy(self._agent, self._permission_allowed)
            self._snapshot_model_settings()

    def get_control_status(self) -> RuntimeControlStatus:
        return RuntimeControlStatus(
            read_allowed=self._permissions["read"],
            write_allowed=self._permissions["write"],
            execute_allowed=self._permissions["execute"],
        )

    def _active_profile(self) -> ConfiguredModelProfile:
        if self._profile is None:
            profile = configured_model_profile()
            self._profile = profile
            self._model_info = profile.info
            self._context_filter = make_fit_model_input(profile.info.context_tokens)
        return self._profile

    def _active_model_info(self) -> ModelInfo:
        return self._active_profile().info

    @staticmethod
    def _model_settings_snapshot(agent: Any) -> Any | None:
        settings = getattr(agent, "model_settings", None)
        return settings if is_dataclass(settings) else None

    def _snapshot_model_settings(self) -> None:
        self._base_model_settings = self._model_settings_snapshot(self._agent)

    def _configured_reasoning_mode(self) -> str | None:
        if self._base_model_settings is not None:
            reasoning = getattr(self._base_model_settings, "reasoning", None)
            effort = getattr(reasoning, "effort", None)
            if isinstance(effort, str) and effort:
                return effort
        return self._active_profile().reasoning_effort

    def _apply_effective_model_settings(self) -> None:
        if self._base_model_settings is None:
            return
        model_info = self._active_model_info()
        if self._reasoning_mode_override is None:
            effective_settings = self._base_model_settings
        elif self._reasoning_mode_override == "none" and model_info.none_semantics == "omit":
            effective_settings = replace(self._base_model_settings, reasoning=None)
        else:
            effective_settings = replace(
                self._base_model_settings,
                reasoning=Reasoning(effort=self._reasoning_mode_override),
            )
        self._agent.model_settings = effective_settings

    def _new_session_id(self) -> str:
        return f"hans-tui-{uuid4().hex}"

    def select_model(self, model_id: str) -> ModelChanged | RuntimeControlRejected | None:
        if self._request_active:
            return RuntimeControlRejected("Model selection is available when HANS is idle.")
        normalized_id = model_id.strip().lower()
        if not normalized_id or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,127}", normalized_id):
            return RuntimeControlRejected("Unknown configured model.")
        try:
            profiles = configured_model_profiles()
            target_profile = next(
                (profile for profile in profiles if profile.info.id.lower() == normalized_id),
                None,
            )
            current_profile = self._profile or configured_model_profile()
        except ConfigurationError:
            return RuntimeControlRejected("Model selection is unavailable.")
        if target_profile is None:
            return RuntimeControlRejected(f"Unknown configured model: {normalized_id}")
        if target_profile.info.id == current_profile.info.id:
            return None

        target_session: Any | None = None
        try:
            target_agent = self._agent_factory(self._workspace, journal=self._journal, profile=target_profile)
            if target_agent is None:
                raise RuntimeError("target agent is unavailable")
            apply_tool_permission_policy(target_agent, self._permission_allowed)
            target_base_settings = self._model_settings_snapshot(target_agent)
            target_context_filter = make_fit_model_input(target_profile.info.context_tokens)
            target_session = self._session_factory(self._new_session_id())
            if target_session is None:
                raise RuntimeError("target session is unavailable")
        except Exception:
            if target_session is not None:
                try:
                    target_session.close()
                except Exception:
                    pass
            return RuntimeControlRejected("Unable to start the selected model.")

        previous_model_id = current_profile.info.id
        previous_session = self._session
        previous_session_owned = self._owns_session
        self._profile = target_profile
        self._model_info = target_profile.info
        self._agent = target_agent
        self._session = target_session
        self._owns_session = True
        self._base_model_settings = target_base_settings
        self._context_filter = target_context_filter
        self._reasoning_mode_override = None
        if previous_session_owned and previous_session is not None:
            try:
                previous_session.close()
            except Exception:
                pass
        return ModelChanged(previous_model_id, target_profile.info, True, True)

    def get_model_status(self) -> ModelStatus:
        return ModelStatus(
            self._active_model_info(),
            self._reasoning_mode_override or self._configured_reasoning_mode(),
            tuple(profile.info for profile in configured_model_profiles()),
        )

    def get_reasoning_mode_status(self) -> ReasoningModeStatus:
        model_info = self._active_model_info()
        mode = self._reasoning_mode_override or self._configured_reasoning_mode()
        return ReasoningModeStatus(mode, model_info.supported_reasoning_modes, model_info.none_semantics)

    def set_reasoning_mode(self, mode: str | None) -> ReasoningModeChanged | RuntimeControlRejected:
        if self._request_active:
            return RuntimeControlRejected("Reasoning mode can be changed when HANS is idle.")
        if mode is None:
            self._reasoning_mode_override = None
            self._apply_effective_model_settings()
            return ReasoningModeChanged(None)
        normalized_mode = mode.strip().lower()
        model_info = self._active_model_info()
        supported = model_info.supported_reasoning_modes
        if not supported:
            return RuntimeControlRejected("Reasoning mode support is not declared for the active configured model.")
        if not normalized_mode or normalized_mode not in supported:
            return RuntimeControlRejected(
                f"Unsupported reasoning mode: {normalized_mode or mode}. Supported modes: {', '.join(supported)}."
            )
        self._reasoning_mode_override = normalized_mode
        self._apply_effective_model_settings()
        return ReasoningModeChanged(normalized_mode)

    def set_permission(self, category: str, allowed: bool) -> PermissionPolicyChanged | RuntimeControlRejected:
        if category not in self._permissions:
            raise ValueError(f"Unknown permission: {category}")
        if self._request_active:
            return RuntimeControlRejected("Permission changes are available when HANS is idle.")
        self._permissions[category] = allowed
        return PermissionPolicyChanged(category, allowed)

    async def clear_session_history(self) -> SessionCleared | RuntimeControlRejected:
        if self._request_active:
            return RuntimeControlRejected("Cannot clear the session while HANS is busy.")
        if self._session is None:
            return RuntimeControlRejected("The session is unavailable.")
        await self._session.clear_session()
        return SessionCleared()

    def _permission_allowed(self, category: str) -> bool:
        return self._permissions[category]

    def cancel_active(self) -> None:
        self._cancel_requested = True
        if self._active_result is not None:
            self._active_result.cancel()
        if self._stream_waiter is not None:
            self._stream_waiter.cancel()

    def close(self) -> None:
        if self._owns_session and self._session is not None:
            self._session.close()
            self._session = None

    def task_diff(self, *, max_chars: int | None = None) -> TaskDiff:
        return TaskDiff(self._journal.unified_diff(max_chars=max_chars))

    def undo_task(self) -> TaskUndoSucceeded | TaskUndoRefused:
        outcome = self._journal.undo()
        conflicts = tuple(outcome["conflicts"])
        if conflicts:
            return TaskUndoRefused(conflicts)
        restored = tuple(outcome["restored"])
        removed = tuple(outcome["removed"])
        if restored or removed:
            self._evidence = None
        return TaskUndoSucceeded(restored, removed)

    def _task_change_summary(self) -> TaskChangeSummary | None:
        summary = self._journal.summary()
        if not summary["changed_files"]:
            return None
        return TaskChangeSummary(self._journal.compact_summary())

    async def submit(self, message: str) -> AsyncIterator[HansEvent]:
        if self._request_active:
            raise RuntimeError("a request is already active")
        self._request_active = True
        try:
            self._cancel_requested = False
            self._journal.begin_task()
            self._tool_calls = {}
            self._evidence = None
            self._assistant_text = []
            yield UserMessageSubmitted(message)
            yield RequestStarted(message)
            profile = self._active_profile()
            if self._agent is None:
                self._agent = self._agent_factory(self._workspace, journal=self._journal, profile=profile)
                apply_tool_permission_policy(self._agent, self._permission_allowed)
                self._snapshot_model_settings()
            self._apply_effective_model_settings()
            result = self._runner.run_streamed(
                self._agent,
                message,
                session=self._session,
                run_config=run_config(self._context_filter),
            )
            self._active_result = result
            stream = result.stream_events().__aiter__()
            while True:
                self._stream_waiter = asyncio.ensure_future(anext(stream))
                try:
                    stream_event = await self._stream_waiter
                except StopAsyncIteration:
                    break
                finally:
                    self._stream_waiter = None
                for event in self.translate_stream_event(stream_event):
                    yield event
            if self._cancel_requested:
                yield RequestCancelled()
                summary = self._task_change_summary()
                if summary is not None:
                    yield summary
            else:
                yield AssistantMessageComplete("".join(self._assistant_text))
                yield RequestCompleted(self._evidence)
                summary = self._task_change_summary()
                if summary is not None:
                    yield summary
                yield ConnectionChanged(True)
        except asyncio.CancelledError:
            self.cancel_active()
            yield RequestCancelled()
            summary = self._task_change_summary()
            if summary is not None:
                yield summary
        except Exception as exc:
            if self._cancel_requested:
                yield RequestCancelled()
                summary = self._task_change_summary()
                if summary is not None:
                    yield summary
            else:
                category, detail, debug_message = _failure_details(exc)
                yield RequestFailed(category, detail, debug_message)
                summary = self._task_change_summary()
                if summary is not None:
                    yield summary
                yield ConnectionChanged(False)
        finally:
            self._active_result = None
            self._request_active = False

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
            arguments = _call_arguments(getattr(item, "raw_item", None))
            detail = _tool_detail(tool_name, arguments)
            purpose = _tool_purpose(tool_name, arguments)
            self._tool_calls[call_id] = _ToolCall(tool_name, detail, purpose)
            started: list[HansEvent] = [ToolStarted(call_id, tool_name, detail, purpose)]
            if tool_name == "run_command" and purpose == "verify":
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
            events.append(
                ToolCompleted(call_id, call.name, call.detail, evidence.success, evidence.exit_code)
            )
            if call.purpose == "verify":
                self._evidence = evidence
                if evidence.success:
                    events.append(VerificationPassed(call_id, evidence))
                else:
                    events.append(VerificationFailed(call_id, evidence))
            return tuple(events)
        success = not rendered_output.startswith(("Error:", "Permission denied:"))
        if call.name in {"write_file", "replace_in_file"} and success:
            self._evidence = None
        events.append(ToolCompleted(call_id, call.name, call.detail, success))
        return tuple(events)
