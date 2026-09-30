"""Agents SDK integration and translation into HANS semantic events."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, is_dataclass, replace
from uuid import uuid4
from datetime import UTC, datetime
from typing import Any, AsyncIterator, Callable, Iterable

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
from bolt_next.workspace import ExternalPathAuthorizer, TaskMutationJournal, resolve_workspace
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
    ToolApprovalDisplay,
    ToolApprovalRequested,
    ToolApprovalResolved,
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


@dataclass(slots=True)
class _PendingApproval:
    request_id: str
    call_id: str
    item: Any
    state: Any
    tool_name: str
    category: str
    display: ToolApprovalDisplay
    request_generation: int
    agent: Any
    session: Any
    model_generation: int
    session_generation: int
    external: bool = False
    external_path: str | None = None
    external_call_id: str | None = None


_TOOL_CATEGORIES = {
    "list_directory": "read",
    "search_files": "read",
    "read_file": "read",
    "read_image": "read",
    "write_file": "write",
    "replace_in_file": "write",
    "run_command": "execute",
}
_APPROVAL_DISPLAY_MAX_CHARS = 240
_APPROVAL_MAX_LINE = 1_000_000


def run_config(context_filter: Callable[[Any], Any] = fit_model_input) -> RunConfig:
    return RunConfig(call_model_input_filter=context_filter, tracing_disabled=True)


def _tool_detail(name: str, arguments: Any) -> str:
    arguments = _argument_mapping(arguments)
    if name == "read_file":
        detail = str(arguments.get("path") or "")
        start = arguments.get("start_line") or 0
        end = arguments.get("end_line") or 0
        if start and end:
            detail = f"{detail}:{start}-{end}"
        elif start and int(start) > 1:
            detail = f"{detail}:{start}"
        return _bounded_tool_text(detail)
    if name == "read_image":
        return _bounded_tool_text(arguments.get("path"))
    if name in {"list_directory", "write_file", "replace_in_file"}:
        return _bounded_tool_text(arguments.get("path"))
    if name == "search_files":
        return _bounded_tool_text(f"{arguments.get('path') or '.'}: {arguments.get('query') or ''}".rstrip())
    if name == "run_command":
        return _bounded_tool_text(arguments.get("command"))
    return ""


def _member(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _call_arguments(item: Any) -> Any:
    sources = (item, _member(item, "raw_item"), _member(item, "item"))
    for source in sources:
        for name in ("arguments", "params", "input"):
            value = _member(source, name)
            if value is not None:
                return value
    return None


def _argument_mapping(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _approval_call_id(item: Any, request_generation: int, index: int) -> str:
    sources = (item, _member(item, "raw_item"), _member(item, "item"))
    for source in sources:
        for name in ("call_id", "tool_call_id", "id"):
            candidate = _member(source, name)
            if isinstance(candidate, str) and candidate.strip():
                return candidate
    return f"missing-call-{request_generation}-{index}"


def _sdk_approval_call_id(item: Any) -> str | None:
    candidate = _member(item, "call_id")
    return candidate if isinstance(candidate, str) and candidate.strip() else None


def _redact_approval_text(value: str) -> str:
    return _redact_sensitive_text(value)


def _bounded_approval_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = _redact_approval_text(" ".join(value.split()))
    return value if len(value) <= _APPROVAL_DISPLAY_MAX_CHARS else f"{value[:_APPROVAL_DISPLAY_MAX_CHARS - 1]}…"


def _safe_path(arguments: dict[str, Any]) -> str:
    return _bounded_approval_text(arguments.get("path")) or "."


def _safe_read_range(arguments: dict[str, Any]) -> str | None:
    start = arguments.get("start_line", 1)
    end = arguments.get("end_line", 0)
    if isinstance(start, bool) or not isinstance(start, int) or not 1 <= start <= _APPROVAL_MAX_LINE:
        return None
    if isinstance(end, bool) or not isinstance(end, int) or not 0 <= end <= _APPROVAL_MAX_LINE:
        return None
    return f"{start}+" if end == 0 else f"{start}-{end}"


def _approval_display(tool_name: str, arguments: Any) -> ToolApprovalDisplay:
    values = _argument_mapping(arguments)
    if tool_name in {"write_file", "replace_in_file"}:
        return ToolApprovalDisplay((("path", _safe_path(values)),))
    if tool_name == "read_file":
        fields: list[tuple[str, str]] = [("path", _safe_path(values))]
        safe_range = _safe_read_range(values)
        if safe_range is not None:
            fields.append(("range", safe_range))
        return ToolApprovalDisplay(tuple(fields))
    if tool_name == "read_image":
        return ToolApprovalDisplay((("path", _safe_path(values)),))
    if tool_name == "list_directory":
        return ToolApprovalDisplay((("path", _safe_path(values)),))
    if tool_name == "search_files":
        fields = [("path", _safe_path(values))]
        query = _bounded_approval_text(values.get("query"))
        if query:
            fields.append(("query", query))
        return ToolApprovalDisplay(tuple(fields))
    if tool_name == "run_command":
        fields = []
        command = _bounded_approval_text(values.get("command"))
        if command:
            fields.append(("command", command))
        fields.append(("purpose", _tool_purpose(tool_name, values)))
        return ToolApprovalDisplay(tuple(fields))
    return ToolApprovalDisplay()


def _tool_purpose(name: str, arguments: Any) -> str:
    if name != "run_command":
        return "inspect"
    purpose = _argument_mapping(arguments).get("purpose")
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


_TOOL_DETAIL_MAX_CHARS = 240
_TOOL_OUTPUT_MAX_CHARS = 4_000
_SECRET_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)([^\s,;]+)"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password)\s*[:=]\s*[\"']?)([^\s,;\"']+)"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
)
_URL_USERINFO_PATTERN = re.compile(r"([A-Za-z][A-Za-z0-9+.-]*://)([^\s/@:]+):[^\s/@]+@")
_IMAGE_DATA_URL_PATTERN = re.compile(r"^data:([a-z0-9.+-]+/[a-z0-9.+-]+);base64,", re.IGNORECASE)


def _redact_sensitive_text(value: str) -> str:
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            value = pattern.sub(r"\1[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED]", value)
    return _URL_USERINFO_PATTERN.sub(r"\1[REDACTED]@", value)


def _bounded_tool_text(value: Any, limit: int = _TOOL_DETAIL_MAX_CHARS) -> str:
    text = _redact_sensitive_text(" ".join(str(value or "").split()))
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _rendered_image_output(value: str) -> str | None:
    match = _IMAGE_DATA_URL_PATTERN.match(value)
    if match is None:
        return None
    return f"Image loaded for model input ({match.group(1).lower()})."


def _render_tool_output(value: Any, tool_name: str | None = None) -> str:
    if isinstance(value, str):
        return _rendered_image_output(value) or value
    image_url = _member(value, "image_url")
    if isinstance(image_url, str):
        return _rendered_image_output(image_url) or "Image loaded for model input."
    if tool_name == "read_image":
        return "Image loaded for model input."
    return str(value)


def _safe_tool_output(value: str, *, failed: bool) -> str:
    if failed and "traceback (most recent call last):" in value.lower():
        return "Tool diagnostic omitted."
    value = _redact_sensitive_text(value)
    return value if len(value) <= _TOOL_OUTPUT_MAX_CHARS else f"{value[:_TOOL_OUTPUT_MAX_CHARS - 1]}…"


_UNKNOWN_TOOL_FAILURE = "The tool operation failed without a detailed diagnostic."


def _failure_reason_text(value: str) -> str:
    reason = _bounded_tool_text(value)
    if not reason:
        return _UNKNOWN_TOOL_FAILURE
    if reason[-1] not in ".!?":
        reason += "."
    return reason[0].upper() + reason[1:]


def _workspace_failure_reason(name: str, output: str) -> str | None:
    prefixes = {
        "read_file": "Error reading",
        "read_image": "Error reading image",
        "write_file": "Error writing",
        "replace_in_file": "Error replacing",
        "list_directory": "Error listing",
        "search_files": "Error searching",
    }
    prefix = prefixes.get(name)
    if prefix is None:
        return None
    if output.startswith(("Error: file does not exist:", "Error: image file does not exist:")):
        return "File does not exist."
    if output.startswith(f"{prefix}:"):
        return _failure_reason_text(output[len(prefix) + 1 :])
    pattern = re.compile(rf"^{re.escape(prefix)}\s+.+?:\s*(.+)$", re.DOTALL)
    match = pattern.match(output)
    if match is not None:
        return _failure_reason_text(match.group(1))
    if output.startswith("Error:"):
        return _failure_reason_text(output.removeprefix("Error:"))
    if output.startswith(prefix):
        return _UNKNOWN_TOOL_FAILURE
    return None


def _tool_failure_reason(name: str, output: str, exit_code: int | None = None) -> str | None:
    if "traceback (most recent call last):" in output.lower():
        return _UNKNOWN_TOOL_FAILURE
    if output.startswith("Permission denied:"):
        return "Permission denied."
    if name == "run_command":
        if exit_code is not None:
            return None if exit_code == 0 else f"Command exited with code {exit_code}."
        for prefix in ("Error running command:", "Error:"):
            if output.startswith(prefix):
                return _failure_reason_text(output.removeprefix(prefix))
        return _UNKNOWN_TOOL_FAILURE
    return _workspace_failure_reason(name, output)


def _debug_detail(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else "request failed"
    return _redact_sensitive_text(text)


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
        interactive: Callable[[], bool] | None = None,
    ) -> None:
        set_tracing_disabled(True)
        # Build the agent on the first request so the terminal UI can still
        # open when model configuration is missing. The resulting startup
        # error is then rendered as a normal request failure instead of a
        # Python traceback before the user sees HANS.
        self._agent = agent
        self._workspace = resolve_workspace(workspace or os.environ.get("BOLT_WORKSPACE"))
        self._permissions = {"read": "allow", "write": "allow", "execute": "allow", "external": "allow"}
        self._external_path_authorizer = ExternalPathAuthorizer(
            self._workspace,
            get_external_policy=self._external_permission_policy,
        )
        self._journal = TaskMutationJournal(self._workspace)
        self._agent_factory = agent_factory or create_agent
        self._session_factory = session_factory or SQLiteSession
        self._session = session if session is not None else self._session_factory("hans-tui")
        self._owns_session = session is None
        self._runner = runner
        self._permission_toggle_snapshot: dict[str, str] | None = None
        self._interactive = interactive or (lambda: sys.stdin.isatty() and sys.stdout.isatty())
        self._request_active = False
        self._active_result: Any | None = None
        self._stream_waiter: asyncio.Future[Any] | None = None
        self._cancel_requested = False
        self._request_generation = 0
        self._model_generation = 0
        self._session_generation = 0
        self._pending_approvals: list[_PendingApproval] = []
        self._tool_calls: dict[str, _ToolCall] = {}
        self._tool_failure: str | None = None
        self._evidence: VerificationEvidence | None = None
        self._assistant_text: list[str] = []
        self._profile: ConfiguredModelProfile | None = None
        self._model_info: ModelInfo | None = None
        self._context_filter: Callable[[Any], Any] = fit_model_input
        self._reasoning_mode_override: str | None = None
        self._base_model_settings: Any | None = None
        if self._agent is not None:
            apply_tool_permission_policy(
                self._agent,
                self._permission_allowed,
                authorizer=self._external_path_authorizer,
                get_external_policy=self._external_permission_policy,
            )
            self._snapshot_model_settings()

    def get_control_status(self) -> RuntimeControlStatus:
        return RuntimeControlStatus(
            read_policy=self._permissions["read"],
            write_policy=self._permissions["write"],
            execute_policy=self._permissions["execute"],
            external_policy=self._permissions["external"],
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
            target_agent = self._agent_factory(
                self._workspace,
                journal=self._journal,
                profile=target_profile,
                authorizer=self._external_path_authorizer,
            )
            if target_agent is None:
                raise RuntimeError("target agent is unavailable")
            apply_tool_permission_policy(
                target_agent,
                self._permission_allowed,
                authorizer=self._external_path_authorizer,
                get_external_policy=self._external_permission_policy,
            )
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
        self._external_path_authorizer.clear()
        previous_session = self._session
        previous_session_owned = self._owns_session
        self._profile = target_profile
        self._model_info = target_profile.info
        self._agent = target_agent
        self._session = target_session
        self._owns_session = True
        self._model_generation += 1
        self._session_generation += 1
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

    def toggle_permissions(self) -> RuntimeControlStatus | RuntimeControlRejected:
        if self._request_active:
            return RuntimeControlRejected("Permission changes are available when HANS is idle.")
        if self._permission_toggle_snapshot is None:
            self._permission_toggle_snapshot = dict(self._permissions)
            self._permissions.update({category: "allow" for category in self._permissions})
        else:
            self._permissions.update(self._permission_toggle_snapshot)
            self._permission_toggle_snapshot = None
        return self.get_control_status()

    def set_permission(self, category: str, policy: str) -> PermissionPolicyChanged | RuntimeControlRejected:
        if category not in self._permissions:
            raise ValueError(f"Unknown permission: {category}")
        normalized_policy = policy.strip().lower()
        if normalized_policy not in {"allow", "deny", "ask"}:
            raise ValueError(f"Unknown permission policy: {policy}")
        if self._request_active:
            return RuntimeControlRejected("Permission changes are available when HANS is idle.")
        self._permissions[category] = normalized_policy
        self._permission_toggle_snapshot = None
        return PermissionPolicyChanged(category, normalized_policy)

    async def clear_session_history(self) -> SessionCleared | RuntimeControlRejected:
        if self._request_active:
            return RuntimeControlRejected("Cannot clear the session while HANS is busy.")
        if self._session is None:
            return RuntimeControlRejected("The session is unavailable.")
        await self._session.clear_session()
        self._external_path_authorizer.clear()
        self._session_generation += 1
        return SessionCleared()

    def _permission_allowed(self, category: str) -> str:
        return self._permissions[category]

    def _external_permission_policy(self) -> str:
        return self._permissions["external"]

    def _invalidate_pending_approvals(self) -> None:
        self._pending_approvals.clear()
        self._external_path_authorizer.clear()

    def cancel_active(self) -> RequestCancelled | None:
        if not self._request_active:
            return None
        was_paused = bool(self._pending_approvals)
        self._cancel_requested = True
        self._invalidate_pending_approvals()
        if self._active_result is not None:
            self._active_result.cancel()
        if self._stream_waiter is not None:
            self._stream_waiter.cancel()
        if was_paused:
            self._active_result = None
            self._request_active = False
            return RequestCancelled()
        return None

    def close(self) -> None:
        self.cancel_active()
        self._invalidate_pending_approvals()
        self._session_generation += 1
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

    def _record_interruptions(self, result: Any) -> ToolApprovalRequested | None:
        interruptions = tuple(getattr(result, "interruptions", ()))
        if not interruptions:
            return None
        state = result.to_state()
        request_id = f"request-{self._request_generation}"
        pending: list[_PendingApproval] = []
        for index, item in enumerate(interruptions, start=1):
            raw_item = _member(item, "raw_item")
            tool_name = _member(item, "tool_name") or _member(item, "name") or _member(raw_item, "name") or "tool"
            tool_name = tool_name if isinstance(tool_name, str) and tool_name else "tool"
            arguments = _call_arguments(item)
            call_id = _approval_call_id(item, self._request_generation, index)
            sdk_call_id = _sdk_approval_call_id(item)
            access = (
                self._external_path_authorizer.proposal_for(tool_name, sdk_call_id)
                if sdk_call_id is not None
                else None
            )
            if access is not None:
                call_id = sdk_call_id
            external_path = _bounded_approval_text(access.display_path) if access is not None else None
            display = _approval_display(tool_name, arguments)
            if access is not None:
                resolved_path = _bounded_approval_text(str(access.path))
                display_fields = tuple(
                    (name, external_path if name == "path" else value)
                    for name, value in display.fields
                )
                if resolved_path and resolved_path != external_path:
                    display_fields += (("resolved", resolved_path),)
                display = ToolApprovalDisplay(display_fields + (("scope", "outside workspace"),))
            pending.append(
                _PendingApproval(
                    request_id=request_id,
                    call_id=call_id,
                    item=item,
                    state=state,
                    tool_name=tool_name,
                    category=_TOOL_CATEGORIES.get(tool_name, "execute"),
                    display=display,
                    request_generation=self._request_generation,
                    agent=self._agent,
                    session=self._session,
                    model_generation=self._model_generation,
                    session_generation=self._session_generation,
                    external=access is not None,
                    external_path=external_path,
                    external_call_id=sdk_call_id if access is not None else None,
                )
            )
        self._pending_approvals = pending
        return self._requested_approval_event(pending[0])

    @staticmethod
    def _requested_approval_event(pending: _PendingApproval) -> ToolApprovalRequested:
        return ToolApprovalRequested(
            pending.request_id,
            pending.call_id,
            pending.tool_name,
            pending.category,
            pending.display,
            pending.external,
            pending.external_path,
        )

    def _approval_context_is_current(self, pending: _PendingApproval) -> bool:
        return (
            self._request_active
            and not self._cancel_requested
            and pending.request_generation == self._request_generation
            and pending.agent is self._agent
            and pending.session is self._session
            and pending.model_generation == self._model_generation
            and pending.session_generation == self._session_generation
        )

    def _approval_is_current(self, pending: _PendingApproval, request_id: str, call_id: str) -> bool:
        return (
            bool(self._pending_approvals)
            and self._pending_approvals[0] is pending
            and pending.request_id == request_id
            and pending.call_id == call_id
            and self._approval_context_is_current(pending)
        )

    async def _consume_result(self, result: Any) -> AsyncIterator[HansEvent]:
        while True:
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
                return
            approval_event = self._record_interruptions(result)
            if approval_event is None:
                return
            if self._interactive():
                self._active_result = None
                yield approval_event
                return
            for pending in self._pending_approvals:
                self._external_path_authorizer.revoke(
                    pending.tool_name, pending.external_call_id or pending.call_id
                )
                pending.state.reject(pending.item)
            state = self._pending_approvals[0].state
            self._invalidate_pending_approvals()
            result = self._runner.run_streamed(
                self._agent,
                state,
                session=self._session,
                run_config=run_config(self._context_filter),
            )

    async def _terminal_events(self) -> AsyncIterator[HansEvent]:
        if self._cancel_requested:
            yield RequestCancelled()
        elif self._tool_failure is not None:
            yield RequestFailed(FailureCategory.TOOL, self._tool_failure, self._tool_failure)
            yield ConnectionChanged(True)
        else:
            yield AssistantMessageComplete("".join(self._assistant_text))
            yield RequestCompleted(self._evidence)
            yield ConnectionChanged(True)
        summary = self._task_change_summary()
        if summary is not None:
            yield summary

    async def _failure_events(self, exc: BaseException) -> AsyncIterator[HansEvent]:
        if self._cancel_requested:
            yield RequestCancelled()
        else:
            category, detail, debug_message = _failure_details(exc)
            yield RequestFailed(category, detail, debug_message)
            yield ConnectionChanged(False)
        summary = self._task_change_summary()
        if summary is not None:
            yield summary

    def _finish_request(self) -> None:
        self._active_result = None
        self._stream_waiter = None
        self._invalidate_pending_approvals()
        self._request_active = False

    async def submit(self, message: str) -> AsyncIterator[HansEvent]:
        if self._request_active:
            raise RuntimeError("a request is already active")
        self._request_active = True
        self._request_generation += 1
        try:
            self._cancel_requested = False
            self._journal.begin_task()
            self._tool_calls = {}
            self._tool_failure = None
            self._evidence = None
            self._assistant_text = []
            yield UserMessageSubmitted(message)
            yield RequestStarted(message)
            profile = self._active_profile()
            if self._agent is None:
                self._agent = self._agent_factory(
                    self._workspace,
                    journal=self._journal,
                    profile=profile,
                    authorizer=self._external_path_authorizer,
                )
                apply_tool_permission_policy(
                    self._agent,
                    self._permission_allowed,
                    authorizer=self._external_path_authorizer,
                    get_external_policy=self._external_permission_policy,
                )
                self._snapshot_model_settings()
            self._apply_effective_model_settings()
            result = self._runner.run_streamed(
                self._agent,
                message,
                session=self._session,
                run_config=run_config(self._context_filter),
            )
            async for event in self._consume_result(result):
                yield event
            if self._pending_approvals:
                return
            async for event in self._terminal_events():
                yield event
        except asyncio.CancelledError:
            self.cancel_active()
            async for event in self._terminal_events():
                yield event
        except Exception as exc:
            async for event in self._failure_events(exc):
                yield event
        finally:
            if not self._pending_approvals:
                self._finish_request()

    async def resolve_tool_approval(
        self, request_id: str, call_id: str, approved: bool
    ) -> AsyncIterator[HansEvent]:
        if not self._pending_approvals:
            yield RuntimeControlRejected("The approval request is no longer active.")
            return
        pending = self._pending_approvals[0]
        if not self._approval_is_current(pending, request_id, call_id):
            yield RuntimeControlRejected("The approval request is stale or does not match the active request.")
            return
        try:
            granted = False
            external_call_id = pending.external_call_id or pending.call_id
            if approved:
                if pending.external:
                    if self._external_path_authorizer.approve_exact(pending.tool_name, external_call_id) is None:
                        yield RuntimeControlRejected("The approval request is stale or does not match the active request.")
                        return
                    granted = True
                try:
                    pending.state.approve(pending.item)
                except Exception:
                    if granted:
                        self._external_path_authorizer.revoke(pending.tool_name, external_call_id)
                    raise
            else:
                self._external_path_authorizer.revoke(pending.tool_name, external_call_id)
                pending.state.reject(pending.item)
            if not self._approval_is_current(pending, request_id, call_id):
                if granted:
                    self._external_path_authorizer.revoke(pending.tool_name, external_call_id)
                yield RuntimeControlRejected("The approval request is stale or does not match the active request.")
                return
            self._pending_approvals.pop(0)
            yield ToolApprovalResolved(pending.request_id, pending.call_id, approved)
            if self._pending_approvals:
                next_pending = self._pending_approvals[0]
                if not self._approval_context_is_current(next_pending):
                    yield RuntimeControlRejected("The approval request is stale or does not match the active request.")
                    return
                yield self._requested_approval_event(next_pending)
                return
            if not self._approval_context_is_current(pending):
                yield RuntimeControlRejected("The approval request is stale or does not match the active request.")
                return
            result = self._runner.run_streamed(
                pending.agent,
                pending.state,
                session=pending.session,
                run_config=run_config(self._context_filter),
            )
            async for event in self._consume_result(result):
                yield event
            if self._pending_approvals:
                return
            async for event in self._terminal_events():
                yield event
        except asyncio.CancelledError:
            self.cancel_active()
            async for event in self._terminal_events():
                yield event
        except Exception as exc:
            async for event in self._failure_events(exc):
                yield event
        finally:
            if not self._pending_approvals:
                self._finish_request()

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
        call = self._tool_calls.pop(call_id, None)
        rendered_output = _render_tool_output(output, call.name if call is not None else None)
        if call is None:
            return (
                ToolOutput(
                    call_id,
                    _safe_tool_output(
                        rendered_output,
                        failed="traceback (most recent call last):" in rendered_output.lower(),
                    ),
                ),
            )
        if call.name == "run_command":
            evidence = _verification_evidence(call.detail, rendered_output)
            failure_reason = _tool_failure_reason(call.name, rendered_output, evidence.exit_code)
            events: list[HansEvent] = [
                ToolOutput(call_id, _safe_tool_output(rendered_output, failed=not evidence.success)),
                ToolCompleted(
                    call_id,
                    call.name,
                    call.detail,
                    evidence.success,
                    evidence.exit_code,
                    failure_reason,
                ),
            ]
            if call.purpose == "verify":
                self._evidence = evidence
                if evidence.success:
                    events.append(VerificationPassed(call_id, evidence))
                else:
                    events.append(VerificationFailed(call_id, evidence))
            return tuple(events)
        failure_reason = _tool_failure_reason(call.name, rendered_output)
        success = failure_reason is None
        if failure_reason is not None and self._tool_failure is None:
            self._tool_failure = failure_reason
        events = [ToolOutput(call_id, _safe_tool_output(rendered_output, failed=not success))]
        if call.name in {"write_file", "replace_in_file"} and success:
            self._evidence = None
        events.append(ToolCompleted(call_id, call.name, call.detail, success, failure_reason=failure_reason))
        return tuple(events)
