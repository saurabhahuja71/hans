"""Framework-neutral events emitted by the HANS runtime."""

from __future__ import annotations

from dataclasses import dataclass

from bolt_next.errors import FailureCategory
from bolt_next.model_catalog import ModelInfo


@dataclass(frozen=True, slots=True)
class VerificationEvidence:
    command: str
    exit_code: int | None
    stdout_available: bool
    stderr_available: bool
    success: bool
    timestamp: str


@dataclass(frozen=True, slots=True)
class UserMessageSubmitted:
    message: str


@dataclass(frozen=True, slots=True)
class AssistantMessageDelta:
    delta: str


@dataclass(frozen=True, slots=True)
class AssistantMessageComplete:
    text: str


@dataclass(frozen=True, slots=True)
class ToolStarted:
    call_id: str
    name: str
    detail: str
    purpose: str = "inspect"


@dataclass(frozen=True, slots=True)
class ToolOutput:
    call_id: str
    output: str


@dataclass(frozen=True, slots=True)
class ToolCompleted:
    call_id: str
    name: str
    detail: str
    success: bool
    exit_code: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class RequestStarted:
    message: str


@dataclass(frozen=True, slots=True)
class RequestCompleted:
    evidence: VerificationEvidence | None


@dataclass(frozen=True, slots=True)
class RequestFailed:
    category: FailureCategory
    message: str
    debug_message: str | None = None


@dataclass(frozen=True, slots=True)
class RequestCancelled:
    pass


@dataclass(frozen=True, slots=True)
class VerificationStarted:
    call_id: str
    command: str


@dataclass(frozen=True, slots=True)
class VerificationPassed:
    call_id: str
    evidence: VerificationEvidence


@dataclass(frozen=True, slots=True)
class VerificationFailed:
    call_id: str
    evidence: VerificationEvidence


@dataclass(frozen=True, slots=True)
class ConnectionChanged:
    connected: bool


@dataclass(frozen=True, slots=True)
class TaskChangeSummary:
    summary: str


@dataclass(frozen=True, slots=True)
class TaskDiff:
    diff: str


@dataclass(frozen=True, slots=True)
class TaskUndoSucceeded:
    restored_files: tuple[str, ...]
    removed_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TaskUndoRefused:
    conflicting_files: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeControlStatus:
    read_policy: str
    write_policy: str
    execute_policy: str
    external_policy: str


@dataclass(frozen=True, slots=True)
class ModelStatus:
    model: ModelInfo
    current_reasoning_mode: str | None
    models: tuple[ModelInfo, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelChanged:
    previous_model_id: str
    model: ModelInfo
    new_session_started: bool
    reasoning_reset: bool


@dataclass(frozen=True, slots=True)
class ReasoningModeStatus:
    mode: str | None
    available_modes: tuple[str, ...]
    none_semantics: str


@dataclass(frozen=True, slots=True)
class ReasoningModeChanged:
    mode: str | None


@dataclass(frozen=True, slots=True)
class PermissionPolicyChanged:
    category: str
    policy: str


@dataclass(frozen=True, slots=True)
class ToolApprovalDisplay:
    fields: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ToolApprovalRequested:
    request_id: str
    call_id: str
    tool_name: str
    category: str
    display: ToolApprovalDisplay
    external: bool = False
    external_path: str | None = None


@dataclass(frozen=True, slots=True)
class ToolApprovalResolved:
    request_id: str
    call_id: str
    approved: bool


@dataclass(frozen=True, slots=True)
class SessionCleared:
    pass


@dataclass(frozen=True, slots=True)
class RuntimeControlRejected:
    message: str


HansEvent = (
    UserMessageSubmitted
    | AssistantMessageDelta
    | AssistantMessageComplete
    | ToolStarted
    | ToolOutput
    | ToolCompleted
    | RequestStarted
    | RequestCompleted
    | RequestFailed
    | RequestCancelled
    | VerificationStarted
    | VerificationPassed
    | VerificationFailed
    | ConnectionChanged
    | TaskChangeSummary
    | TaskDiff
    | TaskUndoSucceeded
    | TaskUndoRefused
    | RuntimeControlStatus
    | ModelStatus
    | ModelChanged
    | ReasoningModeStatus
    | ReasoningModeChanged
    | PermissionPolicyChanged
    | ToolApprovalRequested
    | ToolApprovalResolved
    | SessionCleared
    | RuntimeControlRejected
)
