"""Framework-neutral events emitted by the HANS runtime."""

from __future__ import annotations

from dataclasses import dataclass


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


@dataclass(frozen=True, slots=True)
class RequestStarted:
    message: str


@dataclass(frozen=True, slots=True)
class RequestCompleted:
    evidence: VerificationEvidence | None


@dataclass(frozen=True, slots=True)
class RequestFailed:
    category: str
    message: str


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
)
