"""Provider-neutral semantic failure contract."""

from __future__ import annotations

from enum import StrEnum


class FailureCategory(StrEnum):
    CONFIGURATION = "configuration"
    AUTHENTICATION = "authentication"
    CONNECTION = "connection"
    CONTEXT = "context"
    MODEL = "model"
    TOOL = "tool"
    CANCELLATION = "cancellation"
    RUNTIME = "runtime"


class ConfigurationError(RuntimeError):
    """Raised when local BOLT_MODEL configuration is invalid."""
