"""Local, secret-safe metadata and transport configuration for configured models."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Literal

from bolt_next.context_budget import DEFAULT_CONTEXT_TOKENS, context_token_limit
from bolt_next.errors import ConfigurationError

_DEFAULT_MODEL_ID = "qwen3.6-27b"
_ENDPOINT_PROFILE = "configured OpenAI-compatible endpoint"
_PROFILE_ID_PATTERN = re.compile(r"[a-z][a-z0-9-]*\Z")


@dataclass(frozen=True, slots=True)
class ModelInfo:
    id: str
    display_name: str
    endpoint_profile: str
    context_tokens: int
    supported_reasoning_modes: tuple[str, ...]
    none_semantics: Literal["literal", "omit"]
    supports_image_input: bool = False


@dataclass(frozen=True, slots=True)
class ConfiguredModelProfile:
    """Internal transport configuration paired with its UI-safe model metadata."""

    info: ModelInfo
    model: str = field(repr=False)
    base_url: str | None = field(repr=False)
    api_key: str | None = field(repr=False)
    openai_project: str | None = field(repr=False)
    timeout_seconds: float = field(repr=False)
    max_retries: int = field(repr=False)
    max_completion_tokens: int | None = field(repr=False)
    reasoning_effort: str | None = field(repr=False)
    transport: Literal["chat_completions", "responses"] = field(repr=False)
    supports_image_input: bool = field(repr=False)


def _configured_model_id() -> str:
    return os.environ.get("BOLT_MODEL", _DEFAULT_MODEL_ID).strip() or _DEFAULT_MODEL_ID


def _configured_reasoning_modes(raw_modes: str | None = None) -> tuple[str, ...]:
    modes: list[str] = []
    for raw_mode in (raw_modes if raw_modes is not None else os.environ.get("BOLT_MODEL_REASONING_MODES", "")).split(","):
        mode = raw_mode.strip().lower()
        if mode and mode not in modes:
            modes.append(mode)
    return tuple(modes)


def _configured_none_semantics(raw_semantics: str | None = None, *, variable: str = "BOLT_MODEL_REASONING_NONE_SEMANTICS") -> Literal["literal", "omit"]:
    semantics = (raw_semantics if raw_semantics is not None else os.environ.get(variable, "literal")).strip().lower() or "literal"
    if semantics not in {"literal", "omit"}:
        raise ConfigurationError(f"{variable} must be 'literal' or 'omit'")
    return semantics  # type: ignore[return-value]


def _normalized_profile_id(value: str, *, variable: str) -> str:
    profile_id = value.strip().lower()
    if not _PROFILE_ID_PATTERN.fullmatch(profile_id):
        raise ConfigurationError(f"{variable} must be a comma-separated list of profile IDs matching [a-z][a-z0-9-]*")
    return profile_id


def _profile_prefix(profile_id: str) -> str:
    return f"BOLT_MODEL_PROFILE_{profile_id.upper().replace('-', '_')}_"


def _optional_value(variable: str) -> str | None:
    value = os.environ.get(variable)
    return value.strip() if value and value.strip() else None


def _required_profile_value(profile_id: str, prefix: str, suffix: str) -> str:
    value = _optional_value(prefix + suffix)
    if value is None:
        raise ConfigurationError(
            f"Configured model profile {profile_id!r} is incomplete; set {prefix}MODEL, {prefix}BASE_URL, and {prefix}API_KEY"
        )
    return value


def _positive_number(variable: str, value: str | None, *, default: float) -> float:
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError as error:
        raise ConfigurationError(f"{variable} must be a positive number") from error
    if parsed <= 0:
        raise ConfigurationError(f"{variable} must be a positive number")
    return parsed


def _non_negative_integer(variable: str, value: str | None, *, default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{variable} must be a non-negative integer") from error
    if parsed < 0:
        raise ConfigurationError(f"{variable} must be a non-negative integer")
    return parsed


def _positive_integer(variable: str, value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{variable} must be a positive integer") from error
    if parsed <= 0:
        raise ConfigurationError(f"{variable} must be a positive integer")
    return parsed


def _profile_transport(variable: str, value: str | None) -> Literal["chat_completions", "responses"]:
    transport = (value or "chat_completions").lower()
    if transport not in {"chat_completions", "responses"}:
        raise ConfigurationError(f"{variable} must be 'chat_completions' or 'responses'")
    return transport  # type: ignore[return-value]


def _profile_image_support(variable: str, value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.lower()
    if normalized not in {"true", "false"}:
        raise ConfigurationError(f"{variable} must be 'true' or 'false'")
    return normalized == "true"


def _profile_context_tokens(variable: str, value: str | None) -> int:
    if value is None:
        return DEFAULT_CONTEXT_TOKENS
    try:
        tokens = int(value)
    except ValueError as error:
        raise ConfigurationError(f"{variable} must be an integer") from error
    return context_token_limit(tokens)


def _legacy_profile() -> ConfiguredModelProfile:
    model_id = _configured_model_id()
    max_completion = _positive_integer("BOLT_MODEL_MAX_COMPLETION_TOKENS", _optional_value("BOLT_MODEL_MAX_COMPLETION_TOKENS"))
    supports_image_input = _profile_image_support(
        "BOLT_MODEL_SUPPORTS_IMAGE_INPUT", _optional_value("BOLT_MODEL_SUPPORTS_IMAGE_INPUT"), default=True
    )
    return ConfiguredModelProfile(
        info=ModelInfo(
            id=model_id,
            display_name=model_id,
            endpoint_profile=_ENDPOINT_PROFILE,
            context_tokens=context_token_limit(),
            supported_reasoning_modes=_configured_reasoning_modes(),
            none_semantics=_configured_none_semantics(),
            supports_image_input=supports_image_input,
        ),
        model=model_id,
        base_url=_optional_value("BOLT_MODEL_BASE_URL"),
        api_key=_optional_value("BOLT_MODEL_API_KEY"),
        openai_project=_optional_value("BOLT_MODEL_OPENAI_PROJECT"),
        timeout_seconds=_positive_number(
            "BOLT_MODEL_TIMEOUT_SECONDS", _optional_value("BOLT_MODEL_TIMEOUT_SECONDS"), default=90.0
        ),
        max_retries=_non_negative_integer(
            "BOLT_MODEL_MAX_RETRIES", _optional_value("BOLT_MODEL_MAX_RETRIES"), default=0
        ),
        max_completion_tokens=max_completion,
        reasoning_effort=_optional_value("BOLT_MODEL_REASONING_EFFORT"),
        transport="chat_completions",
        supports_image_input=supports_image_input,
    )


def _profile_from_environment(profile_id: str) -> ConfiguredModelProfile:
    prefix = _profile_prefix(profile_id)
    model = _required_profile_value(profile_id, prefix, "MODEL")
    base_url = _required_profile_value(profile_id, prefix, "BASE_URL")
    api_key = _required_profile_value(profile_id, prefix, "API_KEY")
    transport = _profile_transport(prefix + "TRANSPORT", _optional_value(prefix + "TRANSPORT"))
    supports_image_input = _profile_image_support(
        prefix + "SUPPORTS_IMAGE_INPUT", _optional_value(prefix + "SUPPORTS_IMAGE_INPUT")
    )
    return ConfiguredModelProfile(
        info=ModelInfo(
            id=profile_id,
            display_name=_optional_value(prefix + "DISPLAY_NAME") or profile_id,
            endpoint_profile=_optional_value(prefix + "ENDPOINT_PROFILE") or _ENDPOINT_PROFILE,
            context_tokens=_profile_context_tokens(prefix + "CONTEXT_TOKENS", _optional_value(prefix + "CONTEXT_TOKENS")),
            supported_reasoning_modes=_configured_reasoning_modes(_optional_value(prefix + "REASONING_MODES") or ""),
            none_semantics=_configured_none_semantics(
                _optional_value(prefix + "REASONING_NONE_SEMANTICS"), variable=prefix + "REASONING_NONE_SEMANTICS"
            ),
            supports_image_input=supports_image_input,
        ),
        model=model,
        base_url=base_url,
        api_key=api_key,
        openai_project=_optional_value(prefix + "OPENAI_PROJECT"),
        timeout_seconds=_positive_number(
            prefix + "TIMEOUT_SECONDS", _optional_value(prefix + "TIMEOUT_SECONDS"), default=90.0
        ),
        max_retries=_non_negative_integer(prefix + "MAX_RETRIES", _optional_value(prefix + "MAX_RETRIES"), default=0),
        max_completion_tokens=_positive_integer(
            prefix + "MAX_COMPLETION_TOKENS", _optional_value(prefix + "MAX_COMPLETION_TOKENS")
        ),
        reasoning_effort=_optional_value(prefix + "REASONING_EFFORT"),
        transport=transport,
        supports_image_input=supports_image_input,
    )


def configured_model_profiles() -> tuple[ConfiguredModelProfile, ...]:
    """Return all explicitly configured local profiles, without remote discovery."""
    raw_profiles = os.environ.get("BOLT_MODEL_PROFILES")
    if raw_profiles is None:
        return (_legacy_profile(),)
    profile_ids: list[str] = []
    for raw_profile_id in raw_profiles.split(","):
        profile_id = _normalized_profile_id(raw_profile_id, variable="BOLT_MODEL_PROFILES")
        if profile_id in profile_ids:
            raise ConfigurationError("BOLT_MODEL_PROFILES must not contain duplicate profile IDs")
        profile_ids.append(profile_id)
    if not profile_ids:
        raise ConfigurationError("BOLT_MODEL_PROFILES must contain at least one profile ID")
    return tuple(_profile_from_environment(profile_id) for profile_id in profile_ids)


def configured_model_profile(profile_id: str | None = None) -> ConfiguredModelProfile:
    """Return an explicit profile, or the configured active profile by default."""
    profiles = configured_model_profiles()
    selected_id = profile_id
    if selected_id is None:
        selected_id = os.environ.get("BOLT_MODEL_ACTIVE_PROFILE", "")
    if not selected_id or not selected_id.strip():
        return profiles[0]
    normalized_id = _normalized_profile_id(selected_id, variable="BOLT_MODEL_ACTIVE_PROFILE")
    for profile in profiles:
        if profile.info.id == normalized_id:
            return profile
    raise ConfigurationError(f"BOLT_MODEL_ACTIVE_PROFILE selects unknown profile {normalized_id!r}")


def configured_model_info() -> ModelInfo:
    """Return active local model metadata without endpoint or credential data."""
    return configured_model_profile().info


def configured_model_catalog() -> tuple[ModelInfo, ...]:
    """Return the locally configured catalog; remote model discovery is unsupported."""
    return tuple(profile.info for profile in configured_model_profiles())
