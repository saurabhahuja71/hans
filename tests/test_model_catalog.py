from __future__ import annotations

import os

import pytest

from bolt_next.errors import ConfigurationError
from bolt_next.model_catalog import (
    ModelInfo,
    configured_model_catalog,
    configured_model_info,
    configured_model_profile,
    configured_model_profiles,
)


def test_configured_model_info_is_deterministic_non_secret_and_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOLT_MODEL_PROFILES", raising=False)
    monkeypatch.delenv("BOLT_MODEL_ACTIVE_PROFILE", raising=False)
    monkeypatch.setenv("BOLT_MODEL", " qwen-custom ")
    monkeypatch.setenv("BOLT_MODEL_BASE_URL", "https://private.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "catalog-secret")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "32768")
    monkeypatch.setenv("BOLT_MODEL_REASONING_MODES", " low, NONE ,high ")
    monkeypatch.setenv("BOLT_MODEL_REASONING_NONE_SEMANTICS", " OMIT ")
    environment_before = dict(os.environ)

    first = configured_model_info()
    second = configured_model_info()

    assert first == ModelInfo(
        id="qwen-custom",
        display_name="qwen-custom",
        endpoint_profile="configured OpenAI-compatible endpoint",
        context_tokens=32768,
        supported_reasoning_modes=("low", "none", "high"),
        none_semantics="omit",
    )
    assert second == first
    assert configured_model_catalog() == (first,)
    assert dict(os.environ) == environment_before
    rendered = repr(first)
    assert "private.example.test" not in rendered
    assert "catalog-secret" not in rendered


def test_profile_catalog_is_ordered_normalized_and_secret_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_PROFILES", " Small , large-context ")
    monkeypatch.setenv("BOLT_MODEL_ACTIVE_PROFILE", " LARGE-CONTEXT ")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_MODEL", "small-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_BASE_URL", "https://small.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_API_KEY", "small-profile-secret")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_MODEL", "large-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_BASE_URL", "https://large.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_API_KEY", "large-profile-secret")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_DISPLAY_NAME", "Large context")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_CONTEXT_TOKENS", "32768")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_REASONING_MODES", "low, none, high")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_REASONING_NONE_SEMANTICS", "omit")
    environment_before = dict(os.environ)

    profiles = configured_model_profiles()
    catalog = configured_model_catalog()
    active = configured_model_profile()

    assert [profile.info.id for profile in profiles] == ["small", "large-context"]
    assert [info.id for info in catalog] == ["small", "large-context"]
    assert active == profiles[1]
    assert configured_model_info() == ModelInfo(
        id="large-context",
        display_name="Large context",
        endpoint_profile="configured OpenAI-compatible endpoint",
        context_tokens=32768,
        supported_reasoning_modes=("low", "none", "high"),
        none_semantics="omit",
    )
    assert configured_model_profile("SMALL") == profiles[0]
    assert dict(os.environ) == environment_before
    rendered = repr(profiles)
    assert "small-profile-secret" not in rendered
    assert "large-profile-secret" not in rendered
    assert "small.example.test" not in rendered
    assert "large.example.test" not in rendered


@pytest.mark.parametrize(
    ("profiles", "active", "extra", "message"),
    [
        ("good, GOOD", None, {}, "duplicate"),
        ("invalid_id", None, {}, "matching"),
        ("good", "missing", {"BOLT_MODEL_PROFILE_GOOD_MODEL": "model", "BOLT_MODEL_PROFILE_GOOD_BASE_URL": "url", "BOLT_MODEL_PROFILE_GOOD_API_KEY": "key"}, "unknown"),
        ("good", None, {"BOLT_MODEL_PROFILE_GOOD_MODEL": "model", "BOLT_MODEL_PROFILE_GOOD_BASE_URL": "url"}, "incomplete"),
        (
            "good",
            None,
            {
                "BOLT_MODEL_PROFILE_GOOD_MODEL": "model",
                "BOLT_MODEL_PROFILE_GOOD_BASE_URL": "url",
                "BOLT_MODEL_PROFILE_GOOD_API_KEY": "key",
                "BOLT_MODEL_PROFILE_GOOD_CONTEXT_TOKENS": "not-a-number",
            },
            "integer",
        ),
    ],
)
def test_profile_catalog_rejects_invalid_configuration_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
    profiles: str,
    active: str | None,
    extra: dict[str, str],
    message: str,
) -> None:
    monkeypatch.setenv("BOLT_MODEL_PROFILES", profiles)
    if active is not None:
        monkeypatch.setenv("BOLT_MODEL_ACTIVE_PROFILE", active)
    for name, value in extra.items():
        monkeypatch.setenv(name, value)

    with pytest.raises(ConfigurationError, match=message) as error:
        configured_model_profile()

    assert "not-a-number" not in str(error.value)


def test_catalog_defaults_to_one_model_with_undeclared_reasoning_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable in (
        "BOLT_MODEL_PROFILES",
        "BOLT_MODEL_ACTIVE_PROFILE",
        "BOLT_MODEL",
        "BOLT_MODEL_REASONING_MODES",
        "BOLT_MODEL_REASONING_NONE_SEMANTICS",
        "BOLT_MODEL_CONTEXT_TOKENS",
    ):
        monkeypatch.delenv(variable, raising=False)

    info = configured_model_info()

    assert info.id == "qwen3.6-27b"
    assert info.display_name == "qwen3.6-27b"
    assert info.context_tokens == 16384
    assert info.supported_reasoning_modes == ()
    assert info.none_semantics == "literal"


def test_catalog_rejects_unknown_none_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOLT_MODEL_PROFILES", raising=False)
    monkeypatch.setenv("BOLT_MODEL_REASONING_NONE_SEMANTICS", "disabled")

    with pytest.raises(ConfigurationError, match="literal.*omit"):
        configured_model_info()
