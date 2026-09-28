from pathlib import Path

import pytest

import bolt_next.agent as agent_module


@pytest.mark.parametrize(
    ("configured_project", "expected_project"),
    [(None, None), ("project-123", "project-123")],
)
def test_create_agent_forwards_optional_openai_project(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    configured_project: str | None,
    expected_project: str | None,
) -> None:
    monkeypatch.setenv("BOLT_MODEL_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("BOLT_MODEL", "test-model")
    if configured_project is None:
        monkeypatch.delenv("BOLT_MODEL_OPENAI_PROJECT", raising=False)
    else:
        monkeypatch.setenv("BOLT_MODEL_OPENAI_PROJECT", configured_project)

    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client_options"] = kwargs
            captured["client"] = self

    def fake_model(**kwargs: object) -> dict[str, object]:
        captured["model"] = kwargs
        return kwargs

    def fake_agent(**kwargs: object) -> dict[str, object]:
        captured["agent"] = kwargs
        return kwargs

    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", fake_model)
    monkeypatch.setattr(agent_module, "Agent", fake_agent)

    agent_module.create_agent(tmp_path)

    assert captured["client_options"] == {
        "base_url": "https://models.example.test/v1",
        "api_key": "test-key",
        "project": expected_project,
        "timeout": 90.0,
        "max_retries": 0,
    }
    assert captured["model"] == {"model": "test-model", "openai_client": captured["client"]}
    assert captured["agent"]["model_settings"].reasoning is None


def test_reasoning_effort_is_opt_in_provider_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("BOLT_MODEL_REASONING_EFFORT", raising=False)
    assert agent_module._model_settings().reasoning is None

    monkeypatch.setenv("BOLT_MODEL_REASONING_EFFORT", "none")
    reasoning = agent_module._model_settings().reasoning
    assert reasoning is not None
    assert reasoning.effort == "none"


def test_create_agent_trims_configured_model_values(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BOLT_MODEL_BASE_URL", " https://models.example.test/v1 ")
    monkeypatch.setenv("BOLT_MODEL_API_KEY", " test-key ")
    monkeypatch.setenv("BOLT_MODEL", " test-model ")
    monkeypatch.setenv("BOLT_MODEL_OPENAI_PROJECT", " project-123 ")
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs

    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent_module, "Agent", lambda **kwargs: kwargs)

    agent_module.create_agent(tmp_path)

    assert captured["client"] == {
        "base_url": "https://models.example.test/v1",
        "api_key": "test-key",
        "project": "project-123",
        "timeout": 90.0,
        "max_retries": 0,
    }


def test_create_agent_forwards_generic_transport_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BOLT_MODEL_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("BOLT_MODEL_TIMEOUT_SECONDS", " 12.5 ")
    monkeypatch.setenv("BOLT_MODEL_MAX_RETRIES", " 3 ")
    captured: dict[str, object] = {}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent_module, "Agent", lambda **kwargs: kwargs)

    agent_module.create_agent(tmp_path)

    assert captured["timeout"] == 12.5
    assert captured["max_retries"] == 3


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("BOLT_MODEL_TIMEOUT_SECONDS", "0"),
        ("BOLT_MODEL_TIMEOUT_SECONDS", "invalid"),
        ("BOLT_MODEL_MAX_RETRIES", "-1"),
        ("BOLT_MODEL_MAX_RETRIES", "invalid"),
    ),
)
def test_invalid_transport_configuration_is_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(agent_module.ConfigurationError):
        if name == "BOLT_MODEL_TIMEOUT_SECONDS":
            agent_module._configured_timeout_seconds()
        else:
            agent_module._configured_max_retries()


@pytest.mark.parametrize("value", ("0", "-1", "invalid"))
def test_invalid_max_completion_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "16384")
    monkeypatch.setenv("BOLT_MODEL_MAX_COMPLETION_TOKENS", value)

    with pytest.raises(agent_module.ConfigurationError):
        agent_module._model_settings()


def test_max_completion_is_capped_to_context_reserve(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "16384")
    monkeypatch.setenv("BOLT_MODEL_MAX_COMPLETION_TOKENS", "8192")

    settings = agent_module._model_settings()

    assert settings.extra_args == {"max_completion_tokens": 4096}


def test_stage_4_instructions_require_purpose_and_evidenced_constraints() -> None:
    instructions = agent_module.STAGE_4_INSTRUCTIONS

    assert "read-only" in instructions
    assert "Clarify material ambiguity" in instructions
    assert "purpose=inspect" in instructions
    assert "purpose=verify" in instructions
    assert "environment, dependency, configuration, or tool failures" in instructions
    assert "verification evidence" in instructions
