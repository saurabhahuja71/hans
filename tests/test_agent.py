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
