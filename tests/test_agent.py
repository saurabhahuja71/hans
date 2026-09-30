import os
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


def test_create_agent_binds_selected_profile_without_mutating_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("BOLT_MODEL_PROFILES", "small,selected")
    monkeypatch.setenv("BOLT_MODEL_ACTIVE_PROFILE", "small")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_MODEL", "small-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_BASE_URL", "https://small.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_API_KEY", "small-key")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_MODEL", "selected-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_BASE_URL", "https://selected.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_API_KEY", "selected-key")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_OPENAI_PROJECT", "selected-project")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_MAX_RETRIES", "3")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_CONTEXT_TOKENS", "8192")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_REASONING_EFFORT", "high")
    environment_before = dict(os.environ)
    captured: dict[str, object] = {"tool_capacities": []}

    class FakeAsyncOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["client"] = kwargs
            captured["openai_client"] = self

    def record_capacity(name: str):
        def tool(_root: Path, *, context_tokens: int | None = None, authorizer: object = None, **_kwargs: object) -> str:
            captured["tool_capacities"].append((name, context_tokens))
            return name

        return tool

    monkeypatch.setattr(agent_module, "AsyncOpenAI", FakeAsyncOpenAI)
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent_module, "Agent", lambda **kwargs: kwargs)
    monkeypatch.setattr(agent_module, "make_list_directory_tool", record_capacity("list"))
    monkeypatch.setattr(agent_module, "make_search_files_tool", record_capacity("search"))
    monkeypatch.setattr(agent_module, "make_read_file_tool", record_capacity("read"))
    monkeypatch.setattr(agent_module, "make_run_command_tool", record_capacity("command"))
    monkeypatch.setattr(agent_module, "make_replace_in_file_tool", lambda *_args, **_kwargs: "replace")
    monkeypatch.setattr(agent_module, "make_write_file_tool", lambda *_args, **_kwargs: "write")

    agent = agent_module.create_agent(tmp_path, profile="selected")

    assert captured["client"] == {
        "base_url": "https://selected.example.test/v1",
        "api_key": "selected-key",
        "project": "selected-project",
        "timeout": 12.5,
        "max_retries": 3,
    }
    assert agent["model"] == {"model": "selected-model", "openai_client": captured["openai_client"]}
    assert agent["model_settings"].reasoning.effort == "high"
    assert agent["model_settings"].extra_args == {"max_completion_tokens": 2048}
    assert captured["tool_capacities"] == [
        ("list", 8192),
        ("search", 8192),
        ("read", 8192),
        ("command", 8192),
    ]
    assert dict(os.environ) == environment_before


@pytest.mark.parametrize(
    ("transport", "image_support", "expected_model", "has_image_tool"),
    [
        ("chat_completions", "false", "chat", False),
        ("responses", "true", "responses", True),
    ],
)
def test_create_agent_selects_configured_transport_and_image_tool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    transport: str,
    image_support: str,
    expected_model: str,
    has_image_tool: bool,
) -> None:
    monkeypatch.setenv("BOLT_MODEL_PROFILES", "selected")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_MODEL", "selected-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_BASE_URL", "https://selected.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_API_KEY", "selected-key")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_TRANSPORT", transport)
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SELECTED_SUPPORTS_IMAGE_INPUT", image_support)
    read_file_capabilities: list[bool] = []
    original_read_file_tool = agent_module.make_read_file_tool

    def record_read_file_capability(*args: object, read_image_available: bool = False, **kwargs: object):
        read_file_capabilities.append(read_image_available)
        return original_read_file_tool(*args, read_image_available=read_image_available, **kwargs)

    monkeypatch.setattr(agent_module, "AsyncOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", lambda **_kwargs: "chat")
    monkeypatch.setattr(agent_module, "OpenAIResponsesModel", lambda **_kwargs: "responses")
    monkeypatch.setattr(agent_module, "make_read_file_tool", record_read_file_capability)
    monkeypatch.setattr(agent_module, "Agent", lambda **kwargs: kwargs)

    agent = agent_module.create_agent(tmp_path)

    assert agent["model"] == expected_model
    assert read_file_capabilities == [has_image_tool]
    assert ("read_image" in [tool.name for tool in agent["tools"]]) is has_image_tool


@pytest.mark.parametrize(
    ("image_support", "has_image_tool"),
    [("true", True), ("false", False)],
)
def test_create_agent_legacy_image_support_configures_image_tools(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, image_support: str, has_image_tool: bool
) -> None:
    monkeypatch.delenv("BOLT_MODEL_PROFILES", raising=False)
    monkeypatch.setenv("BOLT_MODEL_BASE_URL", "https://models.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "test-key")
    monkeypatch.setenv("BOLT_MODEL_SUPPORTS_IMAGE_INPUT", image_support)
    monkeypatch.setattr(agent_module, "AsyncOpenAI", lambda **_kwargs: object())
    monkeypatch.setattr(agent_module, "OpenAIChatCompletionsModel", lambda **_kwargs: "chat")
    monkeypatch.setattr(agent_module, "Agent", lambda **kwargs: kwargs)

    agent = agent_module.create_agent(tmp_path)
    tools = {tool.name: tool for tool in agent["tools"]}

    assert "read_file" in tools
    assert ("read_image" in tools) is has_image_tool
    if has_image_tool:
        assert "UTF-8 text" in tools["read_file"].description
        assert "model-visible image data" not in tools["read_file"].description
        assert "model-visible image data" in tools["read_image"].description


def test_stage_4_instructions_require_purpose_and_evidenced_constraints() -> None:
    instructions = agent_module.STAGE_4_INSTRUCTIONS

    assert "read-only" in instructions
    assert "Clarify material ambiguity" in instructions
    assert "purpose=inspect" in instructions
    assert "purpose=verify" in instructions
    assert "environment, dependency, configuration, or tool failures" in instructions
    assert "verification evidence" in instructions
    assert "Never use run_command to list, find, or inspect" in instructions
    assert "including for a literal ~/ path" in instructions
