from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import Agent, ModelSettings, SQLiteSession
from agents.model_settings import Reasoning
from agents.testing import ModelStep, ScriptedModel, assistant_message, function_call

import bolt_next.runtime as runtime_module
from bolt_next.events import (
    ModelChanged,
    PermissionPolicyChanged,
    ReasoningModeChanged,
    RuntimeControlRejected,
    SessionCleared,
    ToolCompleted,
    ToolOutput,
)
from bolt_next.runtime import HansRuntime
from bolt_next.workspace import make_write_file_tool


def run(coro):
    return asyncio.run(coro)


async def collect(runtime: HansRuntime, message: str):
    return [event async for event in runtime.submit(message)]


def session_items(session: SQLiteSession):
    return run(session.get_items())


def configure_profiles(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_PROFILES", "small,large")
    monkeypatch.setenv("BOLT_MODEL_ACTIVE_PROFILE", "small")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_MODEL", "small-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_BASE_URL", "https://small.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_API_KEY", "small-key")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_CONTEXT_TOKENS", "1024")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_REASONING_MODES", "low,high")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_SMALL_REASONING_EFFORT", "low")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_MODEL", "large-model")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_BASE_URL", "https://large.example.test/v1")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_API_KEY", "large-key")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_CONTEXT_TOKENS", "4096")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_REASONING_MODES", "none")
    monkeypatch.setenv("BOLT_MODEL_PROFILE_LARGE_REASONING_EFFORT", "none")


def agent_for(profile, model: ScriptedModel, workspace: Path, journal) -> Agent:
    effort = profile.reasoning_effort
    return Agent(
        name=f"{profile.info.id} test agent",
        instructions="Use the provided tools.",
        model=model,
        model_settings=ModelSettings(reasoning=Reasoning(effort=effort) if effort else None),
        tools=[make_write_file_tool(workspace, journal)],
    )


def test_switch_starts_clean_sqlite_session_without_history_migration_and_clear_targets_new_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_profiles(monkeypatch)
    models = {
        "small": ScriptedModel([ModelStep(output=[assistant_message("small remembered")])]),
        "large": ScriptedModel(
            [
                ModelStep(output=[assistant_message("large fresh")]),
                ModelStep(output=[assistant_message("large after clear")]),
            ]
        ),
    }

    def factory(workspace, *, journal, profile):
        return agent_for(profile, models[profile.info.id], workspace, journal)

    old_session = SQLiteSession("switch-history-old")
    runtime = HansRuntime(
        workspace=str(tmp_path),
        session=old_session,
        agent_factory=factory,
    )
    try:
        run(collect(runtime, "Remember small history."))
        assert session_items(old_session)

        changed = runtime.select_model("large")
        assert changed == ModelChanged("small", runtime.get_model_status().model, True, True)
        new_session = runtime._session
        assert isinstance(new_session, SQLiteSession)
        assert new_session is not old_session
        assert new_session.session_id != old_session.session_id
        assert session_items(new_session) == []
        assert session_items(old_session)

        run(collect(runtime, "Do you have the old history?"))
        assert "Remember small history" not in repr(models["large"].calls[0].input)
        assert session_items(new_session)
        assert run(runtime.clear_session_history()) == SessionCleared()
        assert session_items(new_session) == []
        assert session_items(old_session)
    finally:
        runtime.close()
        old_session.close()


def test_select_model_same_and_unknown_leave_current_state_untouched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_profiles(monkeypatch)
    session = SQLiteSession("switch-same-unknown")
    runtime = HansRuntime(workspace=str(tmp_path), agent=SimpleNamespace(tools=()), session=session)
    try:
        assert runtime.select_model("small") is None
        assert runtime._session is session
        assert runtime.get_model_status().model.id == "small"

        rejected = runtime.select_model("missing")
        assert rejected == RuntimeControlRejected("Unknown configured model: missing")
        assert runtime._session is session
        assert runtime.get_model_status().model.id == "small"
    finally:
        runtime.close()
        session.close()


def test_selecting_the_active_legacy_model_preserves_its_session(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BOLT_MODEL_PROFILES", raising=False)
    monkeypatch.delenv("BOLT_MODEL_ACTIVE_PROFILE", raising=False)
    monkeypatch.setenv("BOLT_MODEL", "qwen3.6-27b")
    session = SQLiteSession("switch-legacy-same")
    runtime = HansRuntime(workspace=str(tmp_path), agent=SimpleNamespace(tools=()), session=session)
    try:
        assert runtime.select_model("QWEN3.6-27B") is None
        assert runtime._session is session
        assert runtime.get_model_status().model.id == "qwen3.6-27b"
    finally:
        runtime.close()
        session.close()


@pytest.mark.parametrize("failure", ("agent", "session"))
def test_failed_switch_is_atomic_and_old_runtime_can_still_submit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, failure: str
) -> None:
    configure_profiles(monkeypatch)
    old_model = ScriptedModel([ModelStep(output=[assistant_message("old runtime still works")])])
    old_agent = Agent(name="old", instructions="Answer.", model=old_model)
    old_session = SQLiteSession(f"switch-atomic-{failure}")

    def agent_factory(workspace, *, journal, profile):
        if failure == "agent":
            raise RuntimeError("api_key=must-not-leak")
        return agent_for(profile, ScriptedModel([ModelStep(output=[assistant_message("unused")])]), workspace, journal)

    def session_factory(session_id: str):
        if failure == "session":
            raise RuntimeError("token=must-not-leak")
        return SQLiteSession(session_id)

    runtime = HansRuntime(
        workspace=str(tmp_path),
        agent=old_agent,
        session=old_session,
        agent_factory=agent_factory,
        session_factory=session_factory,
    )
    try:
        assert runtime.set_permission("write", "deny") == PermissionPolicyChanged("write", "deny")
        before_filter = runtime._context_filter
        rejected = runtime.select_model("large")
        assert rejected == RuntimeControlRejected("Unable to start the selected model.")
        assert runtime._agent is old_agent
        assert runtime._session is old_session
        assert runtime._context_filter is before_filter
        assert runtime.get_control_status().write_policy == "deny"

        events = run(collect(runtime, "Can the old runtime answer?"))
        assert any(getattr(event, "text", None) == "old runtime still works" for event in events)
        assert session_items(old_session)
    finally:
        runtime.close()
        old_session.close()


def test_switch_binds_profile_context_resets_mode_and_reports_catalog(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_profiles(monkeypatch)
    capacities: list[int] = []

    def fake_filter(context_tokens: int):
        capacities.append(context_tokens)
        return lambda data: data.model_data

    monkeypatch.setattr(runtime_module, "make_fit_model_input", fake_filter)

    class Result:
        async def stream_events(self):
            return
            yield None

    class Runner:
        def __init__(self) -> None:
            self.configs = []

        def run_streamed(self, _agent, _message, *, session, run_config):
            self.configs.append(run_config)
            return Result()

    def factory(workspace, *, journal, profile):
        return agent_for(profile, ScriptedModel([]), workspace, journal)

    runner = Runner()
    session = SQLiteSession("switch-context-mode")
    runtime = HansRuntime(workspace=str(tmp_path), session=session, runner=runner, agent_factory=factory)
    try:
        run(collect(runtime, "Use small."))
        first_filter = runner.configs[-1].call_model_input_filter
        assert capacities == [1024]
        assert runtime.set_reasoning_mode("high") == ReasoningModeChanged("high")

        assert isinstance(runtime.select_model("large"), ModelChanged)
        assert capacities == [1024, 4096]
        large_status = runtime.get_model_status()
        assert [model.id for model in large_status.models] == ["small", "large"]
        assert large_status.current_reasoning_mode == "none"
        assert runtime.set_reasoning_mode("high") == RuntimeControlRejected(
            "Unsupported reasoning mode: high. Supported modes: none."
        )
        run(collect(runtime, "Use large."))
        assert runner.configs[-1].call_model_input_filter is not first_filter

        assert isinstance(runtime.select_model("small"), ModelChanged)
        assert runtime.get_reasoning_mode_status().mode == "low"
        assert runtime.set_reasoning_mode("high") == ReasoningModeChanged("high")
    finally:
        runtime.close()
        session.close()


def test_permission_policy_wraps_target_tools_after_switch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    configure_profiles(monkeypatch)
    large_model = ScriptedModel(
        [
            ModelStep(output=[function_call("write_file", {"path": "blocked.txt", "content": "blocked"}, call_id="write")]),
            ModelStep(output=[assistant_message("write attempted")]),
        ]
    )

    def factory(workspace, *, journal, profile):
        model = large_model if profile.info.id == "large" else ScriptedModel([])
        return agent_for(profile, model, workspace, journal)

    old_session = SQLiteSession("switch-permissions-old")
    runtime = HansRuntime(workspace=str(tmp_path), session=old_session, agent_factory=factory)
    try:
        assert runtime.set_permission("write", "deny") == PermissionPolicyChanged("write", "deny")
        assert isinstance(runtime.select_model("large"), ModelChanged)

        events = run(collect(runtime, "Try writing."))
        assert ToolOutput("write", "Permission denied: write operations are disabled.") in events
        assert ToolCompleted("write", "write_file", "blocked.txt", False, None, "Permission denied.") in events
        assert not (tmp_path / "blocked.txt").exists()
    finally:
        runtime.close()
        old_session.close()


def test_selection_is_rejected_while_busy_and_works_after_cancellation_settles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configure_profiles(monkeypatch)

    class BlockedResult:
        def cancel(self) -> None:
            pass

        async def stream_events(self):
            await asyncio.Event().wait()
            yield None

    class Runner:
        def run_streamed(self, *_args, **_kwargs):
            return BlockedResult()

    def factory(workspace, *, journal, profile):
        return agent_for(profile, ScriptedModel([]), workspace, journal)

    async def scenario():
        old_session = SQLiteSession("switch-busy-old")
        runtime = HansRuntime(workspace=str(tmp_path), session=old_session, runner=Runner(), agent_factory=factory)
        task = asyncio.create_task(collect(runtime, "Wait."))
        await asyncio.sleep(0.01)
        busy = runtime.select_model("large")
        runtime.cancel_active()
        await asyncio.wait_for(task, timeout=1)
        settled = runtime.select_model("large")
        runtime.close()
        old_session.close()
        return busy, settled

    busy, settled = run(scenario())
    assert busy == RuntimeControlRejected("Model selection is available when HANS is idle.")
    assert isinstance(settled, ModelChanged)
