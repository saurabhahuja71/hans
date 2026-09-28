import os
from pathlib import Path

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from agents.model_settings import Reasoning
from openai import AsyncOpenAI

from bolt_next.context_budget import completion_token_reserve
from bolt_next.errors import ConfigurationError
from bolt_next.workspace import (
    make_list_directory_tool,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_run_command_tool,
    make_search_files_tool,
    make_write_file_tool,
    resolve_workspace,
)


STAGE_4_INSTRUCTIONS = (
    "You are Hans, a verification-oriented coding assistant. A claimed fix is not completion. "
    "Follow the available tools and normal agent runtime: INVESTIGATE → ACT → VERIFY → REASON → "
    "ACT AGAIN → VERIFY → DONE. First identify explicit constraints, including read-only or "
    "no-change requests; honor them and do not modify files unless requested. Clarify material "
    "ambiguity before acting rather than guessing. Investigate from repository evidence with targeted "
    "listings, searches, files, metadata, status, and diffs; preserve unrelated changes. Form a "
    "concrete, testable hypothesis and make the smallest necessary source change. Use run_command "
    "with purpose=inspect for repository inspection and purpose=verify only for commands intended "
    "to validate the requested behavior. After changes, inspect the diff and run repository-native "
    "verification when available. Treat a failing command as source evidence only after separating "
    "environment, dependency, configuration, or tool failures from source failures. If verification "
    "fails, diagnose, correct if appropriate, and verify again; do not modify tests merely to pass. "
    "Never claim a check passed without its tool result. Tool results are authoritative; if verification "
    "cannot run, say why. Final responses must be concise and state changes, verification evidence, "
    "and remaining limits. Use list_directory and search_files to discover files, read_file to inspect "
    "them, replace_in_file for one precise edit, write_file only to create or replace an entire file, "
    "and run_command for a direct workspace command. Do not invent patch syntax. When read_file "
    "reports remaining_ranges, request the next start_line instead of assuming the rest of the file."
)


def _model_settings() -> ModelSettings:
    reasoning_effort = os.environ.get("BOLT_MODEL_REASONING_EFFORT", "").strip()
    raw_max_completion = os.environ.get("BOLT_MODEL_MAX_COMPLETION_TOKENS", "").strip()
    extra_args: dict[str, object] = {}
    if raw_max_completion:
        try:
            max_completion_tokens = int(raw_max_completion)
        except ValueError as error:
            raise ConfigurationError("BOLT_MODEL_MAX_COMPLETION_TOKENS must be an integer") from error
        reserve = completion_token_reserve()
        if max_completion_tokens <= 0:
            raise ConfigurationError("BOLT_MODEL_MAX_COMPLETION_TOKENS must be positive")
        # Keep the request valid for the configured context. A provider or
        # profile may advertise a completion value larger than the remaining
        # reserve; cap it instead of failing before the first prompt.
        extra_args["max_completion_tokens"] = min(max_completion_tokens, reserve)
    return ModelSettings(
        reasoning=Reasoning(effort=reasoning_effort) if reasoning_effort else None,
        extra_args=extra_args or None,
    )


def _configured_value(name: str, *, default: str | None = None, required: bool = False) -> str | None:
    value = os.environ.get(name, default)
    value = value.strip() if value is not None else None
    if required and not value:
        raise ConfigurationError("Set BOLT_MODEL_BASE_URL and BOLT_MODEL_API_KEY before starting Hans")
    return value or None


def _configured_timeout_seconds() -> float:
    value = _configured_value("BOLT_MODEL_TIMEOUT_SECONDS", default="90")
    try:
        timeout = float(value) if value is not None else 90.0
    except ValueError as error:
        raise ConfigurationError("BOLT_MODEL_TIMEOUT_SECONDS must be a positive number") from error
    if timeout <= 0:
        raise ConfigurationError("BOLT_MODEL_TIMEOUT_SECONDS must be a positive number")
    return timeout


def _configured_max_retries() -> int:
    value = _configured_value("BOLT_MODEL_MAX_RETRIES", default="0")
    try:
        retries = int(value) if value is not None else 0
    except ValueError as error:
        raise ConfigurationError("BOLT_MODEL_MAX_RETRIES must be a non-negative integer") from error
    if retries < 0:
        raise ConfigurationError("BOLT_MODEL_MAX_RETRIES must be a non-negative integer")
    return retries


def create_agent(workspace: str | Path | None = None) -> Agent:
    """Build the Hans agent using the configured OpenAI-compatible endpoint."""
    base_url = _configured_value("BOLT_MODEL_BASE_URL", required=True)
    api_key = _configured_value("BOLT_MODEL_API_KEY", required=True)
    project = _configured_value("BOLT_MODEL_OPENAI_PROJECT")
    model_name = _configured_value("BOLT_MODEL", default="qwen3.6-27b")
    if model_name is None:
        raise ConfigurationError("BOLT_MODEL must not be empty")
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        project=project,
        timeout=_configured_timeout_seconds(),
        max_retries=_configured_max_retries(),
    )

    model = OpenAIChatCompletionsModel(
        model=model_name,
        openai_client=client,
    )

    root = resolve_workspace(workspace or os.environ.get("BOLT_WORKSPACE"))
    return Agent(
        name="Hans",
        instructions=STAGE_4_INSTRUCTIONS,
        model=model,
        model_settings=_model_settings(),
        tools=[
            make_list_directory_tool(root),
            make_search_files_tool(root),
            make_read_file_tool(root),
            make_replace_in_file_tool(root),
            make_write_file_tool(root),
            make_run_command_tool(root),
        ],
    )
