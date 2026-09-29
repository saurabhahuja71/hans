import inspect
import json
import os
from collections.abc import Callable
from pathlib import Path

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from agents.model_settings import Reasoning
from openai import AsyncOpenAI

from bolt_next.context_budget import completion_token_reserve
from bolt_next.errors import ConfigurationError
from bolt_next.model_catalog import ConfiguredModelProfile, configured_model_profile
from bolt_next.workspace import (
    ExternalPathAuthorizer,
    TaskMutationJournal,
    make_list_directory_tool,
    make_read_file_tool,
    make_replace_in_file_tool,
    make_run_command_tool,
    make_search_files_tool,
    make_write_file_tool,
    resolve_workspace,
)


_TOOL_PERMISSION_GROUPS = {
    "list_directory": "read",
    "search_files": "read",
    "read_file": "read",
    "write_file": "write",
    "replace_in_file": "write",
    "run_command": "execute",
}


def _external_path_argument(arguments: object) -> str | None:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) else None


def apply_tool_permission_policy(
    agent: Agent,
    get_policy: Callable[[str], str],
    *,
    authorizer: ExternalPathAuthorizer | None = None,
) -> None:
    """Install runtime-owned dynamic policy hooks on HANS workspace tools."""
    for tool in getattr(agent, "tools", ()):
        tool_name = getattr(tool, "name", "")
        category = _TOOL_PERMISSION_GROUPS.get(tool_name)
        if category is None:
            continue
        original_invoke = getattr(tool, "_hans_original_invoke", None)
        if original_invoke is None:
            original_invoke = getattr(tool, "on_invoke_tool", None)
            if not callable(original_invoke):
                continue
            setattr(tool, "_hans_original_invoke", original_invoke)
            setattr(tool, "_hans_original_needs_approval", getattr(tool, "needs_approval", False))

            async def invoke(context, arguments, *, tool=tool, category=category):
                if getattr(tool, "_hans_get_policy")(category) == "deny":
                    return f"Permission denied: {category} operations are disabled."
                result = getattr(tool, "_hans_original_invoke")(context, arguments)
                if inspect.isawaitable(result):
                    return await result
                return result

            async def needs_approval(context, arguments, call_id, *, tool=tool, category=category):
                policy = getattr(tool, "_hans_get_policy")(category)
                if policy == "deny":
                    return False
                tool_name = getattr(tool, "name", "")
                path = _external_path_argument(arguments)
                active_authorizer = getattr(tool, "_hans_authorizer", None)
                if (
                    active_authorizer is not None
                    and tool_name in {"read_file", "list_directory", "search_files", "write_file", "replace_in_file"}
                    and isinstance(call_id, str)
                    and path is not None
                ):
                    try:
                        proposal = active_authorizer.propose(
                            tool_name,
                            call_id,
                            path,
                            mutation=tool_name in {"write_file", "replace_in_file"},
                        )
                    except (OSError, ValueError):
                        proposal = None
                    if proposal is not None:
                        return True
                if policy == "ask":
                    return True
                original = getattr(tool, "_hans_original_needs_approval")
                if isinstance(original, bool):
                    return original
                result = original(context, arguments, call_id)
                return bool(await result) if inspect.isawaitable(result) else bool(result)

            tool.on_invoke_tool = invoke
            tool.needs_approval = needs_approval
        setattr(tool, "_hans_get_policy", get_policy)
        setattr(tool, "_hans_authorizer", authorizer)


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
    "and run_command for a direct workspace command. Any filesystem path that resolves outside the workspace "
    "requires explicit approval for that exact tool call. Do not invent patch syntax. When read_file "
    "reports remaining_ranges, request the next start_line instead of assuming the rest of the file."
)


def _model_settings(profile: ConfiguredModelProfile | None = None) -> ModelSettings:
    reasoning_effort = profile.reasoning_effort if profile is not None else os.environ.get("BOLT_MODEL_REASONING_EFFORT", "").strip()
    raw_max_completion = (
        profile.max_completion_tokens if profile is not None else os.environ.get("BOLT_MODEL_MAX_COMPLETION_TOKENS", "").strip()
    )
    extra_args: dict[str, object] = {}
    if raw_max_completion:
        try:
            max_completion_tokens = int(raw_max_completion)
        except (TypeError, ValueError) as error:
            raise ConfigurationError("BOLT_MODEL_MAX_COMPLETION_TOKENS must be an integer") from error
        reserve = completion_token_reserve(profile.info.context_tokens if profile is not None else None)
        if max_completion_tokens <= 0:
            raise ConfigurationError("BOLT_MODEL_MAX_COMPLETION_TOKENS must be positive")
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


def _selected_profile(profile: ConfiguredModelProfile | str | None) -> ConfiguredModelProfile:
    if isinstance(profile, ConfiguredModelProfile):
        return profile
    return configured_model_profile(profile)


def create_agent(
    workspace: str | Path | None = None,
    *,
    journal: TaskMutationJournal | None = None,
    profile: ConfiguredModelProfile | str | None = None,
    authorizer: ExternalPathAuthorizer | None = None,
) -> Agent:
    """Build the Hans agent using the selected OpenAI-compatible model profile."""
    selected_profile = _selected_profile(profile)
    if not selected_profile.base_url or not selected_profile.api_key:
        raise ConfigurationError("Set BOLT_MODEL_BASE_URL and BOLT_MODEL_API_KEY before starting Hans")
    client = AsyncOpenAI(
        base_url=selected_profile.base_url,
        api_key=selected_profile.api_key,
        project=selected_profile.openai_project,
        timeout=selected_profile.timeout_seconds,
        max_retries=selected_profile.max_retries,
    )
    model = OpenAIChatCompletionsModel(
        model=selected_profile.model,
        openai_client=client,
    )

    root = resolve_workspace(workspace or os.environ.get("BOLT_WORKSPACE"))
    context_tokens = selected_profile.info.context_tokens
    return Agent(
        name="Hans",
        instructions=STAGE_4_INSTRUCTIONS,
        model=model,
        model_settings=_model_settings(selected_profile),
        tools=[
            make_list_directory_tool(root, context_tokens=context_tokens, authorizer=authorizer),
            make_search_files_tool(root, context_tokens=context_tokens, authorizer=authorizer),
            make_read_file_tool(root, context_tokens=context_tokens, authorizer=authorizer),
            make_replace_in_file_tool(root, journal, authorizer=authorizer),
            make_write_file_tool(root, journal, authorizer=authorizer),
            make_run_command_tool(root, context_tokens=context_tokens),
        ],
    )
