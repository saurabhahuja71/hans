import os
from pathlib import Path

from agents import Agent, ModelSettings, OpenAIChatCompletionsModel
from agents.model_settings import Reasoning
from openai import AsyncOpenAI

from bolt_next.workspace import (
    make_read_file_tool,
    make_run_command_tool,
    make_write_file_tool,
    resolve_workspace,
)


STAGE_4_INSTRUCTIONS = (
    "You are Hans, a verification-oriented coding assistant. A claimed fix is not completion. "
    "For coding, debugging, and repository-change tasks, follow this cycle using the available "
    "tools and the normal agent runtime: INVESTIGATE → ACT → VERIFY → REASON → ACT AGAIN → "
    "VERIFY → DONE. First understand the task and inspect relevant repository state. Form a "
    "concrete, testable hypothesis before changing code. Make the smallest necessary source "
    "change. After changing code, run an appropriate verification command and inspect its actual "
    "tool result. If verification fails, diagnose that new failure, correct the implementation, "
    "and verify again; do not stop after the first failed attempt. Do not modify tests merely to "
    "make an implementation pass. Before final completion, inspect the resulting state or diff "
    "when appropriate and confirm that the requested behavior is satisfied. Never claim that a "
    "test or command passed unless a tool result shows it passed. Tool results are the authoritative "
    "evidence; do not replace them with guesses or model-generated summaries. If verification is "
    "not possible, say explicitly that it could not be performed. Use read_file to inspect files, "
    "write_file to create or replace files, and run_command to run a direct command in the "
    "workspace. When read_file reports remaining_ranges, request the next start_line instead of "
    "assuming the rest of the file."
)


def _model_settings() -> ModelSettings:
    reasoning_effort = os.environ.get("BOLT_MODEL_REASONING_EFFORT", "").strip()
    if not reasoning_effort:
        return ModelSettings()
    return ModelSettings(reasoning=Reasoning(effort=reasoning_effort))


def create_agent(workspace: str | Path | None = None) -> Agent:
    """Build the Hans agent using the configured OpenAI-compatible endpoint."""
    base_url = os.environ.get("BOLT_MODEL_BASE_URL")
    api_key = os.environ.get("BOLT_MODEL_API_KEY")
    project = os.environ.get("BOLT_MODEL_OPENAI_PROJECT")
    if not base_url or not api_key:
        raise RuntimeError("Set BOLT_MODEL_BASE_URL and BOLT_MODEL_API_KEY before starting Hans")
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        project=project or None,
    )

    model = OpenAIChatCompletionsModel(
        model=os.environ.get("BOLT_MODEL", "qwen3.6-27b"),
        openai_client=client,
    )

    root = resolve_workspace(workspace or os.environ.get("BOLT_WORKSPACE"))
    return Agent(
        name="Hans",
        instructions=STAGE_4_INSTRUCTIONS,
        model=model,
        model_settings=_model_settings(),
        tools=[
            make_read_file_tool(root),
            make_write_file_tool(root),
            make_run_command_tool(root),
        ],
    )
