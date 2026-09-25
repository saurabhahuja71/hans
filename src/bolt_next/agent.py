import os
from pathlib import Path

from agents import Agent, OpenAIChatCompletionsModel
from openai import AsyncOpenAI

from bolt_next.workspace import (
    make_read_file_tool,
    make_run_command_tool,
    make_write_file_tool,
    resolve_workspace,
)


def create_agent(workspace: str | Path | None = None) -> Agent:
    """Build the Hans agent using the configured OpenAI-compatible endpoint."""
    base_url = os.environ.get("BOLT_MODEL_BASE_URL")
    api_key = os.environ.get("BOLT_MODEL_API_KEY")
    if not base_url or not api_key:
        raise RuntimeError("Set BOLT_MODEL_BASE_URL and BOLT_MODEL_API_KEY before starting Hans")
    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
    )

    model = OpenAIChatCompletionsModel(
        model=os.environ.get("BOLT_MODEL", "qwen3.6-27b"),
        openai_client=client,
    )

    root = resolve_workspace(workspace or os.environ.get("BOLT_WORKSPACE"))
    return Agent(
        name="Hans",
        instructions=(
            "You are Hans, a coding assistant. "
            "Answer the user's request clearly and concisely. "
            "Use read_file to inspect files, write_file to create or replace files, "
            "and run_command to run a command in the workspace. "
            "Base verification on the command's actual output."
        ),
        model=model,
        tools=[
            make_read_file_tool(root),
            make_write_file_tool(root),
            make_run_command_tool(root),
        ],
    )
