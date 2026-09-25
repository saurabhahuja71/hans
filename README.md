# HANS

HANS is a small Python terminal coding-agent prototype built on the OpenAI Agents SDK.

The architecture is:

```text
HANS terminal UI
        |
OpenAI Agents SDK
        |
OpenAI-compatible Chat Completions endpoint
        |
Qwen3.6 / llama-server / another compatible local provider
```

HANS does not implement a competing agent runtime. The SDK owns the agent loop, structured tool
calling, tool results, streaming lifecycle, and conversation history. HANS-specific code handles
terminal interaction, model configuration, workspace policy, and presentation of SDK events.

## Current milestone

The prototype provides:

- Interactive terminal prompt with `exit`, `quit`, EOF, and Ctrl-C handling
- Streaming assistant text through `Runner.run_streamed()` and `result.stream_events()`
- Multi-turn history through the SDK's `SQLiteSession`
- OpenAI-compatible Chat Completions through `OpenAIChatCompletionsModel`
- One safe structured function tool: `read_file(path: str)`
- Workspace-bound path resolution with traversal and outside-workspace rejection
- Clear model and tool errors without terminating the TUI

The Python import package remains `bolt_next` for now to avoid a partial package rename; the
distribution, command, and runtime branding are HANS.

## Configuration

```bash
export BOLT_MODEL_BASE_URL="http://localhost:8080/v1"
export BOLT_MODEL_API_KEY="local-key"
export BOLT_MODEL="qwen3.6-27b"
export BOLT_WORKSPACE="$PWD"              # optional; defaults to the current directory
```

Credentials are read from the environment and are not embedded in source code.

## Run

Install the project in the Python 3.12 environment containing the Agents SDK, then run:

```bash
pip install -e .
hans
```

Example prompt:

```text
> inspect src/bolt_next/main.py
```

When the model chooses to inspect the file, it must emit the SDK's structured `read_file` call.
HANS executes that Python tool and the SDK sends its result back to the model. Ordinary model text
is never parsed as a tool call.

## Workspace safety

`read_file` is read-only and restricted to the resolved workspace. It rejects paths resolving
outside the workspace, including traversal paths and symlinks targeting external files. Missing,
unreadable, and non-UTF-8 files are returned as useful tool errors.

No shell, write, edit, search, SSH, or command-execution tools are included in this milestone.

## Development

```bash
python -m compileall src
pytest -q
```

The real Qwen smoke test should verify the complete path from user prompt to structured
`read_file` call, actual Python execution, tool result, second model turn, and final answer. Do not
claim that path is verified unless the endpoint is reachable and the test actually completes.

## Next milestones

Run the integration tests in an environment with pytest installed and complete a real Qwen smoke
test. Only after that should HANS add more SDK-backed tools such as directory listing, search, file
edits, or command execution with appropriate approval handling.
