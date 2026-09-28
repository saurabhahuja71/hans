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
Qwen3.6 / llama-server / another compatible provider
```

HANS does not implement a competing agent runtime. The SDK owns the agent loop, structured tool
calling, tool results, streaming, and conversation history. HANS owns the terminal UI, model
configuration, workspace policy, the tool implementations, and how SDK events are displayed.

The Python import package remains `bolt_next`. The distribution name, console command, and runtime
branding are HANS. Current release: **0.2.1**.

## What 0.2.1 provides

- Interactive prompt with `exit`, `quit`, EOF, and Ctrl-C handling
- Streaming assistant text through `Runner.run_streamed()` and `result.stream_events()`
- Multi-turn history for the life of the process through the SDK's in-memory `SQLiteSession`
- OpenAI-compatible Chat Completions through `OpenAIChatCompletionsModel`
- Six structured workspace tools, executed by HANS only when the SDK emits a tool call:
  - `list_directory(path)` lists bounded, sorted direct entries with their type
  - `search_files(query, path, max_results)` performs bounded literal text search through relevant files
  - `read_file(path, start_line, end_line)` reads a UTF-8 file or an explicit bounded line range
  - `replace_in_file(path, old_text, new_text)` makes one exact, single-occurrence replacement
  - `write_file(path, content)` creates or replaces a UTF-8 file inside the workspace
  - `run_command(command)` runs one direct command with the workspace as its working directory
- Startup line showing the configured model name and endpoint host (the API key is never printed)
- Tool diagnostics: `[tool_called] name=... arguments=...` and `[tool_output]` with the Python result
- Model and tool errors printed without terminating the TUI

Ordinary model text is never parsed as a tool call. There is no custom agent loop, tool-call
parser, or conversation store in HANS.

## Configuration

```bash
export BOLT_MODEL_BASE_URL="https://your-endpoint.example/v1"
export BOLT_MODEL_API_KEY="local-key"
export BOLT_MODEL="qwen3.6-27b"
export BOLT_WORKSPACE="$PWD"              # optional; defaults to the current directory
```

Leave reasoning configuration unset unless the configured endpoint explicitly supports and
requires it. For example, an endpoint that requires tool calls without reasoning can use:

```bash
export BOLT_MODEL_REASONING_EFFORT=none
```

Credentials are read from the environment and are not embedded in source code. A local ignored
`.env.local` file can be loaded with:

```bash
set -a
source .env.local
set +a
```

The verified remote endpoint is a Cloudflare Quick Tunnel in front of Kaggle `llama-server`
running Qwen3.6-27B. Set `BOLT_MODEL_BASE_URL` to that tunnel's `/v1` URL. Do not replace it with
a local model, Ollama, or an OpenAI-hosted model when you intend to exercise the Kaggle path.

On this network the tunnel is reached through the corporate proxy. Export it before starting HANS;
do not unset it for that path:

```bash
export http_proxy=http://www-proxy.us.oracle.com:80
export https_proxy=http://www-proxy.us.oracle.com:80
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

Unset those variables only when the endpoint is reachable directly and the proxy is what blocks it.

## Install

Python 3.12 or newer is required. This installs `hans` into `~/.local/bin`:

```bash
curl -LsSf https://raw.githubusercontent.com/saurabhahuja71/hans/main/install.sh | bash
```

The installer accepts `python3.12` or newer even when `python3` points to an older
system Python. If [uv](https://docs.astral.sh/uv/) is installed, it automatically
downloads a private Python 3.12 when no suitable interpreter is available. Otherwise,
install Python 3.12+ (or uv) and rerun the command.

If `~/.local/bin` is not on `PATH`, the script prints the one `export` to add. Behind the corporate proxy, export `https_proxy` before running the command. The model URL and API key are still set in the shell; the installer does not embed them.

## Fresh installation

HANS requires Python 3.12. From a new checkout:

```bash
cd /home/sauahuja/bolt-next
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python -m pip install -U pytest
```

`pip install -e .` installs the `hans` console script and the `openai-agents` dependency
(`openai-agents>=0.22.3`).

## Upgrade

Quit HANS with Ctrl-Q, then run:

```bash
hans upgrade
```

`hans update` is an equivalent alias. The command downloads the canonical installer over HTTPS and runs it, so the same installation location and PATH guidance apply.

For a trusted HTTPS mirror or controlled test, set `HANS_INSTALLER_URL` to the installer URL before running the command. Non-HTTPS override URLs are rejected.

Versions released before `hans upgrade` existed do not contain this command. Those installations must run the install command above once to acquire it; subsequent upgrades can use `hans upgrade`.

## Run

With the environment activated and the model variables set:

```bash
hans
```

On a TTY, HANS starts the Textual UI by default. Set `HANS_TUI=curses` to use the preserved curses fallback.

In the Textual UI, Enter submits the prompt and Shift+Enter inserts a newline. Ctrl-D also submits and exits when the composer is empty. Ctrl-Q exits immediately, even if the composer is not empty. Ctrl-C cancels an active request and keeps HANS running. A prompt whose entire text is `exit` or `quit` exits without calling the model.

The curses fallback keeps its existing controls: Enter submits, Shift+Enter inserts a newline, Ctrl-D submits or exits when the editor is empty, Ctrl-C cancels, and Ctrl-Q exits. Piped input is still read until EOF and submitted as one message.

Example prompts:

```text
> Read main.go using your read_file tool. Then tell me exactly what the program prints. Do not modify any files.
> Create a small Go program that prints HELLO_HANS, run it, and verify the output.
```

A real read request against the configured Kaggle endpoint produced a structured `read_file` call
for `main.go`, returned the file bytes from the Python tool, and the model's next turn named
`HANS_KAGGLE_TOOL_TEST_123`. A second prompt in the same process, "What exact string did main.go
print?", was answered from the SDK session without the string being placed in that prompt.

A real create-and-run request produced `write_file` for `main.go`, then `run_command` with
`go run main.go`. The tool result was `exit_code=0` and stdout `HELLO_HANS`. The model's following
message verified that output. `go` must be on `PATH` for that command to succeed.

## Workspace safety

All workspace file paths resolve against the workspace root. Traversal (`../`) and symlinks that
resolve outside the workspace are rejected. Missing, unreadable, and non-UTF-8 reads are returned
as tool errors rather than raised out of the SDK loop. `list_directory` reports only direct entries,
in deterministic sorted order, as `directory`, `file`, `symlink`, or `other`; its output is bounded
to the tool-result context budget. A symlink is reported as a symlink rather than followed during
listing.

`search_files` performs a literal, line-by-line search with sorted traversal, a result limit, and
a context budget. It skips default generated and dependency directories during recursive discovery
(such as virtual environments, caches, build output, and `node_modules`), avoids directory
symlinks, and skips binary or unreadable files. An explicitly requested file remains searchable.
`read_file` returns an explicit bounded line range when needed. `replace_in_file` requires exactly
one literal occurrence and uses a temporary file replacement; use it for targeted edits. `write_file`
creates parent directories that stay inside the workspace and replaces the target file.

`run_command` does not use a shell and will not grow one silently. The command string is rejected
if it contains shell metacharacters, including pipes, redirects, `&` (`&&` / `||`), `;`, substitution
(`$`, backticks), globs, or a newline. A shell program (`sh`, `bash`, and the other common shells)
is also rejected. Otherwise the string is split with `shlex` into argv and executed directly. Safe
read-only Git inspection can use direct commands such as `git status --short`, `git diff --check`,
or `git diff`; they do not require a custom Git tool.

The working directory is the workspace. Arguments that are absolute paths outside the workspace,
or that contain a `..` segment, are rejected before the process starts. Arguments that begin with
`-` are treated as flags and are not path-checked. The command receives a reduced environment:
`PATH` and the Go, locale, proxy, and CA variables copied from the parent, plus `HOME`, `TMPDIR`,
and `PWD` set inside the workspace. API keys and the rest of the process environment are not
copied and are not printed.

This is not a complete sandbox. The process still runs as the same user. A tool such as `go` can
read its own `GOROOT` or module cache outside the workspace. There is no seccomp profile, mount
namespace, or approval prompt. The boundary is argv checking plus a reduced environment. There is
no `run_shell`.

## Development

```bash
python -m compileall src
python -m pytest -q
python -m pip check
```

The SDK tests use a scripted model. They check that structured tool calls execute and reach the
next model turn, that streaming completes, that the SQLite session keeps a later turn, and that a
later turn still runs after a model error. Workspace tests cover bounded discovery, path checks,
traversal, symlink safety, targeted replacement, and command stdout.

Do not claim a live Qwen path from the unit tests alone. The live checks above were run against
the configured tunnel with the proxy left set.

## What is not in this version

A human approval step before `run_command` is not implemented. Session history is in memory and
ends when the process exits.
