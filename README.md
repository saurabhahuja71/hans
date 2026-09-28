# HANS

HANS is a small, provider-neutral terminal coding assistant built on the OpenAI Agents SDK.

```text
Textual / curses terminal UI
            |
semantic HANS events and actions
            |
HansRuntime
            |
OpenAI Agents SDK Runner + SQLiteSession
            |
OpenAI-compatible Chat Completions provider
```

The SDK owns the agent loop, structured tool calling, tool results, streaming, and conversation
history. HANS owns terminal presentation, model configuration, workspace policy, workspace tools,
and the translation of SDK activity into semantic UI events. HANS does not implement a competing
agent loop, a tool-call parser, or a custom conversation store.

The Python import package remains `bolt_next`. The distribution name, console command, and runtime
branding are HANS. Current release: **0.5.1**.

## What 0.5.1 provides

- A transcript-first Textual UI with a compact header for model, connection, and workspace; inline
  tool activity; clear lifecycle state; verification and final-result presentation; and a compact,
  context-aware footer.
- A bounded, HANS-only diff view (`Ctrl-G`) and task-scoped safe undo (`Ctrl-Z`), with matching
  essential lifecycle, change, diff, undo, and composer behavior in the curses fallback.
- Streaming responses, cancellation and recovery, bounded transcript/tool rendering, and semantic
  state presentation without exposing raw SDK events.
- Six structured workspace tools, executed only when the SDK emits a tool call:
  - `list_directory(path)` lists bounded, sorted direct entries with their type.
  - `search_files(query, path, max_results)` performs bounded literal text search through relevant
    files.
  - `read_file(path, start_line, end_line)` reads a UTF-8 file or an explicit bounded line range.
  - `replace_in_file(path, old_text, new_text)` makes one exact, single-occurrence replacement.
  - `write_file(path, content)` creates or replaces a UTF-8 file inside the workspace.
  - `run_command(command, purpose)` runs one direct command with the workspace as its working
    directory.

## Configuration

Set the required endpoint and credentials before starting HANS:

```bash
export BOLT_MODEL_BASE_URL="https://your-endpoint.example/v1"
export BOLT_MODEL_API_KEY="your-api-key"
export BOLT_MODEL="qwen3.6-27b"              # optional; this is the default
export BOLT_WORKSPACE="$PWD"                  # optional; defaults to the current directory
```

`BOLT_MODEL_BASE_URL` must identify an OpenAI-compatible Chat Completions endpoint. HANS does not
contain provider-specific branches; the displayed model and provider connection state are data from
the configured runtime.

Leave reasoning configuration unset unless the configured endpoint explicitly supports and requires
it. For example, an endpoint that requires tool calls without reasoning can use:

```bash
export BOLT_MODEL_REASONING_EFFORT=none
```

### Optional settings

| Variable | Behavior |
| --- | --- |
| `BOLT_MODEL_OPENAI_PROJECT` | Optional project value supplied to the OpenAI client. |
| `BOLT_MODEL_TIMEOUT_SECONDS` | Positive request timeout in seconds; defaults to `90`. |
| `BOLT_MODEL_MAX_RETRIES` | Non-negative request retry count; defaults to `0`. |
| `BOLT_MODEL_CONTEXT_TOKENS` | Context limit used by HANS's conservative request guard; defaults to `16384` and must be at least `1024`. |
| `BOLT_MODEL_MAX_COMPLETION_TOKENS` | Optional positive completion limit. HANS caps it to the available completion reserve for the configured context. |
| `HANS_TUI` | Set to `curses` to select the curses fallback; otherwise a TTY uses Textual. |
| `HANS_DEBUG` | Set to `1` for bounded UI diagnostics. Debug output is not a replacement for tool results and does not print credentials. |

Credentials are read from the environment and are not embedded in source code. A local ignored
`.env.local` file can be loaded with:

```bash
set -a
source .env.local
set +a
```

## Install

Python 3.12 or newer is required. This installs `hans` into `~/.local/bin`:

```bash
curl -LsSf https://raw.githubusercontent.com/saurabhahuja71/hans/main/install.sh | bash
```

The installer accepts `python3.12` or newer even when `python3` points to an older system Python.
If [uv](https://docs.astral.sh/uv/) is installed, it automatically downloads a private Python 3.12
when no suitable interpreter is available. Otherwise, install Python 3.12+ (or uv) and rerun the
command.

The installer creates its environment under `~/.hans`, installs the console command, and links it
at `~/.local/bin/hans`. If that directory is not on `PATH`, the installer prints the one `export`
command to add to a shell profile. The model URL and API key remain shell configuration; the
installer does not embed them.

### Fresh checkout installation

For development or a local checkout:

```bash
git clone https://github.com/saurabhahuja71/hans.git hans
cd hans
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

`hans update` is an equivalent alias. The command downloads the canonical installer over HTTPS and
runs it, so the same installation location and PATH guidance apply. It follows the repository's
`main` branch; no GitHub release asset is required for an upgrade.

For a trusted HTTPS mirror or controlled test, set `HANS_INSTALLER_URL` to the installer URL before
running the command. Non-HTTPS override URLs are rejected. Versions released before `hans upgrade`
existed do not contain this command; run the installation command above once to acquire it, then
use subsequent upgrades normally.

The supported command forms are:

```text
hans
hans upgrade
hans update
```

Other command-line arguments are rejected with usage guidance.

## Run

With the environment activated and the model variables set:

```bash
hans
```

On a TTY, HANS starts the Textual UI by default. Set `HANS_TUI=curses` to use the preserved curses
fallback. When stdin or stdout is not a TTY, HANS reads all supplied input to EOF and submits it as
one prompt; embedded newlines are preserved.

### Controls and lifecycle

The footer shows only controls relevant to the current semantic task state:

- **Idle/composer:** Enter submits, Shift+Enter inserts a newline, and Ctrl-D submits. Ctrl-D on an
  empty composer exits. Ctrl-Q exits immediately. Ctrl-C clears an idle draft.
- **Active request:** Ctrl-C cancels the active request while leaving HANS usable for the next
  prompt. Ctrl-Q exits. Current state is shown as `INVESTIGATING`, `EDITING`, `VERIFYING`, or
  `CORRECTING` when supported by actual semantic events.
- **Completed task with HANS-owned changes:** Ctrl-G opens the bounded task diff and Ctrl-Z requests
  safe task undo. Ctrl-G and Ctrl-Z do not run while a request is active.
- **Textual diff:** Esc returns to the main task view. The curses fallback renders the same bounded
  task diff inline.
- A prompt whose entire text is `exit` or `quit` exits locally without calling the model.

The UI reports `COMPLETE`, `FAILED`, or `CANCELLED` from runtime events. Verification is
authoritative only when `run_command(..., purpose="verify")` completes; assistant prose alone does
not establish verification.

Example prompts:

```text
> Read main.go using your read_file tool. Then tell me exactly what the program prints. Do not modify any files.
> Create a small Go program that prints HELLO_HANS, run it, and verify the output.
```

### Task changes, diff, and undo

For each submitted task, HANS records successful `write_file` and `replace_in_file` mutations in a
task-local journal. This is not a full Git worktree diff: it tracks only files HANS changed during
the latest task. If a file already had a Git worktree modification when HANS changed it, the change
summary identifies it as existing work and preserves it.

`Ctrl-G` shows a HANS-only unified diff. The UI bounds rendered diff content to keep the terminal
responsive and marks truncated output. A new submitted prompt resets the task journal.

`Ctrl-Z` restores modified files to their pre-task contents and removes files created by the task,
but only if every tracked file is still exactly as HANS left it. If a tracked file changed afterward,
undo is refused for the task rather than overwriting the later change. HANS does not use destructive
Git commands for this operation.

## Workspace safety and tool limits

All workspace file paths resolve against the workspace root. Traversal (`../`) and symlinks that
resolve outside the workspace are rejected. Missing, unreadable, and non-UTF-8 reads are returned
as tool errors rather than raised out of the SDK loop. `list_directory` reports only direct entries,
in deterministic sorted order, as `directory`, `file`, `symlink`, or `other`; its output is bounded
to the tool-result context budget. A symlink is reported as a symlink rather than followed during
listing.

`search_files` performs literal, line-by-line search with sorted traversal. `max_results` defaults
to `50` and must be from `1` through `100`; results are additionally constrained by the tool-result
context budget. Recursive discovery skips common generated and dependency directories (including
virtual environments, caches, build output, and `node_modules`), avoids directory symlinks, and
skips binary or unreadable files. An explicitly requested file remains searchable.

`read_file` returns a complete small file or an explicit bounded line range. For a larger file, use
`start_line` and `end_line`; `end_line=0` selects the largest range that fits. Returned range data
identifies what remains. An explicitly requested range that does not fit returns an error and a
smaller fitting end line instead of silently returning partial source.

`replace_in_file` requires exactly one literal occurrence and uses a temporary-file replacement for
targeted edits. `write_file` creates parent directories inside the workspace and replaces the target
file. Successful mutations participate in the task-local change journal described above.

`run_command` does not use a shell and will not grow one silently. The command string is rejected
if it contains shell metacharacters, including pipes, redirects, `&` (`&&` / `||`), `;`, substitution
(`$`, backticks), globs, or a newline. Shell programs (`sh`, `bash`, and other common shells) are
also rejected. Otherwise the string is split with `shlex` into argv and executed directly. Use
`purpose="inspect"` for investigation and `purpose="verify"` only for commands that validate the
requested behavior. Commands time out after 120 seconds, and oversized output is truncated to the
context budget with a notice; rerun a narrower command for omitted output.

The command working directory is the workspace. Absolute path arguments outside the workspace and
arguments containing `..` are rejected before the process starts. Commands receive a reduced
environment with toolchain, locale, certificate, temporary-directory, and workspace settings; API
keys and the rest of the process environment are not copied or printed.

This is not a complete sandbox. The process still runs as the same user. A tool such as `go` can
read its own `GOROOT` or module cache outside the workspace. There is no seccomp profile, mount
namespace, or approval prompt. The boundary is argv checking plus a reduced environment. There is
no `run_shell`.

## Development

```bash
python -m compileall src
PYTHONPATH=.:src python -m pytest -q
python -m pip check
```

The test suite uses scripted models rather than a live provider. It covers configuration validation,
context-budget and range behavior, workspace containment and tool limits, semantic runtime events,
streaming and cancellation recovery, task-local change tracking/diff/undo, Textual and curses
controls, and HTTPS-only upgrade dispatch. Unit tests do not establish live-provider behavior.

## What is not in this version

A human approval step before `run_command` is not implemented. Session history is in memory and
ends when the process exits.
