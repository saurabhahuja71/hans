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
OpenAI-compatible configured model transport
```

The SDK owns the agent loop, structured tool calling, tool results, streaming, and conversation
history. HANS owns terminal presentation, model configuration, workspace policy, workspace tools,
and the translation of SDK activity into semantic UI events. HANS does not implement a competing
agent loop, a tool-call parser, or a custom conversation store.

The Python import package remains `bolt_next`. The distribution name, console command, and runtime
branding are HANS. Current release: **0.8.0**.

## What 0.8.0 provides

- `Ctrl+R` during an approval approves the current tool request and temporarily allows all read,
  write, execute, and external-path permissions. While HANS is idle, it toggles those permissions
  and restores the previous process-local per-category policy on the next press. `Ctrl+M` switches
  between native terminal selection/copy mode and HANS mouse-wheel scrolling mode. In the Textual
  composer, `Ctrl+A` selects the full draft so it can be replaced.
- The Textual `/` command-completion panel includes every registered command, stays bounded and
  scrollable, follows Up/Down selection, and supports PageUp/PageDown selection jumps.
- Runtime-enforced, session-scoped read, write, execute, and external-path permissions through the
  local `/permissions` command. All four default to `allow`; existing workspace and command safety
  restrictions remain mandatory.
- Filesystem discovery and inspection use `list_directory`, `search_files`, and `read_file`; external
  paths follow the current external-path policy. `run_command` remains a direct, workspace-confined
  command runner and does not expand shell syntax or home paths.
- `/clear` clears the active SDK `SQLiteSession` conversation history while preserving workspace
  files and local HANS controls.
- A local, secret-free configured model catalog: `/models` shows configured profiles and
  `/models use <id>` changes only to an explicit profile while idle.
- A model switch atomically prepares the new model stack and starts a fresh conversation; prior
  history is never copied or silently reused. Failed switches leave the active model and session
  unchanged.
- `/mode` validates a session-scoped reasoning override against declared model capabilities and
  applies it only to later requests. `/compact` remains unimplemented.
- The `ask` category policy pauses eligible tools for an explicit terminal approval. Approval prompts
  show bounded, redacted tool details rather than tool payloads. External paths use their own policy:
  `ask` requires one exact-operation approval with no permanent trust, `deny` blocks I/O without a
  prompt, and `allow` runs canonical external filesystem targets without external approval unless the
  category or original SDK policy requires one.
- Failed tool rows identify the safe operation/target and a bounded, redacted reason. HANS does not
  automatically retry failed or denied tools; inspect output and submit a new task as needed.
- `/help` is a local guide to configuration, session, safety, and workspace controls. In an idle
  Textual or curses composer, type `/` for bounded local command and argument suggestions, including
  `/exit` and `/quit`; use Up/Down or PageUp/PageDown and Enter or Tab to select, or Esc to dismiss.
  Configured model IDs and declared reasoning modes are suggested without sending a model request.

## Configuration

HANS uses locally configured model profiles; it does not discover models from a remote service.
Set `BOLT_MODEL_PROFILES` to a comma-separated list of profile IDs. IDs are normalized to lowercase
and must match `[a-z][a-z0-9-]*`. `BOLT_MODEL_ACTIVE_PROFILE` selects one configured profile; when
it is unset, HANS uses the first profile.

For a profile ID, replace hyphens with underscores and uppercase it to form the variable prefix:
`BOLT_MODEL_PROFILE_<UPPERCASE_ID_HYPHENS_AS_UNDERSCORES>_`. Each profile requires these suffixes:

| Suffix | Behavior |
| --- | --- |
| `MODEL` | Model name sent by the runtime. |
| `BASE_URL` | Endpoint for the selected transport on that locally configured profile. |
| `API_KEY` | Credential for that profile; it is never shown in the UI. |

The following suffixes are optional:

| Suffix | Behavior |
| --- | --- |
| `DISPLAY_NAME` | Safe human-readable name shown by `/models`; defaults to the profile ID. |
| `ENDPOINT_PROFILE` | Safe endpoint label shown by `/models`; use this rather than the base URL. |
| `TRANSPORT` | `chat_completions` (default) or `responses`. Select the transport required by the configured provider endpoint. |
| `SUPPORTS_IMAGE_INPUT` | `true` or `false` (default). Set `true` for any configured profile whose selected model and transport accept image input. |
| `CONTEXT_TOKENS` | Context limit used by HANS's conservative request guard; defaults to `16384` and must be at least `1024`. |
| `MAX_COMPLETION_TOKENS` | Optional positive completion limit, capped to the available completion reserve. |
| `REASONING_MODES` | Comma-separated declared reasoning modes. If unset, `/mode` reports support as not declared and does not permit overrides. |
| `REASONING_NONE_SEMANTICS` | `literal` (default) sends declared `none` as the effort; `omit` omits reasoning when a declared `/mode none` override is selected. |
| `REASONING_EFFORT` | Configured default reasoning effort. |
| `OPENAI_PROJECT` | Optional project value supplied to the client. |
| `TIMEOUT_SECONDS` | Positive request timeout in seconds; defaults to `90`. |
| `MAX_RETRIES` | Non-negative request retry count; defaults to `0`. |

When `BOLT_MODEL_PROFILES` is unset, HANS retains its legacy single-profile configuration using
`BOLT_MODEL`, `BOLT_MODEL_BASE_URL`, and their related legacy `BOLT_MODEL_*` values. Legacy configuration
enables image input by default; set `BOLT_MODEL_SUPPORTS_IMAGE_INPUT=false` only when the configured
model or transport cannot accept image input. This generic setting enables the SDK-native `read_image`
tool without identifying a provider. `BOLT_WORKSPACE` is optional and defaults to the current directory.

`/models` displays only safe, local catalog metadata: the active profile, display name, endpoint
profile, context limit, declared reasoning modes, image-input support, and the active effective
reasoning mode. It never displays credentials or base URLs. `/models use <id>` is available only while HANS is idle. It starts
a fresh conversation on the selected local profile and resets reasoning to that profile's configured
default; HANS does not migrate conversation history. The workspace, permissions, TODOs, theme, and
other local HANS controls persist.

| Variable | Behavior |
| --- | --- |
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

`hans update` is an equivalent alias. Normal upgrades query the stable [GitHub Releases](https://github.com/saurabhahuja71/hans/releases)
latest-release API over HTTPS. If the installed distribution already matches or is newer than that
release, HANS exits successfully without reinstalling or downgrading. Otherwise it downloads the
installer and source archive pinned to the immutable release tag, then preserves the same installation
location and PATH guidance.

For a trusted HTTPS mirror or controlled test, set `HANS_INSTALLER_URL` to the installer URL before
running the command. This direct override must use HTTPS and does not query GitHub Releases.
Versions released before `hans upgrade` existed do not contain this command; run the installation
command above once to acquire it, then use subsequent upgrades normally.

The supported command forms are:

```text
hans
hans upgrade
hans update
```

Other command-line arguments are rejected with usage guidance.

## Maintainer releases

Run the focused tests and choose the release increment:

```bash
python scripts/release.py patch
# or: python scripts/release.py minor
```

The helper updates `pyproject.toml` and the README release version together, then prints the manual
Git commands. It never commits, tags, or pushes. Review the generated diff, commit the version
change, create the matching `vX.Y.Z` tag, and push that tag. The tag-triggered GitHub Actions release
workflow verifies that the tag exactly matches `pyproject.toml`, runs the test suite, builds the
sdist and wheel, writes SHA-256 checksums, and creates the GitHub Release with those artifacts.

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
  empty composer exits. Ctrl-Q exits immediately. Ctrl-C clears an idle draft. Ctrl-B cycles session
  themes and Ctrl-T toggles the compact local TODO view; neither submits a request. Ctrl+R
  temporarily allows read, write, execute, and external-path permissions; press it again to restore
  the previous process-local per-category policy, including external paths.
- **Approval:** Press `y` to allow only the displayed request or `n` to deny it. Press `Ctrl+R` to
  allow the current request and all permission categories; use Ctrl+R again when idle to restore the
  prior policies. Type the shortcut or key itself, not a word such as `Approved`.
- **Active request:** Ctrl-C cancels the active request while leaving HANS usable for the next
  prompt. Ctrl-Q exits. Ctrl+R is disabled except at an approval. Current state is shown as
  `INVESTIGATING`, `EDITING`, `VERIFYING`, or `CORRECTING` when supported by actual semantic events.
- **Copying and scrolling response text (Textual UI):** HANS starts in **selection mode**, which
  disables application mouse reporting so MATE/native drag selection, right-click Copy, and the
  terminal's usual Ctrl+Shift+C can work. Press **Ctrl+M** to enter **scroll mode**, which enables
  HANS mouse-wheel scrolling; press it again to return to native selection mode. `HANS_MOUSE=1`
  starts in scroll mode. Ctrl-Y copies the bounded, displayed representation of the most recent
  assistant response or open detail view through Textual's terminal clipboard request. HANS reports
  that request rather than claiming desktop clipboard acceptance. The `HANS_TUI=curses` fallback
  sends the same text through terminal OSC 52 and likewise reports only that it sent the request to
  the terminal.
- **Completed task with HANS-owned changes:** Ctrl-G opens the bounded task diff and Ctrl-Z requests
  safe task undo. Ctrl-G and Ctrl-Z do not run while a request is active.
- **Textual diff:** Esc returns to the main task view. The curses fallback renders the same bounded
  task diff inline.
- A prompt whose entire text is `exit`, `quit`, `/exit`, or `/quit` exits locally without calling the model.

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

The workspace is the default boundary for filesystem tools. Paths resolving within it work normally.
HANS canonically resolves traversal (`../`), `~/...` paths expanded from the runtime `HOME`, and
symlinks before applying the external-path policy. An external target runs by default under `allow`,
requires one bounded, redacted approval for that exact canonical tool call under `ask`, and is blocked
before I/O under `deny`. A category-level `deny` always wins. Missing, unreadable, and non-UTF-8 reads
are returned as tool errors rather than raised out of the SDK loop. `list_directory` reports only direct
entries, in deterministic sorted order, as `directory`, `file`, `symlink`, or `other`; its output is
bounded to the tool-result context budget. A symlink is reported as a symlink rather than followed
during listing.

`search_files` performs literal, line-by-line search with sorted traversal. `max_results` defaults
to `50` and must be from `1` through `100`; results are additionally constrained by the tool-result
context budget. Recursive discovery skips common generated and dependency directories (including
virtual environments, caches, build output, and `node_modules`), avoids directory symlinks, and
skips binary or unreadable files. An explicitly requested file remains searchable.

`read_file` returns a complete small UTF-8 file or an explicit bounded line range. For a larger file,
use `start_line` and `end_line`; `end_line=0` selects the largest range that fits. Returned range data
identifies what remains. An explicitly requested range that does not fit returns an error and a
smaller fitting end line instead of silently returning partial source. Binary data and supported image
files return a safe error rather than a UTF-8 decoder exception.

`read_image` is available only to a profile that declares image-input support. It accepts PNG, JPEG,
WebP, and GIF after matching the filename extension and file signature, limits input to 10 MiB, 8192
pixels per dimension, and 32 million pixels total, then passes a bounded local data URL to the OpenAI
Agents SDK as actual image input. It does not OCR or modify the original file. Image token use is
provider-dependent and is not fabricated by HANS's text context accounting. An external image follows
the external-path policy: `ask` requires exact approval, `allow` reads directly, and `deny` blocks it.

`replace_in_file` requires exactly one literal occurrence and uses a temporary-file replacement for
targeted edits. `write_file` creates parent directories inside the workspace and replaces the target
file. Successful mutations participate in the task-local change journal described above. External
writes are revalidated before mutation and are excluded from the task journal and diff.

The five filesystem tools apply this policy to paths resolving outside the workspace. `ask` approval
applies only to that exact invocation and creates no permanent trust. This does not change
`run_command`, which remains workspace constrained.

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
read its own `GOROOT` or module cache outside the workspace. There is no seccomp profile or mount
namespace. An approval prompt is a human control, not a security boundary; the boundary is argv
checking plus a reduced environment. There is no `run_shell`.

## Development

```bash
python -m compileall src
PYTHONPATH=.:src python -m pytest -q
python -m pip check
```

The test suite uses scripted models rather than a live provider. It covers configuration validation,
context-budget and range behavior, workspace containment and tool limits, semantic runtime events,
streaming and cancellation recovery, task-local change tracking/diff/undo, text and image workspace
handling, Textual and curses controls, and HTTPS-only upgrade dispatch. Unit tests do not establish
live-provider or desktop-clipboard behavior.

## What is not in this version

`/compact` is not implemented. Session history is in memory and ends when the process exits.
