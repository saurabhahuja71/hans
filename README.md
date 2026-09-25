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
branding are HANS. Current release: **0.2.0**.

## What 0.2.0 provides

- Interactive prompt with `exit`, `quit`, EOF, and Ctrl-C handling
- Streaming assistant text through `Runner.run_streamed()` and `result.stream_events()`
- Multi-turn history for the life of the process through the SDK's in-memory `SQLiteSession`
- OpenAI-compatible Chat Completions through `OpenAIChatCompletionsModel`
- Three structured function tools, executed by HANS only when the SDK emits a tool call:
  - `read_file(path)` reads a UTF-8 file inside the workspace
  - `write_file(path, content)` creates or replaces a UTF-8 file inside the workspace
  - `run_command(command)` runs one command with the workspace as its working directory
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

## Upgrade from an older install

An existing install does not pick up new tools by pulling source alone. The `hans` command on
`PATH` still points at whatever copy was installed before 0.2.0, which only had `read_file`.

From the checkout that is already installed, upgrade in place:

```bash
cd /home/sauahuja/bolt-next
git pull
# Activate the same environment used for the old install.
# If you used this repository's venv:
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
hash -r
hans
```

That reinstall rewrites the `hans` console script and the package in the active environment.
Confirm the upgrade with:

```bash
python -c "import bolt_next.workspace as w; print(w.make_write_file_tool, w.make_run_command_tool)"
pip show hans | sed -n '1,4p'
```

`pip show` should report version `0.2.0` and the location of this checkout. If `hans` was installed
into a different environment, activate that environment and run `python -m pip install -e .` from
this checkout again. A non-editable `pip install .` from an old tree must be repeated after
`git pull`; otherwise `site-packages` keeps the old package.

## Run

With the environment activated and the model variables set:

```bash
hans
```

The banner is followed by one line of the form:

```text
model=qwen3.6-27b endpoint=https://<tunnel-host>/v1
```

Enter inserts a newline. It does not send the prompt. Ctrl-D submits the whole buffer as one user message and one Agents SDK run, with embedded newlines preserved. A paste, including a multiline paste, is therefore one turn. Ctrl-D on an empty prompt exits. A prompt whose entire text is `exit` or `quit` also exits. Piped input is read until EOF and submitted as that same single message.

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

All three tools resolve paths against the workspace root. Traversal (`../`) and symlinks that land
outside the workspace are rejected. Missing, unreadable, and non-UTF-8 reads are returned as tool
errors rather than raised out of the SDK loop.

`write_file` creates parent directories that stay inside the workspace and replaces the target
file. `run_command` does not use a shell and will not grow one silently. The command string is
rejected if it contains shell metacharacters, including pipes, redirects, `&` (`&&` / `||`),
`;`, substitution (`$`, backticks), globs, or a newline. A shell program (`sh`, `bash`, and the
other common shells) is also rejected. Otherwise the string is split with `shlex` into argv and
executed directly.

The working directory is the workspace. Arguments that are absolute paths outside the workspace,
or that contain a `..` segment, are rejected before the process starts. Arguments that begin with
`-` are treated as flags and are not path-checked. The command receives a reduced environment:
`PATH` and the Go, locale, proxy, and CA variables copied from the parent, plus `HOME`, `TMPDIR`,
and `PWD` set inside the workspace. API keys and the rest of the process environment are not
copied and are not printed.

This is not a complete sandbox. The process still runs as the same user. A tool such as `go` can
read its own `GOROOT` or module cache outside the workspace. There is no seccomp profile, mount
namespace, or approval prompt. The boundary is argv checking plus a reduced environment.

No search, SSH, or separate edit tool is included. File changes go through `write_file`. There is
no `run_shell`.

## Development

```bash
python -m compileall src
python -m pytest -q
python -m pip check
```

The SDK tests use a scripted model. They check that a structured `read_file` call executes and
reaches the next model turn, that `write_file` then `run_command` does the same, that streaming
completes, that the SQLite session keeps a later turn, and that a later turn still runs after a
model error. Workspace tests cover path checks, traversal, an outside symlink, write, and command
stdout.

Do not claim a live Qwen path from the unit tests alone. The live checks above were run against
the configured tunnel with the proxy left set.

## What is not in this version

Directory listing, search, and a human approval step before `run_command` are not implemented.
Session history is in memory and ends when the process exits.
