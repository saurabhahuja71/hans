# HANS Phase 9: Session & Runtime Controls — Architecture Report

This was an architecture/discovery phase only. No Phase 9 commands were implemented, and no source or test files were changed.

## Verification baseline

Verification completed with the project virtual environment:

- `pytest`: **142 passed**
- `compileall`: passed
- `pip check`: passed
- `uv pip check`: passed
- `git diff --check`: passed
- Installed SDK: **openai-agents 0.22.3**

The architecture should remain:

```text
Textual / curses
  → HANS semantic actions and events
  → HansRuntime
  → OpenAI Agents SDK
  → provider/model configuration
```

No custom agent loop, parallel session system, provider-specific UI behavior, or fake local implementations are warranted.

## 1. Current session architecture

`HansRuntime` owns the live SDK session and agent lifecycle.

- `HansRuntime` lazily creates one Agent and one `SQLiteSession` for the runtime lifetime in `src/bolt_next/runtime.py:176-355`.
- The session is created as `SQLiteSession("hans-tui")`.
- Without an explicit database path, SDK `SQLiteSession` uses in-memory SQLite, so conversation history ends when the HANS process exits.
- `submit()` prevents concurrent active requests, starts a task journal, lazily constructs the Agent, then calls `Runner.run_streamed(..., session=self._session, ...)`.
- Continuation is SDK-session-backed, not a custom HANS transcript-memory implementation.
- Cancellation calls `result.cancel()` and cancels the stream waiter. SDK persistence may occur during a run, so cancellation can leave newly persisted user or intermediate items in session history.
- On error, the runtime remains usable for a later request; tests cover multi-turn history and recovery.

The current session is singular, runtime-local, in-memory, and not a user-visible session-management system.

## 2. `SQLiteSession` behavior

The installed SDK public `Session` interface provides:

- `get_items(limit=None)`
- `add_items(items)`
- `pop_item()`
- `clear_session()`

`SQLiteSession` stores JSON-serialized history keyed by `session_id`, returns chronological items, defaults to `:memory:`, and deletes messages plus session metadata for that ID through `clear_session()`. `close()` permanently makes the session unusable.

Consequences:

- `clear_session()` is a real model-history deletion operation for the active SDK session.
- `close()` is not a session-reset mechanism.
- No generic public atomic “replace history with a summary” API exists.
- `SessionSettings.limit` limits retrieved/model-visible history but does not delete stored history or compact it.
- Session mutation belongs in `HansRuntime`, must be serialized, and must be allowed only when no request is active or settling after cancellation.

## 3. Current model configuration architecture

Model configuration is environment-based and is read during Agent construction in `src/bolt_next/agent.py:44-131`.

Current configuration includes:

- `BOLT_MODEL_BASE_URL`
- `BOLT_MODEL_API_KEY`
- `BOLT_MODEL` — default `qwen3.6-27b`
- `BOLT_MODEL_OPENAI_PROJECT`
- `BOLT_MODEL_TIMEOUT_SECONDS`
- `BOLT_MODEL_MAX_RETRIES`
- `BOLT_MODEL_REASONING_EFFORT`
- `BOLT_MODEL_MAX_COMPLETION_TOKENS`
- `BOLT_MODEL_CONTEXT_TOKENS` — used by context budgeting

`create_agent()` reads this configuration, creates an `AsyncOpenAI` client, wraps it in `OpenAIChatCompletionsModel`, and constructs one Agent with its model settings and tool list.

There is currently one configured active model, not a model catalog. HANS does not have configured alternatives, provider discovery, generic capability discovery, a model-selection abstraction, or safe runtime Agent/client rebuilding from a chosen profile.

Accordingly, `/models` cannot truthfully show a multi-model selector today.

## 4. Current `ModelSettings` and reasoning implementation

HANS already exposes a generic reasoning-effort configuration path:

- `BOLT_MODEL_REASONING_EFFORT` is read in `src/bolt_next/agent.py`.
- `_model_settings()` creates `ModelSettings(reasoning=Reasoning(effort=...))` when configured.
- Completion-token configuration is sent through `extra_args`.

The SDK labels reasoning configuration as model/provider dependent. For the active `OpenAIChatCompletionsModel` adapter in SDK 0.22.3:

- `reasoning.effort` is supported.
- `reasoning.mode` and `reasoning.context` are Responses-API-specific.
- Unsupported reasoning fields are ignored with a warning unless strict validation is enabled.

Therefore `/mode` should mean the active model’s **reasoning effort**. It should not mean generic execution behavior, permissions, or a UI mode.

HANS must not assume that `none`, `low`, `medium`, and `high` are supported by every configured OpenAI-compatible endpoint.

## 5. Current tool permission and safety architecture

The Agent unconditionally includes these six tools:

- `list_directory`
- `search_files`
- `read_file`
- `replace_in_file`
- `write_file`
- `run_command`

They are normal SDK function tools in `src/bolt_next/agent.py:123-130`. Existing safety controls are mostly in `workspace.py`, not a runtime permission policy:

- Workspace confinement rejects traversal and symlink resolution outside the workspace.
- `read_file` is UTF-8-oriented and output-bounded.
- `write_file` is confined and journaled.
- `replace_in_file` requires exactly one literal match and writes atomically.
- `list_directory` is bounded and reports symlinks.
- `search_files` is bounded, literal-oriented, skips common generated directories, and avoids directory-symlink traversal.
- `run_command` does not use a shell; it rejects shell syntax, parses an argument vector, rejects unsafe paths, applies a timeout, uses a reduced environment without provider keys, and bounds output.

However:

- There is no allow/deny/ask permission policy.
- `purpose=inspect|verify` is categorization, not enforcement.
- SDK tools currently use default `is_enabled=True` and `needs_approval=False`.
- HANS has no approval event, pending-run state, approve/reject action, or resume lifecycle.
- `run_command` is not a sandbox and runs as the user; it can mutate files, so execute is at least as sensitive as direct write.
- Native write/replace actions are journaled, but arbitrary command effects are not comprehensively reversible.

Existing workspace safety must remain mandatory under every future permission profile.

## 6. `/permissions`

### Proposed semantics

`/permissions` should expose a Runtime-owned policy that controls actual tool availability and execution behavior. The initial categories should be:

- `read`: `list_directory`, `search_files`, `read_file`
- `write`: `write_file`, `replace_in_file`
- `execute`: `run_command`

### Defaults

Current behavior exposes all tools. A desired future secure default may keep read enabled while making write and execute individually controllable, ideally with approval. However, approval must not be claimed until its full lifecycle exists.

### Enforcement point

The Runtime/tool layer must enforce this policy. UI status and prompt instructions are insufficient.

A minimal implementation can enforce `allow` and `deny` through Runtime-owned SDK tool enablement or tool/Agent rebuild. A future `ask` state requires SDK `needs_approval` plus pending `RunState`, semantic events, approve/reject actions, Runner resumption, multi-approval support, and correct cancellation behavior.

### Session scope

Permission overrides should be Runtime/session-scoped:

```text
static defaults
  → active session permission override
  → actual tool exposure/execution
```

They should take effect on subsequent tool selection/execution, but policy changes must be idle-only. No profile may weaken workspace confinement, symlink defenses, command restrictions, secret protection, or output bounds.

## 7. `/models`

### Available-model discovery

Current configuration describes one active model, not a model list. HANS can safely report the active configured model and explicit non-secret metadata, but it cannot safely invent or remotely enumerate available models.

A future list needs a configured, provider-neutral, non-secret model catalog, conceptually:

```text
ModelInfo:
  id
  display_name
  endpoint/profile reference
  context_tokens
  supported_reasoning_modes
  capabilities
```

### Model switching feasibility

Switching requires rebuilding the `AsyncOpenAI` client, `OpenAIChatCompletionsModel`, Agent, and settings. It must not modify environment variables from HANS.

### Session implications

Continuing model-A history under model B is unsafe because models may differ in context limits, tool behavior, instruction adherence, reasoning support, and message compatibility.

The safest eventual behavior is idle-only selection, Runtime rebuild, and a new conversation session or an explicit session clear. HANS should not silently continue incompatible history.

For Phase 9, display-only `/models` is the safest milestone once explicit catalog configuration exists.

## 8. `/clear`

UI clearing and SDK session clearing are not equivalent.

A UI-only clear hides visible transcript content while preserving `SQLiteSession` history that remains visible to future model requests. It must never be presented as conversation deletion.

A session clear calls `clear_session()` and deletes stored SDK history for the active session.

Recommended semantics:

```text
/clear
```

means: clear the active conversation/session history used for future model requests.

The result must state that it cleared SDK conversation history and did not implicitly clear task journals, diff/undo data, todo data, model selection, reasoning override, permission policy, workspace files, or application configuration. Visible transcript handling must be explicitly stated, not implied.

A future split can provide:

```text
/clear session
/clear view
```

If bare `/clear` remains, it should alias `/clear session`, not UI-only clearing. The operation must be idle-only and must not close the session object.

## 9. `/compact`

### SDK support

SDK 0.22.3 includes `OpenAIResponsesCompactionSession` and a compaction-aware session implementation. They are specifically for OpenAI Responses API models, invoke `client.responses.compact`, validate Responses-compatible model names, and use specialized history replacement logic.

### Current HANS support

HANS uses generic `OpenAIChatCompletionsModel` and `SQLiteSession`, currently defaulting to an OpenAI-compatible Qwen profile. The SDK Responses compaction implementation is not applicable as a generic solution.

HANS’s `fit_model_input` in `src/bolt_next/context_budget.py` is request-time context protection: it trims oversized tool output from the next model input while retaining authoritative results elsewhere. It does not mutate `SQLiteSession` and is not conversation compaction.

### Recommendation

A safe generic `/compact` cannot be implemented now. Real compaction needs careful prefix selection, preservation of instructions/task state/tool causality/verification evidence, safe summary generation, atomic replacement or rollback, cancellation and concurrency handling, and privacy rules. Generic public session APIs do not provide atomic history replacement.

Do not implement `/compact` as UI clearing or as `fit_model_input`. A truthful local response is appropriate:

> Conversation compaction is unavailable for the active generic Chat Completions/SQLite session backend. Context-budget protection remains active.

## 10. `/mode`

`/mode` should mean only the selected model’s **reasoning effort**. It should not select a model, alter permissions, or mean a vague execution setting.

The current architecture has no generic capability declaration, so HANS cannot honestly offer a universal list of `none`, `low`, `medium`, and `high` for every provider/model.

A future configured model catalog should declare supported reasoning values. The Runtime can validate requests using this configuration without provider-specific UI conditions.

Precedence should be:

```text
configured default reasoning effort
  → session/runtime /mode override
  → ModelSettings used for subsequent requests
```

A `none` mode must have explicit semantics. If it means no reasoning setting, HANS should omit the setting rather than blindly sending a provider-specific literal value.

Until capabilities are explicitly configured, HANS should display the configured current effort or say support is not declared, and reject arbitrary changes.

## 11. Command syntax proposal

```text
/permissions
/permissions read allow|deny
/permissions write allow|deny
/permissions execute allow|deny

/models
/models use <configured-model-id>

/clear
/clear session
/clear view

/compact

/mode
/mode <configured-supported-value>
```

Rules:

- `/permissions` shows effective Runtime policy and tool groups.
- Do not offer `ask` before approval lifecycle support exists.
- `/models` lists configured entries only; never provider guesses or credentials.
- `/models use` accepts only configured IDs.
- `/clear` is an explicit session-history operation; `/clear view` is presentation-only.
- `/compact` currently reports local capability status without altering history.
- `/mode` shows current reasoning effort and genuinely supported configured choices.
- All commands are local and must never be sent as user prompts.

Existing `/todo clear` parsing must remain unambiguous.

## 12. Configuration versus session override precedence

Separate immutable startup configuration from session-scoped control state:

```text
static configuration/default catalog
  → Runtime session override
  → request-specific Agent/model/tool settings
```

Examples:

```text
BOLT_MODEL_REASONING_EFFORT
  → default reasoning effort

/mode high
  → Runtime/session override for later requests
```

and:

```text
configured tool-policy default
  → /permissions session override
  → actual SDK tool enablement/enforcement
```

HANS must not modify environment variables or shell configuration, and must not expose secrets in status output.

## 13. Required semantic events/actions

UI modules must not receive SDK objects or provider-specific branches. Likely Runtime-neutral operations include:

- read Runtime control status;
- change tool policy;
- clear session history;
- obtain configured model information;
- select a configured model;
- set reasoning effort;
- report compact capability status.

Likely events:

- `RuntimeControlStatus`
- `SessionCleared`
- `ModelChanged`
- `ReasoningModeChanged`
- `PermissionPolicyChanged`

If approval support is added:

- `ToolApprovalRequested`
- `ToolApprovalResolved`

The UI should render events and invoke neutral Runtime actions; it must not manipulate sessions, `ModelSettings`, `RunState`, SDK tools, or provider clients directly.

## 14. Required runtime changes

Eventual implementation requires Runtime work:

1. Runtime-owned session-control methods with idle checks.
2. Runtime-owned tool policy affecting actual tool execution.
3. A non-secret configured model catalog for `/models` listing/selection.
4. Agent/client rebuilding for configured model switches.
5. Explicit session semantics during model changes.
6. Reasoning capability validation before a `/mode` override.
7. Continued separation of request context budgeting from conversation-history operations.
8. A full approval/resume state machine before supporting `ask` permissions.

No parallel runtime or replacement memory layer is necessary.

## 15. Required TUI changes

Textual and curses currently handle only UI-local `/todo` and `/theme` through `handle_local_command()` in `src/bolt_next/tui_screen.py`.

Future UI work should parse controls as typed local intent, dispatch Runtime operations, render concise results, preserve the transcript-first design, and keep commands out of model input/history. Textual, curses, and non-TTY paths should have consistent behavior where command support is expected.

Headers should receive semantic model/session state rather than rereading environment variables. UI code must contain no `if provider == ...` or `if model == ...` rules.

## 16. Security considerations

Phase 9 controls must not weaken:

- workspace confinement;
- traversal prevention;
- symlink escape protections;
- command restrictions;
- reduced command environment;
- API-key exclusion;
- output bounds;
- credential redaction;
- task-journal integrity.

`/permissions` must enforce actual Runtime/tool behavior. Execute must be treated as sensitive because commands can modify state. `/models` and `/mode` must not disclose credentials or raw provider configuration. `/compact` must not create unsafe summaries, external persistence, or lose authoritative state. Local commands must remain outside provider/session user-message history.

## 17. Recommended implementation order

1. Define Runtime control state and neutral interfaces, including idle requirements.
2. Add truthful status-only views for permissions, active model/configuration, and reasoning effort.
3. Implement minimal enforceable `allow`/`deny` permissions.
4. Implement idle-only session clear through `SQLiteSession.clear_session()`.
5. Add explicit non-secret model catalog and capability configuration.
6. Add validated `/mode` overrides for declared supported values.
7. Evaluate model switching with Agent/client rebuild and fresh-session semantics.
8. Add approval-mode permissions only after complete interruption/resume design.
9. Defer generic compaction pending a separate safety design and review.

## 18. Explicit non-implementation list

Do not implement:

- a custom agent loop;
- a custom planner;
- a second memory/session system;
- UI-only fake permissions;
- `ask` permissions without approval/resume support;
- provider-specific TUI/runtime branches;
- remote model enumeration merely to populate `/models`;
- arbitrary freeform model selection;
- environment mutation from HANS;
- silent reuse of model-A history after selecting model B;
- `/compact` as screen clear or input trimming;
- naive `pop_item()` or clear/re-add history rewrites;
- generic use of Responses compaction for the current Chat Completions backend;
- destructive session actions during active, cancelling, or approval-paused requests;
- secret exposure in control output;
- UI redesign unrelated to the controls;
- changes to the pre-existing untracked `src/bolt_next/configuration.py`.

## 19. Test strategy for eventual implementation

Future implementation should test:

1. **Local-command isolation:** commands never become SDK/model user prompts; Textual, curses, and non-TTY parity where supported.
2. **Permission enforcement:** read/write/execute map to intended tools; denied tools are actually blocked; `run_command` cannot bypass policy; existing workspace safety always remains active.
3. **Approval lifecycle, if added:** approval event, approve/resume, reject/recover, multiple approvals, cancellation while paused, and pending-state cleanup.
4. **Session clear:** actual SDK history removal, next requests lack prior context, documented state scope is preserved, and active-run clears are rejected.
5. **Model catalog:** configured-only entries, no secrets, invalid identifiers rejected, correct rebuild behavior, and enforced session-switch policy.
6. **Mode:** unsupported values rejected, validated values map only to `reasoning.effort`, `none` behavior is explicit, and UI remains provider-neutral.
7. **Compaction:** unsupported `/compact` does not alter history; any future real compaction requires causal-preservation, rollback, cancellation, and privacy tests.
8. **Regression:** retain UI/SDK isolation, workspace safety, context-budget non-mutation, focused UI/runtime coverage, and the full suite.

## Direct answers

### Can HANS safely implement `/permissions`?

Yes, only as Runtime-enforced tool policy. A first version can safely provide `allow`/`deny` over read, write, and execute. A true approval/`ask` mode needs complete SDK interruption/resume support first.

### Can HANS safely implement `/models`?

A truthful status/list is possible only after explicit configured model metadata exists. Current configuration supplies one active model, not a list. Model switching should wait for a non-secret catalog and explicit fresh-session semantics.

### What should `/clear` mean?

It should clear active SDK conversation/session history, not merely hide the transcript. The exact non-cleared state must be disclosed. A future `/clear view` can be presentation-only.

### Can HANS implement a real `/compact` using the current SDK/session model?

No, not safely or generically. HANS uses Chat Completions plus `SQLiteSession`; SDK Responses compaction does not apply. Current context budgeting is not conversation compaction.

### Can HANS expose `/mode` generically across providers?

Only with explicit configured capability metadata. `/mode` should mean reasoning effort. The current adapter can forward `reasoning.effort`, but HANS must not assume universal supported values.

---

Files modified: `PHASE9_ARCHITECTURE_REPORT.md`

Tests modified: NONE

Commits: NONE

Pushes: NONE
