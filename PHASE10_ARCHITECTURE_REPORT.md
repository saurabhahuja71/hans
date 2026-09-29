# Phase 10 Architecture Report: Compaction and Approvals

## 1. Executive Summary

Phase 10 is an investigation only. No `/compact` command, approval mode, pending-approval state, runtime event, UI, or session-history transformation was implemented.

HANS currently uses `OpenAIChatCompletionsModel` with an SDK `SQLiteSession`. With this backend, a truthful generic conversation compaction operation is not available. The installed SDK does contain Responses-specific compaction support, but that support uses `client.responses.compact(...)`, produces Responses replay artifacts, and is not applicable to HANS's Chat Completions execution path. `SQLiteSession` itself has no official atomic replacement or compaction API. Manual summarization plus history rewrite would be an application-owned memory system and cannot be made transactionally safe with the current session interface. `/compact` should remain unavailable/deferred.

The installed SDK does support a clean future approval implementation for function tools. A tool can declare `needs_approval`, a run returns `ToolApprovalItem` interruptions, and the caller can retain a `RunState`, call `approve(...)` or `reject(...)`, then resume the same agent and SQLite session through `Runner.run(...)` or `Runner.run_streamed(...)`. This fits HANS's existing runtime boundary without a custom agent loop, but requires a future Runtime-owned `approval_pending` state and careful cancellation/control locking.

Recommended next feature phase: **Phase 10A, per-tool `ask` approval**, limited initially to one in-memory pending `RunState` and one explicit per-call decision. Do not expose persistence or “always approve/reject” in the first version. Defer actual compaction until a deliberate, independently validated migration to a Responses-compatible model/session backend is justified for all configured providers.

## 2. Current HANS Architecture

The current execution architecture remains:

```text
Textual / curses / plain terminal input
        ↓ local-command parsing and framework-neutral events
HansRuntime
        ↓ Runner.run_streamed(..., session=SQLiteSession)
Agent + OpenAIChatCompletionsModel + AsyncOpenAI
        ↓
configured OpenAI-compatible provider/model
```

`src/bolt_next/agent.py` creates an `AsyncOpenAI` client from a selected configured profile, wraps it in `OpenAIChatCompletionsModel`, and constructs one SDK `Agent` with the six HANS workspace tools:

- `list_directory`, `search_files`, `read_file` — read category
- `write_file`, `replace_in_file` — write category
- `run_command` — execute category

`src/bolt_next/runtime.py` owns the current `Agent`, `SQLiteSession`, configured model profile, context filter, and request lifecycle. Normal user messages call `Runner.run_streamed(agent, message, session=session, run_config=...)`; runtime consumes `stream_events()` and converts SDK activity into the dataclasses in `src/bolt_next/events.py`.

The TUI modules do not import or retain SDK runtime primitives. `tui_screen.py` parses local commands, and both Textual and curses consume semantic events. This boundary must remain intact for any approval implementation.

Configured models are local only. `src/bolt_next/model_catalog.py` reads either backward-compatible `BOLT_MODEL*` configuration or explicit `BOLT_MODEL_PROFILES` profiles. A profile contains secret transport settings internally, while `ModelInfo` contains only safe metadata: ID, display name, endpoint profile label, context capacity, declared reasoning modes, and `none` semantics. No remote model discovery occurs.

Current permission enforcement is Runtime-owned but binary. `apply_tool_permission_policy()` wraps each HANS tool's `on_invoke_tool`; a disallowed category returns a local denial before underlying tool execution. Existing workspace confinement, traversal and symlink safeguards, command restrictions, reduced command environment, and output limits remain tool-layer safeguards independent of the policy.

Current cancellation is request lifecycle cancellation, not pause/resume. `cancel_active()` calls `RunResultStreaming.cancel()` and cancels the active stream waiter; `submit()` clears active state in its `finally` path. The runtime currently has no retained `RunState`, no approval-pending state, and no approval events.

Phase 9 controls are idle-only where they mutate runtime state: `/permissions`, `/clear`, `/mode`, and `/models use`. Model switching uses prepare-then-commit construction and starts a fresh SQLite session, preserving permissions while resetting the model-specific reasoning override. It does not migrate conversation history.

## 3. Installed OpenAI Agents SDK Version

The installed package is **`openai-agents 0.22.3`**. Investigation used the installed source under:

```text
.venv/lib/python3.12/site-packages/agents/
```

rather than newer online documentation.

Relevant public SDK surface present in this version includes:

- `Agent`, `Runner`, `RunState`, `SQLiteSession`, `Session`, `SessionABC`, `SessionSettings`
- `function_tool`
- `ToolApprovalItem`
- `OpenAIChatCompletionsModel`
- `OpenAIResponsesCompactionSession`, `OpenAIResponsesCompactionArgs`, and `OpenAIResponsesCompactionAwareSession`
- `is_openai_responses_compaction_aware_session`

Presence of the Responses compaction classes does **not** make compaction generic or Chat-Completions-compatible.

## 4. /compact SDK Investigation

The SDK exposes a Responses-only wrapper:

```python
OpenAIResponsesCompactionSession(
    session_id,
    underlying_session,
    *,
    client: AsyncOpenAI | None = None,
    model: str = "gpt-4.1",
    compaction_mode: Literal["previous_response_id", "input", "auto"] = "auto",
    should_trigger_compaction=...,
)
```

It offers `run_compaction(args: OpenAIResponsesCompactionArgs | None = None, ...)`. Its low-level path calls the OpenAI client method equivalent to:

```python
await client.responses.compact(
    *, model=..., input=..., instructions=..., previous_response_id=..., ...
)
```

This is a `POST /responses/compact` API. The result is a Responses-specific replay artifact: retained user input plus an encrypted item with `type: "compaction"` and `encrypted_content`. It is not a portable plaintext summary usable as ordinary Chat Completions history.

The wrapper snapshots the underlying session, calls `clear_session()`, then calls `add_items(compacted_output)`. It uses an internal lock and attempts restoration on its own failure/cancellation paths, but this is not a generic transaction across arbitrary session users or processes.

The SDK also exposes a Responses server-side `ModelSettings.context_management` option, for example:

```python
context_management=[{"type": "compaction", "compact_threshold": 200_000}]
```

That is likewise Responses-specific.

HANS constructs `OpenAIChatCompletionsModel`, whose request path uses `client.chat.completions.create(**kwargs)`. It does not use `responses.compact`, does not forward Responses `context_management`, and cannot treat the encrypted Responses compaction item as regular Chat Completions conversation input. Therefore neither installed Responses compaction mechanism is applicable to HANS's current backend.

No installed SDK API was found that provides all of the following for `OpenAIChatCompletionsModel + SQLiteSession`: generic history compaction, a provider-neutral summary API, transactional session transformation, or cancellation-safe cross-backend replacement.

## 5. SQLiteSession Investigation

The SDK `SQLiteSession` constructor is:

```python
SQLiteSession(
    session_id: str,
    db_path: str | Path = ":memory:",
    sessions_table="agent_sessions",
    messages_table="agent_messages",
    session_settings: SessionSettings | dict[str, Any] | None = None,
)
```

The generic session surface is intentionally small:

```python
async get_items(limit: int | None = None) -> list[TResponseInputItem]
async add_items(items: list[TResponseInputItem]) -> None
async pop_item() -> TResponseInputItem | None
async clear_session() -> None
```

`SQLiteSession` serializes items as JSON and supports retrieval, appending, popping the most recent item, clearing, and `close()`. Its session identity is the supplied `session_id`; HANS creates a new identifier when Phase 9C switches models.

There is no official `replace_history`, `compact`, `transform_history`, transaction-scoped full-history replacement, or summary operation in this session interface. `clear_session()` and `add_items()` are distinct calls. A HANS implementation that combined them would own the consistency protocol itself.

`SessionSettings(limit=...)` limits how many latest items are retrieved/model-visible. It does not delete, summarize, reorder, or transform persisted session items. It is a retrieval/input-history limit, not conversation compaction.

## 6. Context Budget vs Compaction

HANS's `src/bolt_next/context_budget.py` protects a single request's model input through `run_config(...).call_model_input_filter = fit_model_input`.

When a tool output is too large, the model-visible copy may be replaced with a bounded omission notice. The original session item and original tool output remain intact. The notice explicitly says it is not a summary. If necessary, the filter retains only the newest request input to fit the current model context capacity.

This is deliberately distinct from conversation compaction:

| Concept | Current HANS behavior | Persistent history change |
|---|---|---|
| Context-budget protection | Shapes one request's input before sending it | No |
| Tool-output trimming | Replaces oversized model-visible output with an omission notice | No |
| Session retrieval limit | Can retrieve only the latest session items | No |
| Conversation compaction | Summarizes/replaces/restructures durable history | Not implemented |

`tests/test_context_budget.py` verifies that the original oversized item remains unchanged while the model receives a bounded representation. Calling this behavior “compaction” would be inaccurate.

## 7. Manual Compaction Safety Analysis

A manual sequence—read SQLite history, ask a model for a summary, delete history, then insert the summary—is not acceptable for production HANS on the current architecture.

The actual SDK/session semantics leave HANS responsible for all of the following:

- preserving ordered function-call and function-output relationships;
- preserving assistant item structure, any reasoning-associated items, system/developer instruction behavior, and message ordering;
- correctly accounting for tokens after a transformed history;
- preventing an in-flight request, retry rollback, or concurrent writer from observing a partially cleared session;
- handling cancellation or provider failure between summary generation, `clear_session()`, and `add_items()`;
- rolling back atomically when no official replace transaction exists;
- preventing summary/privacy leakage from tool outputs, workspace data, or credentials that may have appeared in history;
- defining model-switch behavior and ensuring a summary generated under one model is not silently reused under another;
- defining irreversible user-visible loss and undo semantics when there is no session snapshot/archive feature.

The installed SQLite session API has only separately callable clear and append methods. The absence of an official atomic full-history replacement operation is decisive here; this is not merely a theoretical risk. Implementing and maintaining the missing protocol would create a parallel HANS memory/history system, contrary to the architecture constraints.

Moving solely to the Responses API for compaction is also not a small substitution. It would require a deliberate new backend/session compatibility design and validation for streaming, function tools, cancellation, reasoning settings, model profiles, Terra and Qwen compatibility, semantic events, and existing session behavior. It must not be undertaken as a hidden compaction workaround.

## 8. /compact Recommendation

Do not implement `/compact` against the current `OpenAIChatCompletionsModel + SQLiteSession` backend.

The truthful future command behavior, if a command is desired before a backend migration, is local unavailable/deferred status such as: “Conversation compaction is unavailable for the current Chat Completions session backend.” It must not call the model, rewrite session history, or relabel input trimming as compaction.

A real future compaction design requires all of these before implementation:

1. a Responses-compatible model/session backend proven compatible with HANS's configured providers;
2. a supported SDK compaction path with understood replay artifact semantics;
3. idle-only Runtime ownership and exclusive session access;
4. explicit failure, cancellation, privacy, and irreversibility semantics;
5. focused end-to-end tests covering tool-call pairing, history integrity, context reduction, and recovery.

Until then, no user-visible `/compact` should imply that history was summarized or reduced.

## 9. Approval/Ask SDK Investigation

In SDK 0.22.3, `function_tool` supports static and dynamic approval:

```python
function_tool(needs_approval=True)
```

and a callable policy shaped as:

```python
bool | Callable[
    [RunContextWrapper[Any], dict[str, Any], str], Awaitable[bool]
]
```

The SDK also handles synchronous boolean predicate results during evaluation. Conservative behavior can require approval when arguments have been transformed/defaulted or policy inspection fails. Tool approval is therefore not a substitute for HANS tool-side authorization and workspace safety checks.

HANS's six current tools do not declare `needs_approval`. Existing binary permission wrappers remain the enforcement point for `deny`. A future Runtime-owned mapping can be:

```text
allow → no SDK approval required
ask   → SDK needs_approval evaluates true for that tool/category
 deny → HANS wrapper denies before underlying execution
```

This is only a proposed design; no `ask` behavior was added in this phase.

Read, write, and execute can be independently mapped because the current tool-to-category mapping already exists in `agent.py`. The mapping must remain Runtime-owned and provider-neutral.

## 10. Tool Approval Lifecycle

The SDK-supported lifecycle is:

```text
model emits function call
    ↓
SDK evaluates tool needs_approval
    ↓
SDK emits/records ToolApprovalItem; tool body does not run
    ↓
Runner returns an interrupted result
    ↓
Runtime retains result.to_state() and exact interruption
    ↓
user approves or rejects
    ↓
RunState.approve(...) or RunState.reject(...)
    ↓
Runner resumes same agent + same SQLiteSession
    ↓
normal tool/result and assistant processing continue
```

For non-streamed runs, `Runner.run(...)` yields a result with `result.interruptions`. For streamed runs, `Runner.run_streamed(...)` returns `RunResultStreaming`, and its interruptions are available after the caller fully consumes `result.stream_events()`.

The continuation APIs in this installed SDK are:

```python
state = result.to_state()
interruptions = state.get_interruptions()
state.approve(approval_item, always_approve=False)
state.reject(
    approval_item,
    always_reject=False,
    rejection_message: str | None = None,
)

await Runner.run(agent, state, session=same_session)
# or
Runner.run_streamed(agent, state, session=same_session)
```

A rejection creates an ordinary tool-rejection output for the model; the underlying tool body does not execute. The first HANS implementation should not expose `always_approve` or `always_reject`, even though the SDK supports the flags, because their policy scope and persistence semantics have not been designed.

`ToolApprovalItem` exposes tool call information including `name`, `qualified_name`, and `arguments`. It also contains raw call metadata. HANS must retain the exact SDK interruption/state internally; the item identity and nonempty unique call ID matter for a decision. The UI must receive only a sanitized semantic projection.

## 11. RunState / Resumability

`RunState` is the SDK pause/resume boundary. It captures the agent/input/history/model response state, pending decision state, tool invocation ledgers, usage/current resumable step, and session checkpoint information. It is not interchangeable with merely retaining a `SQLiteSession`.

The state can be serialized by SDK APIs such as:

```python
state.to_json(...)
state.to_string(...)
await RunState.from_json(initial_agent, state_json, ...)
await RunState.from_string(...)
```

Serialization is not recommended for the first HANS approval phase. It raises a new persistence and secret-handling design question; even where the SDK defaults to `include_tracing_api_key=False`, HANS must not serialize or expose state without a specific security design.

For a first TTY-only approval feature, retain one state in Runtime memory, resume it once with the same active agent and exact active SQLite session, and discard it after a decision or cancellation. Do not duplicate/copy state, resume divergent state instances, or run another normal request on the same lineage while it is pending. The SDK reconciles pending persistence batches to avoid duplicate tool execution and treats divergent/ambiguous history as an error rather than safe replay.

An interrupted approval is an interrupted completed run, not a live stream paused forever. HANS therefore needs a future explicit state machine at least:

```text
idle → running → approval_pending → resuming/running → idle
                 ↓ cancellation/exit
                 idle
```

The current runtime only distinguishes active/inactive requests and discards the streamed result after completion, so it cannot yet implement this lifecycle.

## 12. Approval + SQLiteSession Semantics

A disposable local SDK experiment used `agents.testing.ScriptedModel`, a temporary SQLite database, and a `function_tool(needs_approval=True)`. It used no credentials or provider call.

Observed results:

1. The initial `Runner.run(...)` returned one interruption.
2. The guarded tool body had not executed.
3. `await session.get_items()` contained two persisted items: `user`, `function_call`.
4. After `state.reject(interruption, rejection_message=...)` and same-session resume, it contained `user`, `function_call`, `function_call_output`, `message`; the tool body still had not executed.
5. In a separate approval run, `state.approve(...)` followed by same-session resume executed the tool body exactly once and produced the same `function_call_output`, `message` continuation shape.

Thus a user message and pending function call can already be in SQLite session history at interruption. The tool output is recorded on the resolved continuation, not before approval. This confirms why `/clear`, model switching, and a competing prompt must not mutate the active session while a state is pending.

The experiment validates the normal interruption/rejection/approval path only. It does not establish an automatic persistence rollback rule for every cancellation timing; a future implementation must explicitly invalidate and discard its own pending state when cancellation occurs.

## 13. Approval + Cancellation

The future Runtime must treat `approval_pending` as a controlled state, not as ordinary idle completion.

Required behavior:

1. While a decision is pending, Ctrl-C/cancel must atomically invalidate and discard the retained `RunState` and pending approval record.
2. The UI must leave approval state and show cancellation/termination consistently.
3. Any later approve/reject input must be rejected by Runtime validation: active state, call ID, request generation, agent/model identity, and session identity must all still match.
4. A cancelled state must never be resumed later.
5. A subsequent normal request must be possible after cleanup.
6. Exit must perform the same invalidation before closing session/runtime resources.

For an active stream before interruption, existing `RunResultStreaming.cancel()` behavior remains relevant. Once an approval interruption has completed stream consumption, there is no indefinitely active stream to cancel; it is the retained state that must be destroyed. This distinction prevents stale approvals from resuming a cancelled request.

Terminal resize must only redraw UI. It must not approve, reject, cancel, or replay a pending state.

## 14. Approval + Model Switching

Model switching must be unavailable during `approval_pending`.

The retained state is tied to the same agent/model/session lineage. Switching would create a new agent/client and fresh SQLite session, violating the continuation precondition and risking a decision being applied to the wrong model/session. The future Runtime should return a local control rejection until the pending approval is approved, rejected, or cancelled.

After resolution/cancellation, existing Phase 9C switching rules remain: prepare target first, then atomically switch to a fresh conversation session; no history migration. A pending approval must never cross that boundary.

## 15. Approval + /clear

`/clear` must be rejected while approval is pending.

The local experiment shows that the active session already contains the user message and function call at interruption. Clearing it would invalidate the state/session lineage required for SDK resume and could leave an ambiguous partial tool call. Only after approval/rejection/cancellation has resolved and removed the pending state may the existing Runtime `clear_session()` path run.

## 16. Approval + /mode

`/mode` must be rejected while approval is pending.

A pending `RunState` represents a concrete agent/model-settings execution lineage. Changing the Runtime's effective reasoning settings while retaining that continuation would make the resumed behavior ambiguous and violates HANS's existing rule that runtime controls are idle-only. After resolution or cancellation, `/mode` may use the ordinary Phase 9B behavior.

For the same reason, `/permissions` mutations and new normal prompts should be rejected during pending approval. The pending tool decision must be resolved under the policy/agent/session that initiated it; it must not silently inherit a later policy change.

## 17. Approval Security Analysis

Approval is an additional gate, not a replacement for existing protection.

| Risk | Required mitigation |
|---|---|
| Model-supplied path/command text tries to mislead the user | Treat all arguments as untrusted; render escaped, bounded, normalized, purpose-specific display fields. |
| Secret leakage in raw arguments/output | Never emit raw SDK objects or arbitrary arguments; redact/suppress secrets and cap displayed text. |
| Path traversal or symlink escape after approval | Keep all existing workspace confinement, traversal, and symlink checks in the actual tool body. |
| Dangerous command after approval | Keep command syntax restrictions, reduced environment, workspace restrictions, and output bounds. Approval cannot authorize prohibited commands. |
| Stale or wrong approval selection | Runtime validates exact pending call ID plus request generation, agent/model identity, and session identity before decision. |
| Race with cancellation/exit | Make pending-state invalidation atomic in Runtime; ignore late UI actions after invalidation. |
| Approval spoofing by text in transcript | Present a structured local approval prompt from semantic event data, never assistant text as authoritative UI. |
| Broad persistent allow from one click | Initial design is one decision per tool call; do not expose SDK always-approve/reject flags. |

Suggested safe display projections are workspace-relative normalized paths for file operations, bounded/redacted query/path information for search, and a bounded/redacted command plus purpose for execution. These display fields are informational only; the actual tool call remains subject to validation and authorization.

Compaction has separate security risks: sending full history/tool output to a summarizer, persisting a summary/replay artifact with sensitive details, unreviewable deletion of original history, and no safe undo snapshot in the current session API. Those risks reinforce the `/compact` recommendation.

## 18. Semantic Event Design

No events were added. If approval is implemented later, the existing framework-neutral dataclass boundary can support proposed events such as:

```text
ToolApprovalRequested(
    call_id,
    tool_name,
    permission_category,
    safe_display_arguments,
    requested_action,
)
ToolApprovalResolved(call_id, approved)
```

The Runtime should own the actual `RunState` and `ToolApprovalItem`; neither belongs in an event. Events must not contain an SDK object, provider client, `SQLiteSession`, Textual widget, credentials, raw arbitrary arguments, or unrestricted output.

A future Runtime action should accept a decision only through its own validation method, rather than allowing Textual or curses to call SDK `approve`/`reject` directly. This preserves the current semantic boundary and enables identical curses/Textual/plain-terminal behavior.

## 19. Textual/Curses Design

No UI was changed.

Textual already renders framework-neutral runtime events and has modal infrastructure used for existing local views. It can eventually present one structured approval prompt without provider-specific UI or direct SDK imports. Curses shares the transcript/event renderer and can use explicit approval keys/prompting. Both must dispatch the same Runtime decision action.

The first UI design should be concise and visibly local, for example showing a category, tool name, safe operation summary, and explicit approve/reject controls. It should not render a free-form assistant message as an approval request and must preserve transcript-first layout rather than add a permanent control panel.

Current UI behavior blocks ordinary submissions while a request is active. Approval pending needs its own explicit state: ordinary prompts cannot start, but a valid approve/reject/cancel action is allowed. Textual/curses must not retain SDK objects or independently decide if a tool is safe.

## 20. Non-TTY Design

The plain terminal `serve()`/`input()` path has no designed secure asynchronous approval interaction. Therefore `ask` must fail closed in non-TTY mode unless and until an explicit, tested non-TTY decision protocol is designed.

It must never silently allow a tool because no interactive channel is available. A safe first implementation may report approval unavailable/denied and resume the model with rejection, or reject `ask` configuration for non-TTY startup; the exact behavior should be selected and tested in the approval implementation phase.

## 21. Test Plan

No permanent feature tests were added in this investigation. Required tests before future implementation include:

### `/compact`

- local unavailable/deferred response makes no model/session mutation;
- if a Responses backend is ever adopted, actual SDK compaction preserves valid tool-call/output relations;
- success reduces the intended model-visible context without claiming deletion incorrectly;
- provider failure and cancellation preserve/recover history according to documented transactional semantics;
- no secret/tool-output privacy regression;
- model-switch and session identity behavior is explicit;
- no manual `clear_session()`/`add_items()` implementation masquerades as a generic safe compaction path.

### `ask` approval

- `allow`, `deny`, and `ask` category policy behavior for all six tools;
- one `ToolApprovalRequested` semantic event has safe, bounded, redacted display fields;
- approval executes exactly the selected tool once;
- rejection produces a model-visible rejection and executes no tool body;
- malformed/wrong/stale call ID cannot resolve a pending decision;
- cancellation and exit discard pending state, unblock later requests, and ignore late decisions;
- no concurrent normal request, `/permissions`, `/clear`, `/mode`, or `/models use` succeeds while pending;
- pending state uses same agent/model/session on resume;
- session history before interruption and after approve/reject remains consistent;
- workspace confinement, symlink/traversal checks, command restrictions, reduced environment, and output bounds still apply after approval;
- Textual and curses produce identical runtime operations without SDK imports;
- non-TTY `ask` fails closed;
- model-switch, clear, cancellation, and reasoning controls work normally again after resolution/cancellation.

Use `ScriptedModel`, a temporary SQLite database, and local test tools for lifecycle tests. Do not require provider credentials or network calls.

## 22. Capability Decision Matrix

| Capability | SDK support | HANS compatibility | Safe now? | Recommended approach |
|---|---|---|---|---|
| `/compact` | Responses-only compaction wrapper and endpoint exist | Not compatible with current Chat Completions model path | No | Defer; expose only truthful unavailable status if needed |
| Session compaction | `OpenAIResponsesCompactionSession` can wrap a session | Requires Responses API replay semantics, not HANS Chat Completions | No | Evaluate only after a deliberate backend migration |
| History replacement | `SQLiteSession` has clear/add only; no official atomic replacement | Manual implementation would own unsafe history transaction/memory semantics | No | Do not implement manual replacement |
| Per-tool approval | `function_tool(needs_approval=...)` and `ToolApprovalItem` exist | Fits HANS tools and Runtime-owned category policy | Yes, in a future phase | Implement single per-call `ask`, Runtime-owned |
| Resumable approval | `RunState`, `approve`, `reject`, and same-session Runner resume exist | Requires new runtime `approval_pending` state | Yes, in a future phase | Retain one in-memory state; resume same agent/session |
| Cancellation during approval | SDK state can be discarded; no custom loop required | Current runtime needs explicit pending-state invalidation | Yes, with future state machine | Ctrl-C/exit invalidates state and ignores late decisions |
| Model switching during approval | Resume requires same agent/session lineage | Switching would replace both | No | Reject model switching until approval resolves/cancels |
| `/clear` during approval | Pending call/user history is already in the session | Clear would invalidate continuation lineage | No | Reject until approval resolves/cancels |
| `/mode` during approval | SDK permits resumable state; setting mutation is HANS concern | Would change active execution lineage | No | Reject until approval resolves/cancels |

## 23. Recommended Phase 10 Implementation Order

1. **Phase 10A — Approval/Ask:** Implement only Runtime-owned category `ask`, per-tool-call approval, one in-memory `RunState`, same-session resumption, semantic events, safe display projections, and explicit pending-state cancellation. Keep non-TTY fail-closed. Do not add persistence, “always” approval, or provider-specific logic.
2. **Phase 10A validation:** Complete the approval test plan against `ScriptedModel`/temporary SQLite sessions and existing Textual/curses runtime test seams. Verify existing permissions, clear, mode, model switching, workspace safety, cancellation, and normal requests remain functional.
3. **Future compaction decision:** Do not schedule implementation until HANS has a separately approved and validated Responses-compatible backend strategy across configured providers. Re-evaluate then using the exact SDK version in use at that time.

This order follows installed SDK support: approvals have a native tool interruption/resume lifecycle compatible with HANS's architecture; generic Chat Completions compaction does not.

## 24. Explicit Non-Implementation List

This investigation did **not** implement or modify:

- `/compact` or any compact/unavailable command;
- approval/ask permissions;
- `ToolApprovalRequested` or `ToolApprovalResolved` events;
- approval UI, pending state, resume controls, or state persistence;
- session summarization, history rewrite, manual deletion, or automatic compaction;
- Agent, Runtime, TUI, event model, context-budget behavior, workspace safety, provider configuration, or model catalog behavior;
- a custom agent loop, planner, model router, second memory system, RAG, embeddings, vector database, or MCP.

No remote/provider investigation or live provider call was required. The local approval experiment used only SDK scripted models and temporary files.

## 25. Validation Results

Initial baseline before the investigation: `PYTHONPATH=.:src .venv/bin/python -m pytest -q` completed with **175 passed**.

A disposable local SDK experiment verified approval interruption/session behavior using `ScriptedModel`, a temporary SQLite database, and no credentials/network:

- interrupted session items: `user`, `function_call`;
- after rejection/resume: `user`, `function_call`, `function_call_output`, `message`, with zero tool executions;
- after approval/resume: one tool execution and the same continuation item shape.

Final repository validation after this report was written:

- `PYTHONPATH=.:src .venv/bin/python -m pytest -q`: **175 passed in 30.32s**
- `.venv/bin/python -m compileall -q src`: passed
- `.venv/bin/python -m pip check`: `No broken requirements found.`
- `uv pip check --python .venv/bin/python`: `All installed packages are compatible`
- `git diff --check`: passed
- installed package recheck: `openai-agents 0.22.3`

`git status --short` contains this new report and the pre-existing untracked `PHASE9_ARCHITECTURE_REPORT.md` and `src/bolt_next/configuration.py`. Only this report is intended as a Phase 10 change. There is no commit and no push.
