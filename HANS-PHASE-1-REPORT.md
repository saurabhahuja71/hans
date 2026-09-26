# HANS Phase 1 — Semantic Event Boundary Report

## Status

Phase 1 is complete. The existing curses TUI now sits above a framework-neutral HANS event boundary. No Textual dependency or migration was introduced, and the OpenAI Agents SDK `Runner` remains the execution runtime.

```text
curses TUI
    ↓ HANS semantic events / runtime actions
HansRuntime adapter
    ↓ Agents SDK stream events
OpenAI Agents SDK Runner + SQLiteSession
    ↓
configured model/provider
```

- The TUI no longer imports or interprets Agents SDK stream events, `Runner`, `SQLiteSession`, provider clients, raw SDK items, or workspace tool wire formats.
- `HansRuntime` owns normal production agent/session construction, `Runner.run_streamed()`, SDK-event translation, cancellation, verification evidence, and neutral error classification.
- No custom model loop, history mechanism, compaction mechanism, tool parser, provider-specific branch, or Textual dependency was added.

## Semantic events

`src/bolt_next/events.py` introduces frozen, slotted dataclasses:

- `UserMessageSubmitted`
- `AssistantMessageDelta`
- `AssistantMessageComplete`
- `ToolStarted`
- `ToolOutput`
- `ToolCompleted`
- `RequestStarted`
- `RequestCompleted`
- `RequestFailed`
- `RequestCancelled`
- `VerificationStarted`
- `VerificationPassed`
- `VerificationFailed`
- `ConnectionChanged`
- `VerificationEvidence`

The events contain no raw Agents SDK objects.

## Runtime adapter and SDK translation

`src/bolt_next/runtime.py` adds `HansRuntime`. It invokes the existing `Runner.run_streamed()`, preserves `SQLiteSession` use, retains the existing context-budget `RunConfig`, translates streamed assistant output and tool lifecycles, produces verification evidence from authoritative `run_command` results, and exposes `submit()`, `cancel_active()`, and `close()` application operations.

Tool results remain authoritative in `ToolOutput`. The runtime alone extracts semantic metadata such as exit code for `ToolCompleted` and `VerificationEvidence`.

Request failures are classified for the UI as `context`, `connection`, or `runtime` without exposing provider-specific exception text.

## Tool-call correlation

Tool lifecycle events use the SDK’s stable `call_id`:

```text
ToolStarted(call_id=...)
ToolOutput(call_id=...)
ToolCompleted(call_id=...)
```

The adapter tracks calls by ID rather than by the last observed tool. Tests cover two interleaved calls whose outputs arrive in reverse order.

## Curses TUI changes

`src/bolt_next/tui.py` now renders HANS semantic events only.

Preserved behavior:

- Enter inserts a newline.
- Ctrl-D submits; Ctrl-D on empty input exits.
- Ctrl-C requests runtime cancellation.
- `exit` and `quit` exit without sending a model prompt.
- Streaming response display, tool activity, verification states, resize/redraw, and terminal cleanup.

The TUI does not parse `exit_code=`, `stdout:`, `stderr:`, or other workspace tool-result wire fields. It receives semantic `success` and `exit_code` values through `ToolCompleted`.

## Tests

New `tests/test_runtime_events.py` covers:

- user submission, request lifecycle, assistant deltas, and completion;
- generic failures and connection changes;
- active-request cancellation;
- tool start/output/completion;
- authoritative verification evidence;
- evidence invalidation after a successful write;
- interleaved tool calls with stable IDs; and
- static enforcement that TUI modules do not depend on `agents`, `openai`, `raw_item`, or workspace tool wire-format fields.

Updated TUI tests cover semantic-event rendering, multiline input, Ctrl-D, cancellation, verification state, and completion evidence.

## Validation

Passed:

```text
focused runtime/TUI/SDK tests: 32 passed
full test suite:               55 passed
python -m compileall -q src tests
python3.12 -m pip check
uv pip check --python /tmp/hans-python312/bin/python
git diff --check
```

No full-suite stall occurred.

## Files changed

Phase 1 implementation:

- `src/bolt_next/events.py`
- `src/bolt_next/runtime.py`
- `src/bolt_next/tui.py`
- `src/bolt_next/tui_screen.py`
- `tests/test_runtime_events.py`
- `tests/test_tui.py`

Preserved existing uncommitted Stage 4 and evaluation work:

- `README.md`
- `src/bolt_next/agent.py`
- `tests/test_sdk_runtime.py`
- `HANS-TUI-FOUNDATION-EVALUATION.md`

## Remaining limitations

- The presentation layer is still curses; it is now replaceable behind the event boundary but has not migrated to Textual.
- Connection state reflects completed request success or failure, not a separate endpoint health probe.
- The TUI retains presentation-only mappings for existing tool names to Stage 4 labels, without provider or SDK coupling.
- No live Qwen/provider call was made during this architectural phase; boundary coverage uses deterministic mocked SDK behavior.

Phase 1 is ready for Phase 2: implementing Textual behind the same `HansRuntime` event boundary.

No commit or push was made.
