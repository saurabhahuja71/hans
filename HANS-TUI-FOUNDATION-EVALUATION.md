# HANS Python TUI Foundation Evaluation

## Scope

This is an architectural evaluation. It recommends a Python-native TUI foundation for HANS without replacing the OpenAI Agents SDK runtime, provider handling, workspace tools, context budgeting, or `SQLiteSession`.

It records the baseline assessed before Phase 1. Phase 1 subsequently introduced the semantic event boundary described below while retaining the existing curses presentation layer.

No TypeScript, Node.js, JavaScript frontend, Node subprocess, or `pi-tui` is appropriate for this Python application.

## Current TUI assessment

HANS currently has a custom `curses` implementation and no third-party TUI framework dependency. Its custom implementation includes:

- a multiline `Editor` with Enter inserting a newline and Ctrl-D submitting;
- custom terminal/signal lifecycle and key decoding;
- a transcript renderer with tool, status, error, and streamed-assistant rows;
- tail-following transcript display and resize redraw;
- direct consumption of OpenAI Agents SDK stream events;
- direct `Runner.run_streamed()` and `SQLiteSession` ownership in the TUI module.

### Strengths

- Existing multiline, cancellation, streaming, tool activity, verification, and clean-exit behavior.
- Rendering starts from semantic transcript data rather than preserving terminal escape output, so it can reflow after resize.
- A non-TTY fallback is available.

### Weaknesses

- The editor does not have cursor movement, insertion at cursor, selection, history, word movement, robust paste support, or an editor viewport.
- The transcript is tail-following only; the user cannot scroll back to inspect earlier output.
- Text wrapping uses Python character count rather than terminal display width, causing incorrect behavior for wide Unicode and grapheme clusters.
- Terminal lifecycle, raw/cbreak handling, resize behavior, rendering, and key handling are maintained by HANS.
- The UI imports SDK/runtime classes, invokes `Runner.run_streamed()`, owns `SQLiteSession`, maps exact HANS tool names, and parses workspace tool-result text. This makes the TUI coupled to runtime/application details.
- Tool output is associated with the most recently observed tool call, which assumes serialized tool activity and is unsafe for future interleaved events.
- Tests are mainly unit tests; there is no framework-level pseudo-terminal, layout, scrollback, focus, or resize test facility.

## Candidate comparison

| Area | Textual | prompt_toolkit | Rich + prompt_toolkit | Urwid |
|---|---|---|---|---|
| Python 3.12 | Current releases support it | Current releases support it | Supported through both libraries | Current releases support it |
| Multiline editor | First-class `TextArea` | Excellent `PromptSession(multiline=True)` and buffers | Same prompt_toolkit editor | Requires more custom assembly |
| Keyboard handling | Strong bindings and actions | Best low-level `KeyBindings` control | Same as prompt_toolkit | Capable, more imperative |
| Ctrl-C / Ctrl-D | Bindable at application/widget level | Direct, precise control | Same as prompt_toolkit | Input filters and handlers |
| Streaming / live updates | Strong incremental widget updates | Supported, but more application code | Rich formats well; renderer coordination is difficult | Supported through walker updates |
| Scrollback and panes | Native scrollable widgets and layouts | Possible, but application-owned | Still requires prompt_toolkit layout and transcript work | Scrollable lists, more manual streaming behavior |
| Resize | Framework layout/reflow | Supported, more manual | Same prompt_toolkit complexity | Supported |
| asyncio | Strong worker model | Native `run_async()` | Inherits prompt_toolkit model | `AsyncioEventLoop` available |
| Testing | Built-in headless testing, Pilot, snapshots | More bespoke harness | Adds integration concerns | More custom fixtures |
| Complexity for HANS | Low-medium | Medium | Medium-high | Medium-high |
| Overall | **Recommended** | **Runner-up** | Not a primary architecture | Credible third option |

## Textual assessment

**Recommendation: adopt Textual as the HANS TUI foundation.**

HANS is a persistent coding-agent application with a conversation transcript, tool and verification status, streamed output, a multiline composer, scrolling, cancellation, and responsive layout. Textual matches those requirements directly:

- `TextArea` supplies a real multiline editor.
- `RichLog` or a dedicated scrollable transcript widget supports incrementally appended output and bounded retention.
- Textual layouts express the fixed composer, transcript, tool/status, and header regions without custom screen arithmetic.
- Bindings and events replace custom terminal key decoding.
- Async workers allow network/SDK work to remain outside the UI path.
- Built-in headless testing supports input, resize, layout, and snapshot regression coverage.

### What Textual replaces

- Custom `curses` setup, cleanup, redraw, resize, and raw-input handling.
- The minimal custom editor.
- Custom viewport/scrollback mechanics.
- Most terminal layout and rendering primitives.

### What Textual does not replace

- OpenAI Agents SDK `Runner`.
- `SQLiteSession`.
- HANS workspace/security tools.
- Context-budget filtering.
- Verification policy and authoritative verification evidence.
- Provider/model configuration or provider adapters.
- HANS event translation.

### Risks and mitigations

- Textual is a substantial new dependency and introduces a widget lifecycle: migrate behind an event boundary, not directly against SDK events.
- Rendering each model token individually can cause excess redraws: batch/coalesce deltas at a bounded cadence.
- Keyboard binding precedence must be designed carefully: write direct tests for Ctrl-D, Ctrl-C, focus, and scroll behavior.
- Transcript retention must be bounded for long sessions while preserving user scroll position.

Expected result: substantially stronger long-running-session behavior, editing, scrollback, resize reliability, and UI testability with less terminal infrastructure maintained in HANS.

## prompt_toolkit assessment

**Runner-up: prompt_toolkit.**

prompt_toolkit is a strong choice when the primary product requirement is a highly capable terminal editor with maximum low-level control. It provides mature multiline editing, history, completion, cursor behavior, Emacs/Vi editing modes, flexible key bindings, and native asyncio integration.

Its tradeoff is that HANS would still own more UI machinery: transcript behavior, scroll-follow policy, panes, layout, streamed rendering, and a larger portion of the testing harness. It can meet all requirements but is less direct than Textual for this pane-oriented coding-agent application.

Choose it instead of Textual only if a deeply customized terminal editor is a deliberate priority over reducing HANS-owned TUI infrastructure.

## Rich + prompt_toolkit assessment

Rich plus prompt_toolkit should not be considered a separate primary framework choice.

Rich is valuable for formatting Markdown, tables, syntax, tracebacks, and renderables. Its `Live` renderer is not an input editor, focus manager, event router, or durable interactive transcript. Running Rich Live independently alongside prompt_toolkit risks competing renderer lifecycles and redraw artifacts.

If prompt_toolkit is chosen, use it as the sole terminal renderer. Rich may be used selectively to create content renderables, not as a second terminal application loop.

## Urwid assessment

Urwid is mature and viable. It offers scrollable list widgets, resize/input support, and asyncio integration. It is a reasonable option for a team already experienced with its widget/canvas model.

For HANS, it requires more custom work to build a refined multiline composer and to manage streamed transcript updates, invalidation, retention, and scroll anchoring. Textual and prompt_toolkit offer a clearer fit for the desired modern coding-agent experience.

## Why pi-tui is not appropriate

`pi-tui` belongs to a TypeScript/Node ecosystem. Adopting it would require a Node runtime, npm/TypeScript supply chain, and either an IPC boundary or a non-Python frontend.

That would create a second runtime, event loop, test stack, package manager, lifecycle model, and operational failure boundary. HANS should remain a single Python application. `pi-tui` is not being recommended or introduced.

## Proposed HANS architecture

```text
┌──────────────────────┐
│   Textual TUI        │
│  input + rendering   │
└──────────┬───────────┘
           │ Input actions / semantic application events
┌──────────▼───────────┐
│ HANS application     │
│ runtime adapter      │
│                      │
│ - turn lifecycle     │
│ - event translation  │
│ - verification state │
│ - error categories   │
└──────────┬───────────┘
           │ SDK calls / provider-independent configuration
┌──────────▼───────────┐
│ OpenAI Agents SDK    │
│ Runner + SQLiteSession│
└──────────┬───────────┘
           │
┌──────────▼───────────┐
│ Provider/model layer │
│ Qwen/OpenAI/Grok/... │
└──────────────────────┘
```

The future runtime adapter should emit stable HANS semantic events such as:

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

Boundary rules:

- The TUI consumes HANS semantic events and emits input/cancellation actions only.
- The HANS runtime adapter owns `Runner.run_streamed()`, session lifecycle, SDK event translation, cancellation, verification evidence, and error classification.
- The provider/model layer owns provider-specific client/model setup.
- The TUI must not import Agents SDK event classes, parse tool wire formats, inspect provider fields, or depend on a model/provider.
- Tool events should carry stable call IDs, not rely on a "last seen tool" assumption.
- The runtime must not write directly to stdout; terminal output belongs only to the TUI.

This architecture permits switching among Qwen, OpenAI, Terra, Grok, Gemini, Kimi, Mistral, and compatible local servers without changing the TUI or HANS application behavior.

## Migration plan

### Phase 1: Introduce the event boundary

- Define immutable HANS application event types.
- Translate SDK stream events in a runtime adapter.
- Keep the existing curses renderer consuming those events initially.
- Add stable call IDs for tool event correlation.

### Phase 2: Implement Textual behind the boundary

- Add Textual.
- Build a minimal app shell: header, transcript, tool/status region, and composer.
- Test it using fake semantic event streams, without SDK/network dependencies.

### Phase 3: Move input/editor

- Replace the custom editor with `TextArea`.
- Preserve Enter = newline, Ctrl-D = send, Ctrl-D on empty = exit, Ctrl-C cancellation policy, and `exit`/`quit` behavior.

### Phase 4: Move streaming/conversation rendering

- Render assistant deltas incrementally.
- Coalesce deltas at a bounded cadence.
- Bound retained transcript items and preserve user scroll position.

### Phase 5: Move tool/status rendering

- Render tool lifecycle, verification evidence, errors, and connection states from semantic events.
- Keep tool results authoritative in HANS/runtime; do not make UI text parsing the source of truth.

### Phase 6: Move cancellation/resize/scroll handling

- Connect UI actions to runtime cancellation.
- Validate resize, reflow, focus, scroll anchoring, terminal restoration, and clean shutdown.

### Phase 7: Remove the custom terminal implementation

- Remove curses-specific loop, terminal/signal mechanics, custom renderer, and custom editor only after parity tests pass.
- Keep framework-neutral domain/transcript types where useful.

## Validation plan

Automated tests after migration must cover:

- Multiline input.
- Enter inserts a newline.
- Ctrl-D submits a single prompt.
- Ctrl-D on empty input exits.
- Ctrl-C while idle and during an active request.
- `exit` / `quit` behavior.
- Streaming assistant output.
- Multiple correlated tool calls and tool output.
- Verification failure, repair, and success.
- Long transcript scrolling and preserved user scroll position.
- Bounded long tool output and transcript retention.
- Terminal resize while idle and while streaming.
- Model/request error, tool error, and context-budget error.
- Long-running session memory behavior.
- Cancellation and clean terminal shutdown.
- Runtime adapter tests with mocked SDK stream events.
- TUI tests with a fake HANS runtime async event iterator and no live provider.

### Real Qwen smoke test

When the Qwen3.6/Kaggle endpoint is configured, run HANS in a real TTY against a small intentionally broken repository and submit:

```text
Find the bug, fix it, run the relevant tests, and verify the fix.
```

Verify multiline submission, streamed rendering, scrollback, tool/verification states, a failed test followed by repair and passing verification, Ctrl-C cancellation, and clean exit.

## Sources

- [Textual TextArea](https://textual.textualize.io/widgets/text_area/)
- [Textual RichLog](https://textual.textualize.io/widgets/rich_log/)
- [Textual workers](https://textual.textualize.io/guide/workers/)
- [Textual testing](https://textual.textualize.io/guide/testing/)
- [prompt_toolkit asyncio integration](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/asyncio.html)
- [prompt_toolkit key bindings](https://python-prompt-toolkit.readthedocs.io/en/stable/pages/advanced_topics/key_bindings.html)
- [Rich Live](https://rich.readthedocs.io/en/stable/live.html)
- [Urwid asyncio event loop](https://urwid.readthedocs.io/en/latest/reference/main_loop.html)
- [Pi monorepo](https://github.com/badlogic/pi-mono)
