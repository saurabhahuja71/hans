"""Presentation state for the HANS terminal.

This module does not talk to the model. It records the conversation and the
prompt editor so the screen can be drawn without mixing them.
"""

from __future__ import annotations

import base64
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

from bolt_next.commands import (
    PERMISSION_CATEGORIES,
    PERMISSION_POLICIES,
    format_help,
    known_command,
)
from bolt_next.events import VerificationEvidence


def format_change_summary(summary: str) -> str:
    """Present the task journal summary without inventing another tracker."""
    fields: dict[str, str] = {}
    for line in summary.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()

    known_keys = {
        "changed_files",
        "created_files",
        "preexisting_git_worktree_changes",
        "git",
    }
    if not fields or not set(fields).intersection(known_keys):
        return summary.strip()

    def paths(key: str) -> list[str]:
        value = fields.get(key, "")
        if not value or value.lower() == "none":
            return []
        return [path.strip() for path in value.split(",") if path.strip()]

    created = set(paths("created_files"))
    changed = paths("changed_files")
    lines = [f"{'A' if path in created else 'M'} {path}  HANS" for path in changed]
    for path in paths("created_files"):
        if path not in changed:
            lines.append(f"A {path}  HANS")
    preexisting = paths("preexisting_git_worktree_changes")
    if preexisting:
        lines.append("Existing worktree changes preserved: " + ", ".join(preexisting))
    return "\n".join(lines) or "No HANS task changes"


def task_summary_has_hans_changes(summary: str) -> bool:
    """Identify HANS-owned mutations in an existing task-local summary."""
    for line in summary.splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"changed_files", "created_files"}:
            if value.strip().lower() not in {"", "none"}:
                return True
    return False


def footer_text(
    state: str, *, request_active: bool, has_task_changes: bool, approval_pending: bool = False
) -> str:
    """Return compact controls for the current semantic task state."""
    stage = state.partition("·")[0].strip() or "IDLE"
    if approval_pending:
        return "? APPROVAL REQUIRED · y approve · n deny · Ctrl-C cancel · Ctrl-Q quit"
    if request_active:
        return f"◉ {stage} · Ctrl-C cancel · Ctrl-Q quit"
    if stage == "COMPLETE":
        if has_task_changes:
            return "✓ COMPLETE · Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"
        return "✓ COMPLETE · Enter new task · Ctrl-Q quit"
    if stage == "CANCELLED":
        return "⏸ CANCELLED · Enter new task · Ctrl-Q quit"
    if stage == "FAILED":
        return "✗ FAILED · Enter retry/new task · Ctrl-Q quit"
    if has_task_changes:
        return "Ctrl-G diff · Ctrl-Z undo · Ctrl-Q quit"
    return "Enter send · Shift+Enter newline · Ctrl-D send / empty exit · Ctrl-Q quit"


def format_approval_request(tool_name: str, category: str, fields: tuple[tuple[str, str], ...]) -> str:
    details = "\n".join(f"{name}: {value}" for name, value in fields)
    prompt = f"APPROVAL REQUIRED\nApprove {tool_name} ({category})? y / n"
    return f"{prompt}\n{details}" if details else prompt


@dataclass
class Piece:
    kind: str
    text: str
    detail: str = ""
    fields: tuple[tuple[str, str], ...] = ()


@dataclass
class Transcript:
    pieces: list[Piece] = field(default_factory=list)

    def user(self, text: str) -> None:
        self.pieces.append(Piece("user", text.rstrip("\n")))

    def thinking(self) -> None:
        self.stage("INVESTIGATING")

    def stage(self, text: str) -> None:
        self._drop_status()
        self.pieces.append(Piece("status", text))

    def stream(self, delta: str) -> None:
        self._drop_status()
        if self.pieces and self.pieces[-1].kind == "assistant":
            self.pieces[-1].text += delta
        else:
            self.pieces.append(Piece("assistant", delta))

    def finish(self) -> None:
        self._drop_status()
        self.pieces.append(Piece("final", "COMPLETE"))

    def completed(self, evidence: VerificationEvidence | None) -> None:
        self._drop_status()
        if evidence is None:
            self.pieces.append(Piece("final", "COMPLETE", "Verification not established"))
        elif evidence.success:
            self.pieces.append(Piece("final", "COMPLETE", f"Verified: {evidence.command}"))
        else:
            self.pieces.append(Piece("final", "VERIFICATION FAILED", f"{evidence.command}\nHANS did not claim completion."))

    def cancelled(self) -> None:
        self._drop_status()
        self.pieces.append(Piece("error", "CANCELLED", "The request was stopped. You can send another prompt."))

    def tool_started(self, label: str) -> None:
        self._drop_status()
        self.pieces.append(Piece("tool", label, "run"))

    def tool_finished(self, label: str, *, ok: bool) -> None:
        self.pieces.append(Piece("tool", label, "ok" if ok else "fail"))

    def verification(self, command: str, *, ok: bool) -> None:
        marker = "passed" if ok else "failed"
        self.pieces.append(Piece("verification", command, marker))

    def approval(self, tool_name: str, category: str, fields: tuple[tuple[str, str], ...]) -> None:
        self._drop_status()
        self.pieces.append(Piece("approval", tool_name, category, fields))

    def change(self, text: str, *, title: str = "CHANGES") -> None:
        self._drop_status()
        self.pieces.append(Piece("change", text, title))

    def diff(self, text: str) -> None:
        self._drop_status()
        self.pieces.append(Piece("diff", text, "TASK DIFF"))

    def error(self, title: str, detail: str = "") -> None:
        self._drop_status()
        self.pieces.append(Piece("error", title, detail))

    def debug(self, text: str) -> None:
        self.pieces.append(Piece("debug", text))

    def _drop_status(self) -> None:
        if self.pieces and self.pieces[-1].kind == "status":
            self.pieces.pop()

    def render(self, width: int) -> list[str]:
        width = max(20, width)
        lines: list[str] = []
        for piece in self.pieces:
            lines.extend(_render_piece(piece, width))
            lines.append("")
        return lines[:-1] if lines else []


def _wrap(text: str, width: int, indent: str = "") -> list[str]:
    room = max(1, width - len(indent))
    if text == "":
        return [indent]
    out: list[str] = []
    for paragraph in text.split("\n"):
        if paragraph == "":
            out.append(indent)
            continue
        while len(paragraph) > room:
            out.append(indent + paragraph[:room])
            paragraph = paragraph[room:]
        out.append(indent + paragraph)
    return out


def _render_piece(piece: Piece, width: int) -> list[str]:
    if piece.kind == "user":
        rows = piece.text.split("\n") or [""]
        return ["YOU"] + [("> " if index == 0 else "  ") + row for index, row in enumerate(rows)]
    if piece.kind == "assistant":
        return ["HANS", *_wrap(piece.text, width)]
    if piece.kind == "tool":
        mark = {"run": "◇", "ok": "✓", "fail": "✗"}.get(piece.detail, "◇")
        return _wrap(f"{mark} TOOL {piece.text}", width, "  ")
    if piece.kind == "verification":
        mark = "✓" if piece.detail == "passed" else "✗"
        return _wrap(f"VERIFICATION\n{mark} {piece.text} {piece.detail}", width, "  ")
    if piece.kind == "approval":
        return _wrap(format_approval_request(piece.text, piece.detail, piece.fields), width, "  ")
    if piece.kind == "change":
        return _wrap(f"{piece.detail}\n{piece.text}", width, "  ")
    if piece.kind == "diff":
        return _wrap(f"{piece.detail}\n{piece.text}", width, "  ")
    if piece.kind == "final":
        mark = "✗" if piece.text == "VERIFICATION FAILED" else "✓"
        lines = ["FINAL RESULT", f"  {mark} {piece.text}"]
        if piece.detail:
            lines.extend(_wrap(piece.detail, width, "  "))
        return lines
    if piece.kind == "status":
        return [f"  ◌ {piece.text}"]
    if piece.kind == "error":
        lines = ["ERROR", f"  ✗ {piece.text}"]
        if piece.detail:
            lines.extend(_wrap(piece.detail, width, "    "))
        return lines
    if piece.kind == "debug":
        return _wrap(piece.text, width, "  ")
    return _wrap(piece.text, width)


class Editor:
    """Prompt editor. Enter submits; Shift+Enter inserts a newline."""

    def __init__(self) -> None:
        self.lines = [""]

    def clear(self) -> None:
        self.lines = [""]

    def replace_text(self, text: str) -> None:
        self.lines = text.split("\n") or [""]

    def on_key(self, key: str) -> str | None:
        """Return text to submit, '' to exit, or None to keep editing."""
        if key == "ctrl-c":
            self.clear()
            return None
        if key == "ctrl-q":
            self.clear()
            return ""
        if key in {"ctrl-d", "enter"}:
            text = "\n".join(self.lines)
            self.clear()
            if not text.strip():
                return ""
            return text
        if key == "shift-enter":
            self.lines.append("")
            return None
        if key == "backspace":
            if self.lines[-1]:
                self.lines[-1] = self.lines[-1][:-1]
            elif len(self.lines) > 1:
                self.lines.pop()
            return None
        if key.startswith("char:"):
            self._insert(key[5:])
        return None

    def _insert(self, text: str) -> None:
        rows = text.split("\n")
        self.lines[-1] += rows[0]
        self.lines.extend(rows[1:])

    def display_lines(self) -> list[str]:
        shown = []
        for index, line in enumerate(self.lines):
            shown.append(("> " if index == 0 else "  ") + line)
        shown.append("  ")
        return shown


def is_exit_command(text: str) -> bool:
    command = text.strip().lower()
    if command in {"exit", "quit"}:
        return True
    spec = known_command(command)
    return bool(spec and spec.exits)


def layout_rows(height: int, editor_lines: int) -> tuple[int, int]:
    """Return (conversation height, editor height) inside the chrome."""
    chrome = 4
    editor_height = max(2, editor_lines + 1)
    conversation = max(1, height - chrome - editor_height)
    return conversation, editor_height


def visible_transcript(lines: list[str], height: int) -> list[str]:
    if len(lines) <= height:
        return lines
    return lines[-height:]


@dataclass(frozen=True)
class TodoItem:
    id: int
    text: str
    done: bool = False


@dataclass
class TodoList:
    """Small UI-local task list; it is never part of an agent request."""

    max_items: int = 32
    max_text_chars: int = 240
    _items: list[TodoItem] = field(default_factory=list)
    _next_id: int = 1

    @property
    def items(self) -> tuple[TodoItem, ...]:
        return tuple(self._items)

    def add(self, text: str) -> str:
        text = " ".join(text.split())
        if not text:
            return "TODO error: provide text after /todo add"
        if len(text) > self.max_text_chars:
            return f"TODO error: item is limited to {self.max_text_chars} characters"
        if len(self._items) >= self.max_items:
            return f"TODO error: list is limited to {self.max_items} items"
        item = TodoItem(self._next_id, text)
        self._next_id += 1
        self._items.append(item)
        return f"TODO added #{item.id}: {item.text}"

    def list_text(self) -> str:
        if not self._items:
            return "TODO\n(no items)"
        rows = ["TODO"]
        rows.extend(f"{'x' if item.done else ' '} #{item.id} {item.text}" for item in self._items)
        return "\n".join(rows)

    def done(self, item_id: str) -> str:
        return self._replace(item_id, done=True, action="completed")

    def remove(self, item_id: str) -> str:
        try:
            value = int(item_id)
        except ValueError:
            return "TODO error: id must be a number"
        for index, item in enumerate(self._items):
            if item.id == value:
                self._items.pop(index)
                return f"TODO removed #{value}"
        return f"TODO error: no item #{value}"

    def clear(self) -> str:
        count = len(self._items)
        self._items.clear()
        return f"TODO cleared ({count} item{'s' if count != 1 else ''})"

    def _replace(self, item_id: str, *, done: bool, action: str) -> str:
        try:
            value = int(item_id)
        except ValueError:
            return "TODO error: id must be a number"
        for index, item in enumerate(self._items):
            if item.id == value:
                self._items[index] = TodoItem(item.id, item.text, done)
                return f"TODO {action} #{value}: {item.text}"
        return f"TODO error: no item #{value}"


def format_todo_view(todos: TodoList, *, max_items: int = 4, max_text_chars: int = 72) -> str:
    """Render a compact, bounded snapshot of UI-local TODO state."""
    max_items = max(1, max_items)
    max_text_chars = max(1, max_text_chars)
    items = todos.items
    if not items:
        return "TODO (0)\n(no items)"
    lines = [f"TODO ({len(items)})"]
    for item in items[:max_items]:
        text = item.text
        if len(text) > max_text_chars:
            text = text[: max_text_chars - 1] + "…" if max_text_chars > 1 else "…"
        lines.append(f"{'x' if item.done else ' '} #{item.id} {text}")
    omitted = len(items) - max_items
    if omitted > 0:
        lines.append(f"… {omitted} more item{'s' if omitted != 1 else ''}")
    return "\n".join(lines)


@dataclass(frozen=True)
class LocalControl:
    kind: str
    category: str | None = None
    policy: str | None = None
    mode: str | None = None
    model_id: str | None = None


@dataclass(frozen=True)
class LocalCommand:
    handled: bool
    text: str = ""
    theme: str | None = None
    show_theme_selector: bool = False
    control: LocalControl | None = None


THEME_NAMES = ("dark", "light", "high-contrast", "terminal")


def next_theme(name: str) -> str:
    try:
        return THEME_NAMES[(THEME_NAMES.index(name) + 1) % len(THEME_NAMES)]
    except ValueError:
        return THEME_NAMES[0]


def _model_label(model: object) -> tuple[str, str]:
    model_id = str(getattr(model, "id", None) or "unknown")
    name = str(getattr(model, "display_name", None) or model_id)
    return name, model_id


def _model_details(model: object, *, indent: str = "") -> list[str]:
    endpoint_profile = getattr(model, "endpoint_profile", None)
    context_tokens = getattr(model, "context_tokens", None)
    modes = tuple(str(mode) for mode in getattr(model, "supported_reasoning_modes", ()) if str(mode))
    lines = [f"{indent}Endpoint: {endpoint_profile or 'not declared'}"]
    if isinstance(context_tokens, int):
        lines.append(f"{indent}Context: {context_tokens:,} tokens")
    lines.append(f"{indent}Reasoning support: {', '.join(modes) if modes else 'not declared'}")
    return lines


def format_model_status(
    model: object, current_reasoning_mode: str | None, models: tuple[object, ...] = ()
) -> str:
    active_name, active_id = _model_label(model)
    catalog = models or (model,)
    lines = ["MODELS", f"* Active: {active_name} ({active_id})"]
    lines.extend(_model_details(model))
    lines.append(f"Current reasoning: {current_reasoning_mode or 'configured default'}")
    for candidate in catalog:
        name, model_id = _model_label(candidate)
        if model_id == active_id:
            continue
        lines.append(f"- {name} ({model_id})")
        lines.extend(_model_details(candidate, indent="  "))
    return "\n".join(lines)


def format_model_changed(
    previous_model_id: str, model: object, *, new_session_started: bool, reasoning_reset: bool
) -> str:
    name, model_id = _model_label(model)
    lines = ["MODEL", f"✓ Switched from {previous_model_id} to {name} ({model_id})."]
    if new_session_started:
        lines.append("✓ New conversation started.")
        lines.append("Previous conversation history was not reused.")
    if reasoning_reset:
        lines.append("✓ Reasoning reset to configured default.")
    return "\n".join(lines)


def format_reasoning_mode_status(mode: str | None, available_modes: tuple[str, ...], none_semantics: str) -> str:
    del none_semantics
    current = mode or "configured default"
    available = ", ".join(available_modes) if available_modes else "not declared"
    return f"REASONING MODE\nCurrent: {current}\nSupported: {available}"


def format_reasoning_mode_changed(mode: str | None) -> str:
    return f"REASONING\n✓ Mode set to {mode or 'none'}."


def handle_local_command(prompt: str, todos: TodoList) -> LocalCommand:
    """Handle local commands, leaving ordinary prompts untouched."""
    stripped = prompt.strip()
    if not stripped.startswith("/"):
        return LocalCommand(False)
    command, _, argument = stripped.partition(" ")
    spec = known_command(command)
    if spec is None:
        return LocalCommand(True, f"Unknown command: {command}")
    normalized_command = spec.name
    argument = argument.strip()
    if spec.exits:
        return LocalCommand(True)
    if normalized_command == "/help":
        if argument:
            return LocalCommand(True, "Usage: /help")
        return LocalCommand(True, format_help())
    if normalized_command == "/models":
        values = argument.split()
        if not values:
            return LocalCommand(True, control=LocalControl("model_status"))
        if values[0].lower() == "use" and len(values) == 2:
            return LocalCommand(True, control=LocalControl("select_model", model_id=values[1].lower()))
        return LocalCommand(True, "Usage: /models [use <model>]")
    if normalized_command == "/mode":
        values = argument.split()
        if not values:
            return LocalCommand(True, control=LocalControl("reasoning_mode_status"))
        if len(values) == 1:
            return LocalCommand(True, control=LocalControl("set_reasoning_mode", mode=values[0].lower()))
        return LocalCommand(True, "Usage: /mode [mode]")
    if normalized_command == "/permissions":
        values = argument.lower().split()
        if not values:
            return LocalCommand(True, control=LocalControl("permissions_status"))
        if len(values) != 2:
            return LocalCommand(True, "Usage: /permissions [read|write|execute] [allow|deny|ask]")
        category, value = values
        if category not in PERMISSION_CATEGORIES:
            return LocalCommand(True, f"Unknown permission: {category}")
        if value not in PERMISSION_POLICIES:
            return LocalCommand(True, "Expected allow, deny, or ask.")
        return LocalCommand(True, control=LocalControl("set_permission", category=category, policy=value))
    if normalized_command == "/clear":
        if argument:
            return LocalCommand(True, "Usage: /clear")
        return LocalCommand(True, control=LocalControl("clear_session"))
    if normalized_command == "/todo":
        if not argument or argument == "list":
            return LocalCommand(True, todos.list_text())
        verb, _, value = argument.partition(" ")
        value = value.strip()
        if verb == "add":
            return LocalCommand(True, todos.add(value))
        if verb == "done":
            return LocalCommand(True, todos.done(value))
        if verb == "remove":
            return LocalCommand(True, todos.remove(value))
        if verb == "clear" and not value:
            return LocalCommand(True, todos.clear())
        return LocalCommand(True, "TODO error: use /todo [list|add|done|remove|clear]")
    if normalized_command == "/theme":
        if not argument:
            return LocalCommand(True, "Theme: choose dark, light, high-contrast, or terminal", show_theme_selector=True)
        if argument in THEME_NAMES:
            return LocalCommand(True, f"Theme selected: {argument}", theme=argument)
        return LocalCommand(True, f"Theme error: choose one of {', '.join(THEME_NAMES)}")
    return LocalCommand(True, f"Unknown command: {command}")


def display_bounded(text: str, limit: int) -> str:
    """Keep a UI snapshot bounded while explaining omitted user-visible data."""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n… display truncated ({len(text) - limit} characters omitted)"


def copy_osc52(
    text: str,
    writer: Callable[[bytes], object] | None = None,
    *,
    max_chars: int | None = 4_000,
) -> tuple[bool, str]:
    """Copy bounded plain text through OSC 52 when the terminal permits it."""
    if writer is None:
        if not sys.stdout.isatty() or not os.environ.get("TERM"):
            return False, "Clipboard unavailable in this terminal"
        writer = lambda data: os.write(sys.stdout.fileno(), data)
    try:
        bounded = text if max_chars is None else display_bounded(text, max(1, max_chars))
        encoded = base64.b64encode(bounded.encode("utf-8")).decode("ascii")
        writer(f"\x1b]52;c;{encoded}\x07".encode("ascii"))
    except Exception:
        return False, "Clipboard unavailable"
    return True, "Copied"
