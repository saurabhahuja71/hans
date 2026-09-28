"""Presentation state for the HANS terminal.

This module does not talk to the model. It records the conversation and the
prompt editor so the screen can be drawn without mixing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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


@dataclass
class Piece:
    kind: str
    text: str
    detail: str = ""


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
    return text.strip().lower() in {"exit", "quit"}


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
