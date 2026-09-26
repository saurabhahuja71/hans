"""Presentation state for the HANS terminal.

This module does not talk to the model. It records the conversation and the
prompt editor so the screen can be drawn without mixing them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bolt_next.events import VerificationEvidence


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
        self.stage("thinking…")

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
        self.pieces.append(Piece("status", "completed"))

    def completed(self, evidence: VerificationEvidence | None) -> None:
        self._drop_status()
        if evidence is None:
            self.pieces.append(Piece("status", "completed (verification not established)"))
        elif evidence.success:
            self.pieces.append(Piece("status", "completed"))
        else:
            self.pieces.append(Piece("status", "completed (verification failed)"))

    def cancelled(self) -> None:
        self._drop_status()
        self.pieces.append(Piece("error", "cancelled"))

    def tool_started(self, label: str) -> None:
        self._drop_status()
        self.pieces.append(Piece("tool", label, "run"))

    def tool_finished(self, label: str, *, ok: bool) -> None:
        self.pieces.append(Piece("tool", label, "ok" if ok else "fail"))

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
        return [("> " if index == 0 else "  ") + row for index, row in enumerate(rows)]
    if piece.kind == "assistant":
        return _wrap(piece.text, width)
    if piece.kind == "tool":
        mark = {"run": "◇", "ok": "✓", "fail": "✗"}.get(piece.detail, "◇")
        return _wrap(f"{mark} {piece.text}", width, "  ")
    if piece.kind == "status":
        mark = "✓" if piece.text.startswith("completed") else "◌"
        return [f"  {mark} {piece.text}"]
    if piece.kind == "error":
        lines = [f"  ✗ {piece.text}"]
        if piece.detail:
            lines.extend(_wrap(piece.detail, width, "    "))
        return lines
    if piece.kind == "debug":
        return _wrap(piece.text, width, "  ")
    return _wrap(piece.text, width)


class Editor:
    """Prompt editor. Enter and Ctrl-D submit. Ctrl-Q exits."""

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
        if key == "backspace":
            if self.lines[-1]:
                self.lines[-1] = self.lines[-1][:-1]
            elif len(self.lines) > 1:
                self.lines.pop()
            return None
        if key.startswith("char:"):
            self.lines[-1] += key[5:]
        return None

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
