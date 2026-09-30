from __future__ import annotations

import asyncio
import base64
import difflib
import hashlib
import os
import shlex
import subprocess
import tempfile
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

from agents import ToolOutputImage, function_tool
from agents.tool_context import ToolContext

from bolt_next.context_budget import estimate_tokens, tool_result_token_budget


class WorkspaceError(ValueError):
    """An attempted workspace access was invalid."""


@dataclass(frozen=True)
class _OriginalFileSnapshot:
    existed: bool
    content: bytes | None
    mode: int | None
    had_git_worktree_change: bool | None


@dataclass(frozen=True)
class _ExpectedFileState:
    digest: str
    mode: int
    content: bytes


@dataclass
class _JournalEntry:
    original: _OriginalFileSnapshot
    expected: _ExpectedFileState


class TaskMutationJournal:
    """Tracks filesystem mutations made while handling one task."""

    def __init__(self, workspace: Path, *, max_diff_chars: int = 12_000) -> None:
        self.workspace = resolve_workspace(workspace)
        self.max_diff_chars = max_diff_chars
        self._entries: dict[str, _JournalEntry] = {}

    def begin_task(self) -> None:
        self.reset_task()

    def reset_task(self) -> None:
        self._entries.clear()

    def prepare_mutation(self, path: Path) -> _OriginalFileSnapshot:
        target = path.resolve()
        relative = target.relative_to(self.workspace).as_posix()
        existing = self._entries.get(relative)
        if existing is not None:
            return existing.original
        if target.exists():
            if not target.is_file():
                raise WorkspaceError(f"Path is not a file: {relative}")
            stat = target.stat()
            return _OriginalFileSnapshot(
                existed=True,
                content=target.read_bytes(),
                mode=stat.st_mode & 0o7777,
                had_git_worktree_change=self._git_path_is_changed(relative),
            )
        return _OriginalFileSnapshot(
            existed=False,
            content=None,
            mode=None,
            had_git_worktree_change=False,
        )

    def record_successful_mutation(self, path: Path, original: _OriginalFileSnapshot) -> None:
        target = path.resolve()
        relative = target.relative_to(self.workspace).as_posix()
        if not target.is_file():
            raise WorkspaceError(f"Path is not a file: {relative}")
        stat = target.stat()
        content = target.read_bytes()
        expected = _ExpectedFileState(
            digest=hashlib.sha256(content).hexdigest(),
            mode=stat.st_mode & 0o7777,
            content=content,
        )
        self._entries.setdefault(relative, _JournalEntry(original=original, expected=expected)).expected = expected

    def summary(self) -> dict[str, object]:
        entries = list(self._entries.items())
        git_available = any(entry.original.had_git_worktree_change is not None for _, entry in entries)
        return {
            "changed_files": [path for path, _ in entries],
            "created_files": [path for path, entry in entries if not entry.original.existed],
            "preexisting_git_worktree_changes": [
                path for path, entry in entries if entry.original.had_git_worktree_change
            ],
            "git_available": git_available,
        }

    def compact_summary(self) -> str:
        summary = self.summary()
        changed = ", ".join(summary["changed_files"]) or "none"
        created = ", ".join(summary["created_files"]) or "none"
        git_changed = ", ".join(summary["preexisting_git_worktree_changes"]) or "none"
        git_state = "available" if summary["git_available"] else "unavailable"
        return (
            f"changed_files: {changed}\ncreated_files: {created}\n"
            f"preexisting_git_worktree_changes: {git_changed}\ngit: {git_state}"
        )

    def unified_diff(self, *, max_chars: int | None = None) -> str:
        limit = self.max_diff_chars if max_chars is None else max_chars
        parts: list[str] = []
        for relative, entry in self._entries.items():
            current = entry.expected.content
            before = entry.original.content if entry.original.existed else b""
            try:
                before_text = before.decode("utf-8").splitlines(keepends=True)
                current_text = current.decode("utf-8").splitlines(keepends=True)
            except UnicodeDecodeError:
                if before != current:
                    parts.append(f"Binary files differ: {relative}")
                continue
            parts.extend(
                difflib.unified_diff(
                    before_text,
                    current_text,
                    fromfile=f"a/{relative}",
                    tofile=f"b/{relative}",
                )
            )
        result = "".join(parts)
        if len(result) <= limit:
            return result
        notice = "\n... HANS-only diff truncated.\n"
        return result[: max(0, limit - len(notice))] + notice

    def undo(self) -> dict[str, list[str]]:
        conflicts: list[str] = []
        for relative, entry in self._entries.items():
            target = self.workspace / relative
            if not self._matches_expected(target, entry.expected):
                conflicts.append(relative)
        if conflicts:
            return {"restored": [], "removed": [], "conflicts": conflicts}

        restored: list[str] = []
        removed: list[str] = []
        for relative, entry in self._entries.items():
            target = self.workspace / relative
            if entry.original.existed:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(entry.original.content or b"")
                os.chmod(target, entry.original.mode or 0)
                restored.append(relative)
            else:
                target.unlink()
                removed.append(relative)
        return {"restored": restored, "removed": removed, "conflicts": []}

    def _matches_expected(self, target: Path, expected: _ExpectedFileState) -> bool:
        try:
            target.resolve().relative_to(self.workspace)
        except (OSError, ValueError):
            return False
        if target.is_symlink() or not target.is_file():
            return False
        stat = target.stat()
        return (
            hashlib.sha256(target.read_bytes()).hexdigest() == expected.digest
            and (stat.st_mode & 0o7777) == expected.mode
        )

    def _git_path_is_changed(self, relative: str) -> bool | None:
        try:
            inside = subprocess.run(
                ["git", "-C", str(self.workspace), "rev-parse", "--is-inside-work-tree"],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if inside.returncode != 0 or inside.stdout.strip() != "true":
                return None
            status = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.workspace),
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=all",
                    "--",
                    relative,
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            if status.returncode != 0 or len(status.stdout) > 8_192:
                return None
            return bool(status.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            return None


def _prepare_journal_mutation(journal: object | None, target: Path) -> object | None:
    if journal is None:
        return None
    prepare = getattr(journal, "prepare_mutation", None)
    return prepare(target) if callable(prepare) else None


def _record_journal_mutation(journal: object | None, target: Path, original: object | None) -> None:
    if journal is None:
        return
    record = getattr(journal, "record_successful_mutation", None)
    if callable(record):
        record(target, original)


def resolve_workspace(path: str | Path | None = None) -> Path:
    workspace = Path(path or Path.cwd()).expanduser().resolve()
    if not workspace.is_dir():
        raise WorkspaceError(f"Workspace is not a directory: {workspace}")
    return workspace


def resolve_workspace_path(workspace: Path, path: str) -> Path:
    if not path or "\x00" in path:
        raise WorkspaceError("Path must be a non-empty relative path")
    candidate = (workspace / path).resolve()
    try:
        candidate.relative_to(workspace)
    except ValueError as exc:
        raise WorkspaceError("Path is outside the workspace") from exc
    return candidate


@dataclass(frozen=True)
class ExternalPathAccess:
    tool_name: str
    call_id: str
    path: Path
    display_path: str


class ExternalPathAuthorizer:
    """Holds exact, one-use approvals for filesystem targets outside the workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = resolve_workspace(workspace)
        self._proposals: dict[tuple[str, str], ExternalPathAccess] = {}
        self._grants: dict[tuple[str, str], ExternalPathAccess] = {}

    def classify(self, path: str, *, mutation: bool = False) -> tuple[Path, bool]:
        if not isinstance(path, str) or not path or "\x00" in path:
            raise WorkspaceError("Path must be a non-empty relative path")
        candidate = Path(path).expanduser()
        target = (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (self.workspace / candidate).resolve(strict=False)
        )
        try:
            target.relative_to(self.workspace)
        except ValueError:
            return target, True
        return target, False

    def propose(self, tool_name: str, call_id: str, path: str, *, mutation: bool = False) -> ExternalPathAccess | None:
        target, external = self.classify(path, mutation=mutation)
        if not external:
            return None
        access = ExternalPathAccess(tool_name, call_id, target, path)
        self._proposals[(tool_name, call_id)] = access
        return access

    def proposal_for(self, tool_name: str, call_id: str) -> ExternalPathAccess | None:
        return self._proposals.get((tool_name, call_id))

    def approve_exact(self, tool_name: str, call_id: str) -> ExternalPathAccess | None:
        key = (tool_name, call_id)
        access = self._proposals.pop(key, None)
        if access is not None:
            self._grants[key] = access
        return access

    def revoke(self, tool_name: str, call_id: str) -> None:
        key = (tool_name, call_id)
        self._proposals.pop(key, None)
        self._grants.pop(key, None)

    def clear(self) -> None:
        self._proposals.clear()
        self._grants.clear()

    def consume_target(
        self,
        context: ToolContext,
        tool_name: str,
        path: str,
        *,
        mutation: bool = False,
    ) -> tuple[Path, bool]:
        call_id = getattr(context, "tool_call_id", None)
        if not isinstance(call_id, str) or not call_id:
            raise WorkspaceError("External path requires approval")
        target, external = self.classify(path, mutation=mutation)
        key = (tool_name, call_id)
        granted = self._grants.get(key)
        if not external:
            if granted is not None:
                self._grants.pop(key, None)
                raise WorkspaceError("External path requires approval")
            return target, False
        if granted is None or granted.path != target:
            self._grants.pop(key, None)
            raise WorkspaceError("External path requires approval")
        self._grants.pop(key, None)
        return target, True


def _tool_target(
    workspace: Path,
    authorizer: ExternalPathAuthorizer | None,
    context: ToolContext,
    tool_name: str,
    path: str,
    *,
    mutation: bool = False,
) -> tuple[Path, bool]:
    if authorizer is None:
        target = resolve_workspace_path(workspace, path)
        return target, False
    return authorizer.consume_target(context, tool_name, path, mutation=mutation)


def _revalidate_external_write_target(target: Path) -> None:
    if target.resolve(strict=False) != target or target.parent.resolve(strict=False) != target.parent:
        raise WorkspaceError("External path changed before it could be written")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.resolve(strict=False) != target or target.parent.resolve(strict=False) != target.parent:
        raise WorkspaceError("External path changed before it could be written")


def _range_result(path: str, lines: list[str], start_line: int, end_line: int) -> str:
    total = len(lines)
    body = "\n".join(lines[start_line - 1 : end_line])
    shown_end = start_line + body.count("\n") if body else start_line - 1
    if body:
        shown_end = start_line + body.count("\n")
    header = (
        f"path: {path}\n"
        f"total_lines: {total}\n"
        f"returned_range: {start_line}-{shown_end}\n"
    )
    if shown_end < total:
        header += (
            f"remaining_ranges: {shown_end + 1}-{total}\n"
            f"request_next: read_file path={path} start_line={shown_end + 1}\n"
        )
    else:
        header += "remaining_ranges: none\n"
    header += (
        "note: the lines below are the authoritative file text for returned_range. "
        "Lines outside that range were not included.\n"
        "---\n"
    )
    return header + body


def _fitting_end(
    path: str,
    lines: list[str],
    start_line: int,
    end_line: int,
    *,
    context_tokens: int | None = None,
) -> int | None:
    budget = tool_result_token_budget(context_tokens)
    if estimate_tokens(_range_result(path, lines, start_line, end_line)) <= budget:
        return end_line
    low = start_line
    high = end_line
    best = None
    while low <= high:
        mid = (low + high) // 2
        if estimate_tokens(_range_result(path, lines, start_line, mid)) <= budget:
            best = mid
            low = mid + 1
        else:
            high = mid - 1
    return best


_IMAGE_TYPES = {
    ".png": ("image/png", b"\x89PNG\r\n\x1a\n"),
    ".jpg": ("image/jpeg", b"\xff\xd8\xff"),
    ".jpeg": ("image/jpeg", b"\xff\xd8\xff"),
    ".webp": ("image/webp", b"RIFF"),
    ".gif": ("image/gif", b"GIF"),
}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_IMAGE_DIMENSION = 8_192
_MAX_IMAGE_PIXELS = 32_000_000


def _image_type(path: Path, content: bytes) -> str | None:
    configured = _IMAGE_TYPES.get(path.suffix.lower())
    if configured is None:
        return None
    mime_type, signature = configured
    if not content.startswith(signature):
        return None
    if mime_type == "image/webp" and content[8:12] != b"WEBP":
        return None
    if mime_type == "image/gif" and not content.startswith((b"GIF87a", b"GIF89a")):
        return None
    return mime_type


def _image_dimensions(mime_type: str, content: bytes) -> tuple[int, int] | None:
    if mime_type == "image/png":
        return _png_dimensions(content)
    if mime_type == "image/jpeg":
        return _jpeg_dimensions(content)
    if mime_type == "image/gif":
        return _gif_dimensions(content)
    return _webp_dimensions(content)


def _png_dimensions(content: bytes) -> tuple[int, int] | None:
    position = 8
    dimensions: tuple[int, int] | None = None
    saw_idat = False
    while position + 12 <= len(content):
        length = int.from_bytes(content[position : position + 4], "big")
        chunk_type = content[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if data_end > len(content) - 4 or not all(65 <= byte <= 90 or 97 <= byte <= 122 for byte in chunk_type):
            return None
        if zlib.crc32(content[position + 4 : data_end]) & 0xFFFFFFFF != int.from_bytes(content[data_end:crc_end], "big"):
            return None
        if dimensions is None:
            if chunk_type != b"IHDR" or length != 13:
                return None
            width = int.from_bytes(content[data_start : data_start + 4], "big")
            height = int.from_bytes(content[data_start + 4 : data_start + 8], "big")
            bit_depth, color_type, compression, filter_method, interlace = content[data_start + 8 : data_end]
            valid_depths = {0: {1, 2, 4, 8, 16}, 2: {8, 16}, 3: {1, 2, 4, 8}, 4: {8, 16}, 6: {8, 16}}
            if bit_depth not in valid_depths.get(color_type, set()) or compression or filter_method or interlace > 1:
                return None
            dimensions = width, height
        elif chunk_type == b"IHDR":
            return None
        if chunk_type == b"IDAT":
            saw_idat = True
        if chunk_type == b"IEND":
            return dimensions if length == 0 and saw_idat and crc_end == len(content) else None
        position = crc_end
    return None


def _jpeg_dimensions(content: bytes) -> tuple[int, int] | None:
    position = 2
    dimensions: tuple[int, int] | None = None
    saw_scan_data = False
    sof_markers = {*range(0xC0, 0xC4), *range(0xC5, 0xC8), *range(0xC9, 0xCC), *range(0xCD, 0xD0)}
    while position < len(content):
        if content[position] != 0xFF:
            return None
        while position < len(content) and content[position] == 0xFF:
            position += 1
        if position >= len(content):
            return None
        marker = content[position]
        position += 1
        if marker == 0xD9:
            return dimensions if dimensions is not None and saw_scan_data and position == len(content) else None
        if marker in {0x00, 0xD8} or 0xD0 <= marker <= 0xD7:
            return None
        if position + 2 > len(content):
            return None
        length = int.from_bytes(content[position : position + 2], "big")
        end = position + length
        if length < 2 or end > len(content):
            return None
        if marker in sof_markers:
            components = content[position + 7] if length >= 8 else 0
            if components == 0 or length != 8 + 3 * components:
                return None
            dimensions = (
                int.from_bytes(content[position + 5 : position + 7], "big"),
                int.from_bytes(content[position + 3 : position + 5], "big"),
            )
        if marker != 0xDA:
            position = end
            continue
        components = content[position + 2] if length >= 3 else 0
        if dimensions is None or components == 0 or length != 6 + 2 * components:
            return None
        position = end
        scan_has_entropy = False
        while position < len(content):
            if content[position] != 0xFF:
                scan_has_entropy = True
                position += 1
                continue
            marker_start = position
            while position < len(content) and content[position] == 0xFF:
                position += 1
            if position >= len(content):
                return None
            marker = content[position]
            position += 1
            if marker == 0x00:
                scan_has_entropy = True
                continue
            if 0xD0 <= marker <= 0xD7:
                continue
            if marker == 0xD9:
                return dimensions if scan_has_entropy and position == len(content) else None
            position = marker_start
            break
        else:
            return None
        saw_scan_data = saw_scan_data or scan_has_entropy
    return None


def _gif_sub_blocks(content: bytes, position: int) -> int | None:
    while position < len(content):
        length = content[position]
        position += 1
        if length == 0:
            return position
        if position + length > len(content):
            return None
        position += length
    return None


def _gif_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 13:
        return None
    width = int.from_bytes(content[6:8], "little")
    height = int.from_bytes(content[8:10], "little")
    position = 13
    if content[10] & 0x80:
        position += 3 * (1 << ((content[10] & 0x07) + 1))
    if position > len(content):
        return None
    saw_image = False
    while position < len(content):
        introducer = content[position]
        position += 1
        if introducer == 0x3B:
            return (width, height) if saw_image and position == len(content) else None
        if introducer == 0x21:
            if position >= len(content):
                return None
            label = content[position]
            position += 1
            if label == 0xF9:
                if position + 6 > len(content) or content[position] != 4 or content[position + 5] != 0:
                    return None
                position += 6
            else:
                if label == 0xFF and (position >= len(content) or content[position] != 11):
                    return None
                if label == 0x01 and (position >= len(content) or content[position] != 12):
                    return None
                position = _gif_sub_blocks(content, position)
                if position is None:
                    return None
            continue
        if introducer != 0x2C or position + 9 > len(content):
            return None
        packed = content[position + 8]
        position += 9
        if packed & 0x80:
            position += 3 * (1 << ((packed & 0x07) + 1))
        if position >= len(content) or not 2 <= content[position] <= 8:
            return None
        position = _gif_sub_blocks(content, position + 1)
        if position is None:
            return None
        saw_image = True
    return None


def _webp_dimensions(content: bytes) -> tuple[int, int] | None:
    if len(content) < 12 or int.from_bytes(content[4:8], "little") != len(content) - 8:
        return None
    position = 12
    vp8x_dimensions: tuple[int, int] | None = None
    image_dimensions: tuple[int, int] | None = None
    while position < len(content):
        if position + 8 > len(content):
            return None
        chunk_type = content[position : position + 4]
        length = int.from_bytes(content[position + 4 : position + 8], "little")
        data_start = position + 8
        data_end = data_start + length
        next_position = data_end + (length & 1)
        if next_position > len(content):
            return None
        data = content[data_start:data_end]
        if chunk_type == b"VP8X":
            if len(data) != 10 or data[1:4] != b"\x00\x00\x00":
                return None
            vp8x_dimensions = int.from_bytes(data[4:7], "little") + 1, int.from_bytes(data[7:10], "little") + 1
        elif chunk_type == b"VP8 ":
            first_partition_size = int.from_bytes(data[:3], "little") >> 5
            if (
                len(data) < 11
                or data[0] & 1
                or data[3:6] != b"\x9d\x01\x2a"
                or len(data) < 10 + first_partition_size
            ):
                return None
            image_dimensions = int.from_bytes(data[6:8], "little") & 0x3FFF, int.from_bytes(data[8:10], "little") & 0x3FFF
        elif chunk_type == b"VP8L":
            if len(data) < 6 or data[0] != 0x2F:
                return None
            packed = int.from_bytes(data[1:5], "little")
            image_dimensions = (packed & 0x3FFF) + 1, ((packed >> 14) & 0x3FFF) + 1
        position = next_position
    if position != len(content) or image_dimensions is None:
        return None
    return vp8x_dimensions or image_dimensions


def _image_error(path: str, target: Path, content: bytes, *, read_image_available: bool) -> str:
    if _image_type(target, content) is not None and read_image_available:
        return f"Error reading {path!r}: file is binary or an image; use read_image for PNG, JPEG, WebP, or GIF files"
    return f"Error reading {path!r}: file cannot be read as text; supported image formats are PNG, JPEG, WebP, and GIF"


def make_read_image_tool(workspace: Path, *, authorizer: ExternalPathAuthorizer | None = None):
    @function_tool
    async def read_image(context: ToolContext, path: str) -> ToolOutputImage | str:
        """Read a PNG, JPEG, WebP, or GIF image as model-visible image data.

        The path is inside the workspace unless this exact call is approved for an external path that resolves outside the workspace. The image is not OCR'd.

        Args:
            path: A workspace-relative image path, or an approved external image path.
        """
        try:
            target, _external = _tool_target(workspace, authorizer, context, "read_image", path)
            if target.is_dir():
                return f"Error reading image {path!r}: path is a directory; use list_directory instead"
            if not target.is_file():
                return f"Error: image file does not exist: {path}"
            if target.stat().st_size > _MAX_IMAGE_BYTES:
                return f"Error reading image {path!r}: image exceeds {_MAX_IMAGE_BYTES} bytes"
            content = target.read_bytes()
            if len(content) > _MAX_IMAGE_BYTES:
                return f"Error reading image {path!r}: image exceeds {_MAX_IMAGE_BYTES} bytes"
            mime_type = _image_type(target, content)
            if mime_type is None:
                return f"Error reading image {path!r}: only PNG, JPEG, WebP, and GIF files with matching file signatures are supported"
            dimensions = _image_dimensions(mime_type, content)
            if dimensions is None:
                return f"Error reading image {path!r}: invalid or truncated {mime_type.removeprefix('image/').upper()} image data"
            width, height = dimensions
            if not width or not height or width > _MAX_IMAGE_DIMENSION or height > _MAX_IMAGE_DIMENSION or width * height > _MAX_IMAGE_PIXELS:
                return f"Error reading image {path!r}: image dimensions exceed limits"
            encoded = base64.b64encode(content).decode("ascii")
            return ToolOutputImage(image_url=f"data:{mime_type};base64,{encoded}")
        except (OSError, WorkspaceError) as exc:
            return f"Error reading image {path!r}: {exc}"

    return read_image


def make_read_file_tool(
    workspace: Path,
    *,
    context_tokens: int | None = None,
    authorizer: ExternalPathAuthorizer | None = None,
    read_image_available: bool = False,
):
    @function_tool
    async def read_file(context: ToolContext, path: str, start_line: int = 1, end_line: int = 0) -> str:
        """Read a UTF-8 text file inside the workspace unless this exact call is approved for an external path that resolves outside the workspace.

        Small files are returned in full. A large file is returned as an explicit
        line range, never as a summary. Use start_line and end_line to inspect
        another range. end_line 0 means "as far as the context budget allows".

        Args:
            path: A workspace-relative path, or an external path that resolves outside the workspace and is approved for this exact call.
            start_line: First line to return, starting at 1.
            end_line: Last line to return, inclusive. 0 selects a budget-sized range.
        """
        try:
            target, _external = _tool_target(workspace, authorizer, context, "read_file", path)
            if target.is_dir():
                return f"Error reading {path!r}: path is a directory; use list_directory instead"
            if not target.is_file():
                return f"Error: file does not exist: {path}"
            content = target.read_bytes()
            try:
                text = content.decode("utf-8")
            except UnicodeError:
                return _image_error(path, target, content, read_image_available=read_image_available)
            if "\x00" in text:
                return _image_error(path, target, content, read_image_available=read_image_available)
            lines = text.splitlines()
            total = len(lines)
            if total == 0:
                return ""
            if start_line < 1 or start_line > total:
                return (
                    f"Error reading {path!r}: start_line {start_line} is outside 1-{total}. "
                    f"total_lines: {total}"
                )
            explicit = end_line > 0
            if not explicit:
                end_line = total
            end_line = min(end_line, total)
            if end_line < start_line:
                return f"Error reading {path!r}: end_line must be >= start_line"
            if (
                not explicit
                and start_line == 1
                and end_line == total
                and estimate_tokens(text) <= tool_result_token_budget(context_tokens)
            ):
                return text
            fitted = _fitting_end(path, lines, start_line, end_line, context_tokens=context_tokens)
            if fitted is None:
                return (
                    f"Error reading {path!r}: requested range {start_line}-{end_line} "
                    f"does not fit in the tool result budget of {tool_result_token_budget(context_tokens)} tokens. "
                    f"total_lines: {total}. Request a smaller end_line. "
                    "No partial source was returned."
                )
            if explicit and fitted < end_line:
                return (
                    f"Error reading {path!r}: requested range {start_line}-{end_line} "
                    f"does not fit in the tool result budget of {tool_result_token_budget(context_tokens)} tokens. "
                    f"total_lines: {total}. A range ending at {fitted} fits. "
                    "No partial source was returned."
                )
            if fitted == total and start_line == 1 and estimate_tokens(text) <= tool_result_token_budget(context_tokens):
                return text
            return _range_result(path, lines, start_line, fitted)
        except (OSError, WorkspaceError) as exc:
            return f"Error reading {path!r}: {exc}"

    return read_file


def make_write_file_tool(
    workspace: Path,
    journal: object | None = None,
    *,
    authorizer: ExternalPathAuthorizer | None = None,
):
    @function_tool
    async def write_file(context: ToolContext, path: str, content: str) -> str:
        """Create or replace a UTF-8 text file inside the workspace unless this exact call is approved for an external path that resolves outside the workspace.

        Args:
            path: A workspace-relative path, or an external path that resolves outside the workspace and is approved for this exact call.
            content: The full file contents to write.
        """
        try:
            target, external = _tool_target(workspace, authorizer, context, "write_file", path, mutation=True)
            original = None if external else _prepare_journal_mutation(journal, target)
            if external:
                _revalidate_external_write_target(target)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.parent.resolve().is_relative_to(workspace):
                    return f"Error writing {path!r}: Path is outside the workspace"
            target.write_text(content, encoding="utf-8")
            if not external:
                _record_journal_mutation(journal, target, original)
            return f"Wrote {path} ({len(content.encode('utf-8'))} bytes)"
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"Error writing {path!r}: {exc}"

    return write_file


def _bounded_lines(
    lines: list[str], *, max_results: int | None = None, context_tokens: int | None = None
) -> str:
    budget = tool_result_token_budget(context_tokens)
    limited = lines if max_results is None else lines[:max_results]
    if not limited:
        return ""
    shown: list[str] = []
    for index, line in enumerate(limited):
        candidate = "\n".join([*shown, line])
        remaining = len(limited) - index - 1
        if not remaining and estimate_tokens(candidate) <= budget:
            return candidate
        notice = f"... {remaining} result(s) omitted to fit the tool result budget."
        rendered = f"{candidate}\n{notice}" if candidate else notice
        if estimate_tokens(rendered) > budget:
            break
        shown.append(line)
    omitted = len(limited) - len(shown)
    notice = f"... {omitted} result(s) omitted to fit the tool result budget."
    result = "\n".join(shown)
    return f"{result}\n{notice}" if result else notice


_SEARCH_IGNORED_DIRECTORIES = {
    ".eggs",
    ".git",
    ".hans-tmp",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
    "venv",
}


def _search_match_line(path: Path, line_number: int, line: str, *, context_tokens: int | None = None) -> str:
    prefix = f"{path}:{line_number}: "
    suffix = " ... [matching line truncated]"
    maximum_line_length = max(0, tool_result_token_budget(context_tokens) * 3 - len(prefix) - len(suffix))
    if len(line) <= maximum_line_length:
        return prefix + line
    return prefix + line[:maximum_line_length] + suffix


def _search_candidates(root: Path, target: Path, *, skip_ignored_root: bool = False):
    if target.is_file():
        yield target
        return
    try:
        relative_target = target.relative_to(root)
    except ValueError:
        return
    if skip_ignored_root and any(part in _SEARCH_IGNORED_DIRECTORIES for part in relative_target.parts):
        return
    for candidate in sorted(target.iterdir(), key=lambda entry: entry.name):
        try:
            candidate.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        if candidate.is_symlink() and candidate.is_dir():
            continue
        if candidate.is_dir():
            if candidate.name not in _SEARCH_IGNORED_DIRECTORIES:
                yield from _search_candidates(root, candidate)
        elif candidate.is_file():
            yield candidate


def make_list_directory_tool(
    workspace: Path,
    *,
    context_tokens: int | None = None,
    authorizer: ExternalPathAuthorizer | None = None,
):
    @function_tool
    async def list_directory(context: ToolContext, path: str = ".") -> str:
        """List direct directory entries in sorted order inside the workspace unless this exact call is approved for an external path that resolves outside the workspace.

        Args:
            path: A workspace-relative path, or an external path that resolves outside the workspace and is approved for this exact call.
        """
        try:
            target, _external = _tool_target(workspace, authorizer, context, "list_directory", path)
            if target.is_file():
                return f"Error listing {path!r}: path is a file; use read_file instead"
            if not target.is_dir():
                return f"Error listing {path!r}: directory does not exist"
            entries = []
            for entry in sorted(target.iterdir(), key=lambda candidate: candidate.name):
                if entry.is_symlink():
                    kind = "symlink"
                elif entry.is_dir():
                    kind = "directory"
                elif entry.is_file():
                    kind = "file"
                else:
                    kind = "other"
                entries.append(f"{kind}: {entry.name}")
            return _bounded_lines(entries, context_tokens=context_tokens)
        except (OSError, WorkspaceError) as exc:
            return f"Error listing {path!r}: {exc}"

    return list_directory


def make_search_files_tool(
    workspace: Path,
    *,
    context_tokens: int | None = None,
    authorizer: ExternalPathAuthorizer | None = None,
):
    @function_tool
    async def search_files(context: ToolContext, query: str, path: str = ".", max_results: int = 50) -> str:
        """Search UTF-8 text files and return matching path:line text.

        The path is inside the workspace unless this exact call is approved for an external path that resolves outside the workspace.
        Binary and unreadable files are skipped. Results are sorted and constrained by both
        max_results and the tool result context budget.

        Args:
            query: Literal text to find. It must not be empty.
            path: A workspace-relative path, or an external path that resolves outside the workspace and is approved for this exact call.
            max_results: Maximum matching lines to return, from 1 through 100.
        """
        if not query:
            return "Error searching: query must be a non-empty string"
        if max_results < 1 or max_results > 100:
            return "Error searching: max_results must be between 1 and 100"
        try:
            target, external = _tool_target(workspace, authorizer, context, "search_files", path)
            if not target.exists():
                return f"Error searching {path!r}: path does not exist"
            search_root = target if external else workspace
            display_root = target if external and target.is_dir() else target.parent if external else workspace
            matches: list[str] = []
            for candidate in _search_candidates(search_root, target, skip_ignored_root=not external):
                try:
                    relative = candidate.relative_to(display_root)
                    file_matches: list[str] = []
                    binary = False
                    with candidate.open(encoding="utf-8") as source:
                        for line_number, line in enumerate(source, start=1):
                            if "\x00" in line:
                                binary = True
                                break
                            line = line.rstrip("\r\n")
                            if query in line and len(matches) + len(file_matches) < max_results:
                                file_matches.append(
                                    _search_match_line(relative, line_number, line, context_tokens=context_tokens)
                                )
                    if binary:
                        continue
                    matches.extend(file_matches)
                except (OSError, UnicodeError, ValueError):
                    continue
                if len(matches) >= max_results:
                    return _bounded_lines(matches, max_results=max_results, context_tokens=context_tokens)
            return _bounded_lines(matches, max_results=max_results, context_tokens=context_tokens)
        except (OSError, WorkspaceError) as exc:
            return f"Error searching {path!r}: {exc}"

    return search_files


def make_replace_in_file_tool(
    workspace: Path,
    journal: object | None = None,
    *,
    authorizer: ExternalPathAuthorizer | None = None,
):
    @function_tool
    async def replace_in_file(context: ToolContext, path: str, old_text: str, new_text: str) -> str:
        """Replace exactly one literal text occurrence in an existing UTF-8 file inside the workspace unless this exact call is approved for an external path that resolves outside the workspace.

        Args:
            path: A workspace-relative path, or an external path that resolves outside the workspace and is approved for this exact call.
            old_text: Existing text that must occur exactly once.
            new_text: Replacement text.
        """
        if not old_text:
            return "Error replacing: old_text must be a non-empty string"
        try:
            target, external = _tool_target(workspace, authorizer, context, "replace_in_file", path)
            if not target.is_file():
                return f"Error replacing {path!r}: file does not exist"
            if external:
                _revalidate_external_write_target(target)
            original = None if external else _prepare_journal_mutation(journal, target)
            source_stat = target.stat()
            text = target.read_text(encoding="utf-8")
            occurrences = text.count(old_text)
            if occurrences != 1:
                return f"Error replacing {path!r}: old_text must occur exactly once (found {occurrences})"
            replacement = text.replace(old_text, new_text, 1)
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
                ) as temporary:
                    temporary.write(replacement)
                    temporary_path = Path(temporary.name)
                os.chmod(temporary_path, source_stat.st_mode)
                os.utime(
                    temporary_path,
                    ns=(
                        temporary_path.stat().st_atime_ns,
                        max(time.time_ns(), source_stat.st_mtime_ns + 1_000_000_000),
                    ),
                )
                os.replace(temporary_path, target)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            if not external:
                _record_journal_mutation(journal, target, original)
            return f"Replaced text in {path}"
        except (OSError, UnicodeError, WorkspaceError) as exc:
            return f"Error replacing {path!r}: {exc}"

    return replace_in_file


# Characters that only have meaning in a shell. run_command never invokes one.
_SHELL_META = set("|;&<>$`(){}[]*?~!#\\")
_SHELL_PROGRAMS = {"sh", "bash", "dash", "zsh", "fish", "ksh", "csh", "tcsh"}
# Developer tools still need a few variables. Secrets and the rest of the process
# environment are not copied into the command.
_COMMAND_ENV_KEYS = (
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TERM",
    "TZ",
    "GO111MODULE",
    "GOROOT",
    "GOPATH",
    "GOPROXY",
    "GOSUMDB",
    "GOPRIVATE",
    "GOCACHE",
    "GOMODCACHE",
    "GOTOOLCHAIN",
    "GOFLAGS",
    "CGO_ENABLED",
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
)


def command_environment(workspace: Path) -> dict[str, str]:
    """Environment passed to run_command. HOME and TMPDIR stay inside the workspace."""
    env = {key: os.environ[key] for key in _COMMAND_ENV_KEYS if os.environ.get(key)}
    tmp = workspace / ".hans-tmp"
    tmp.mkdir(exist_ok=True)
    env["HOME"] = str(workspace)
    env["TMPDIR"] = str(tmp)
    env["PWD"] = str(workspace)
    return env


def reject_shell_syntax(command: str) -> str | None:
    for argument in command.split():
        if "~" in argument:
            return (
                "Error: run_command does not expand `~` and only runs direct workspace commands. "
                "Use list_directory, search_files, or read_file with the literal `~/...` path; "
                "HANS will request explicit approval for external access."
            )
        if argument == "./...":
            continue
        if any(char in _SHELL_META for char in argument):
            return (
                "Error: run_command executes a direct argv command, not a shell. "
                "Pipes, redirects, &&, ||, globs, and other shell syntax are not supported."
            )
    if "\n" in command or "\r" in command:
        return (
            "Error: run_command executes a direct argv command, not a shell. "
            "Pipes, redirects, &&, ||, globs, and other shell syntax are not supported."
        )
    return None


def reject_workspace_escape(workspace: Path, args: list[str]) -> str | None:
    program = Path(args[0]).name
    if program in _SHELL_PROGRAMS:
        return "Error: run_command does not run a shell. Pass the program and its arguments directly."
    for arg in args:
        if arg.startswith("-"):
            continue
        path = Path(arg)
        if ".." in path.parts:
            return f"Error: command argument escapes the workspace: {arg}"
        if path.is_absolute():
            try:
                path.resolve().relative_to(workspace)
            except ValueError:
                return f"Error: command argument is outside the workspace: {arg}"
    return None


def make_run_command_tool(workspace: Path, *, context_tokens: int | None = None):
    @function_tool
    async def run_command(command: str, purpose: str = "inspect") -> str:
        """Run one direct command in the workspace and return its exit code and output.

        The command is split into argv and executed without a shell. Pipes, redirects,
        &&, ||, globs, substitution, and `~` expansion are rejected. The working
        directory is the workspace. Use list_directory, search_files, or read_file for
        filesystem discovery or inspection, including external paths. The command does
        not receive API keys or the rest of the process environment. purpose must be
        `inspect` for investigation or `verify` for a command intended to validate
        requested behavior.

        Args:
            command: Program and arguments, for example `go test ./...`.
            purpose: `inspect` or `verify`; defaults to `inspect`.
        """
        if purpose not in {"inspect", "verify"}:
            return "Error: purpose must be either 'inspect' or 'verify'"
        if not command or not command.strip() or "\x00" in command:
            return "Error: command must be a non-empty string"
        shell_error = reject_shell_syntax(command)
        if shell_error:
            return shell_error
        try:
            args = shlex.split(command)
        except ValueError as exc:
            return f"Error: could not parse command: {exc}"
        if not args:
            return "Error: command must be a non-empty string"
        escape_error = reject_workspace_escape(workspace, args)
        if escape_error:
            return escape_error

        def execute() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                args,
                cwd=workspace,
                env=command_environment(workspace),
                capture_output=True,
                text=True,
                timeout=120,
            )

        try:
            completed = await asyncio.to_thread(execute)
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 120 seconds"
        except FileNotFoundError:
            return f"Error: command not found: {args[0]}"
        except OSError as exc:
            return f"Error running command: {exc}"
        result = (
            f"exit_code={completed.returncode}\n"
            f"stdout:\n{completed.stdout}"
            f"stderr:\n{completed.stderr}"
        )
        budget = tool_result_token_budget(context_tokens)
        if estimate_tokens(result) <= budget:
            return result
        notice = (
            "\n---\n"
            "command output truncated to fit the context budget. "
            f"original_tokens≈{estimate_tokens(result)} budget_tokens={budget}. "
            "This is not a summary. Re-run a narrower command for the omitted output.\n"
        )
        return result[: max(0, budget * 3 - len(notice))] + notice

    return run_command
