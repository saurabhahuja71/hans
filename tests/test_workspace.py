import asyncio
import hashlib
import inspect
import json
import subprocess
import zlib
from pathlib import Path

import pytest
from agents.items import ItemHelpers
from agents.tool_context import ToolContext

import bolt_next.workspace as workspace_module
from bolt_next.context_budget import estimate_tokens, tool_result_token_budget
from bolt_next.workspace import (
    ExternalPathAuthorizer,
    TaskMutationJournal,
    WorkspaceError,
    _bounded_lines,
    make_list_directory_tool,
    make_read_file_tool,
    make_read_image_tool,
    make_replace_in_file_tool,
    make_run_command_tool,
    make_search_files_tool,
    make_write_file_tool,
    reject_shell_syntax,
    resolve_workspace_path,
)


def invoke(tool, arguments: str, *, call_id: str = "test-call"):
    # The SDK invokes this wrapped function after parsing the structured JSON arguments.
    kwargs = json.loads(arguments)
    if "context" in inspect.signature(tool.__wrapped__).parameters:
        kwargs["context"] = ToolContext(
            None,
            tool_name=tool.name,
            tool_call_id=call_id,
            tool_arguments=arguments,
        )
    return asyncio.run(tool.__wrapped__(**kwargs))


def test_workspace_path_validation(tmp_path: Path) -> None:
    assert resolve_workspace_path(tmp_path, "main.go") == tmp_path / "main.go"


def test_read_file_success(tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    tool = make_read_file_tool(tmp_path)
    assert invoke(tool, '{"path":"note.txt"}') == "hello"


def test_read_file_missing_file(tmp_path: Path) -> None:
    tool = make_read_file_tool(tmp_path)
    result = invoke(tool, '{"path":"missing.txt"}')
    assert "does not exist" in result


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return len(data).to_bytes(4, "big") + chunk_type + data + zlib.crc32(chunk_type + data).to_bytes(4, "big")


def valid_png(width: int = 2, height: int = 3) -> bytes:
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x02\x00\x00\x00"
    scanlines = b"".join(b"\x00" + b"\x00" * (width * 3) for _ in range(height))
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(scanlines)) + _png_chunk(b"IEND", b"")


def valid_jpeg() -> bytes:
    return b"\xff\xd8\xff\xc0\x00\x0b\x08\x00\x03\x00\x02\x01\x01\x11\x00\xff\xda\x00\x08\x01\x01\x00\x00\x3f\x00\x01\xff\xd9"


def valid_gif() -> bytes:
    return b"GIF89a\x02\x00\x03\x00\x80\x00\x00\x00\x00\x00\xff\xff\xff,\x00\x00\x00\x00\x02\x00\x03\x00\x00\x02\x02D\x01\x00;"


def _webp_container(*chunks: tuple[bytes, bytes]) -> bytes:
    payload = b"WEBP"
    for chunk_type, data in chunks:
        payload += chunk_type + len(data).to_bytes(4, "little") + data + (b"\x00" if len(data) & 1 else b"")
    return b"RIFF" + len(payload).to_bytes(4, "little") + payload


def valid_webp() -> bytes:
    return _webp_container((b"VP8 ", b"\x20\x00\x00\x9d\x01\x2a\x02\x00\x03\x00\x00"))


def valid_webp_lossless() -> bytes:
    packed_dimensions = (1 | (2 << 14)).to_bytes(4, "little")
    return _webp_container((b"VP8L", b"\x2f" + packed_dimensions + b"\x00"))


def valid_webp_extended() -> bytes:
    vp8x = b"\x00\x00\x00\x00\x01\x00\x00\x02\x00\x00"
    vp8 = b"\x20\x00\x00\x9d\x01\x2a\x02\x00\x03\x00\x00"
    return _webp_container((b"VP8X", vp8x), (b"VP8 ", vp8))


def test_read_image_returns_a_data_url_and_read_file_rejects_binary_without_mutation(tmp_path: Path) -> None:
    image = valid_png()
    target = tmp_path / "sample.png"
    target.write_bytes(image)
    before = hashlib.sha256(target.read_bytes()).hexdigest()

    result = invoke(make_read_image_tool(tmp_path), '{"path":"sample.png"}')

    assert result.type == "image"
    assert result.image_url is not None
    assert result.image_url.startswith("data:image/png;base64,")
    assert hashlib.sha256(target.read_bytes()).hexdigest() == before
    assert "binary or an image" in invoke(
        make_read_file_tool(tmp_path, read_image_available=True), '{"path":"sample.png"}'
    )


def test_read_file_hides_read_image_guidance_when_image_input_is_unavailable(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(valid_png())

    result = invoke(make_read_file_tool(tmp_path), '{"path":"sample.png"}')

    assert "cannot be read as text" in result
    assert "read_image" not in result


def test_read_image_uses_sdk_structured_image_output(tmp_path: Path) -> None:
    (tmp_path / "sample.png").write_bytes(valid_png())
    tool = make_read_image_tool(tmp_path)
    arguments = '{"path":"sample.png"}'
    context = ToolContext(None, tool_name=tool.name, tool_call_id="image-sdk", tool_arguments=arguments)

    result = asyncio.run(tool.on_invoke_tool(context, arguments))
    serialized = ItemHelpers._convert_tool_output(result)

    assert serialized == [
        {
            "type": "input_image",
            "image_url": result.image_url,
        }
    ]


@pytest.mark.parametrize(
    ("filename", "content", "mime_type"),
    [
        ("sample.gif", valid_gif(), "image/gif"),
        ("sample.jpg", valid_jpeg(), "image/jpeg"),
        ("sample.webp", valid_webp(), "image/webp"),
        ("sample-lossless.webp", valid_webp_lossless(), "image/webp"),
        ("sample-extended.webp", valid_webp_extended(), "image/webp"),
    ],
)
def test_read_image_supports_each_non_png_format(
    tmp_path: Path, filename: str, content: bytes, mime_type: str
) -> None:
    (tmp_path / filename).write_bytes(content)

    result = invoke(make_read_image_tool(tmp_path), json.dumps({"path": filename}))

    assert result.type == "image"
    assert result.image_url is not None
    assert result.image_url.startswith(f"data:{mime_type};base64,")


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("truncated.png", valid_png()[:-1]),
        ("truncated.jpg", valid_jpeg()[:-2]),
        ("truncated.gif", valid_gif()[:-1]),
        ("truncated.webp", valid_webp()[:-1]),
        ("corrupt.png", valid_png()[:-5] + b"\x00" + valid_png()[-4:]),
    ],
)
def test_read_image_rejects_truncated_or_corrupt_supported_images(tmp_path: Path, filename: str, content: bytes) -> None:
    (tmp_path / filename).write_bytes(content)

    result = invoke(make_read_image_tool(tmp_path), json.dumps({"path": filename}))

    assert "invalid or truncated" in result
    assert "image data" in result


def test_read_file_rejects_non_image_binary_without_read_image_guidance(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\xff\x00\x01")

    result = invoke(make_read_file_tool(tmp_path), '{"path":"binary.bin"}')

    assert "cannot be read as text" in result
    assert "PNG, JPEG, WebP, and GIF" in result
    assert "read_image" not in result


def test_read_image_rejects_mismatched_types_and_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "not-image.png").write_bytes(b"not a PNG")
    huge = valid_png(width=9_000, height=1)
    small = valid_png(width=1, height=1)
    (tmp_path / "huge.png").write_bytes(huge)
    (tmp_path / "too-large.png").write_bytes(small)
    tool = make_read_image_tool(tmp_path)

    assert "matching file signatures" in invoke(tool, '{"path":"not-image.png"}')
    assert "dimensions exceed limits" in invoke(tool, '{"path":"huge.png"}')
    monkeypatch.setattr(workspace_module, "_MAX_IMAGE_BYTES", len(small) - 1)
    assert "image exceeds" in invoke(tool, '{"path":"too-large.png"}')


def test_read_image_requires_its_own_exact_external_approval(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-image.png"
    outside.write_bytes(valid_png(width=1, height=1))
    authorizer = ExternalPathAuthorizer(tmp_path)
    tool = make_read_image_tool(tmp_path, authorizer=authorizer)
    arguments = json.dumps({"path": str(outside)})

    assert "External path requires approval" in invoke(tool, arguments, call_id="external-image")
    assert authorizer.propose("read_image", "external-image", str(outside)) is not None
    assert authorizer.approve_exact("read_image", "external-image") is not None
    assert invoke(tool, arguments, call_id="external-image").type == "image"


def test_external_path_tilde_uses_the_runtime_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    runtime_home = tmp_path / "runtime-home"
    target = runtime_home / "Desktop" / "chart.png"
    monkeypatch.setenv("HOME", str(runtime_home))
    authorizer = ExternalPathAuthorizer(workspace)

    access = authorizer.propose("read_image", "home-image", "~/Desktop/chart.png")

    assert access is not None
    assert access.path == target.resolve(strict=False)
    assert access.display_path == "~/Desktop/chart.png"
    assert str(access.path) != "/home/oai/Desktop/chart.png"


def test_path_traversal_rejected(tmp_path: Path) -> None:
    with pytest.raises(WorkspaceError):
        resolve_workspace_path(tmp_path, "../secret.txt")


def test_external_read_requires_exact_one_use_approval(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external-read.txt"
    other = tmp_path.parent / f"{tmp_path.name}-other-read.txt"
    outside.write_text("approved contents", encoding="utf-8")
    other.write_text("other contents", encoding="utf-8")
    authorizer = ExternalPathAuthorizer(tmp_path)
    tool = make_read_file_tool(tmp_path, authorizer=authorizer)
    arguments = json.dumps({"path": str(outside)})

    assert "External path requires approval" in invoke(tool, arguments, call_id="external-read")

    proposal = authorizer.propose("read_file", "external-read", str(outside))
    assert proposal is not None
    assert proposal.path == outside.resolve()
    assert authorizer.approve_exact("read_file", "external-read") == proposal
    assert invoke(tool, arguments, call_id="external-read") == "approved contents"
    assert "External path requires approval" in invoke(tool, arguments, call_id="external-read")

    assert authorizer.propose("read_file", "exact-target", str(outside)) is not None
    assert authorizer.approve_exact("read_file", "exact-target") is not None
    assert "External path requires approval" in invoke(
        tool, json.dumps({"path": str(other)}), call_id="exact-target"
    )

    missing = tmp_path.parent / f"{tmp_path.name}-missing-external-read.txt"
    missing_arguments = json.dumps({"path": str(missing)})
    assert authorizer.propose("read_file", "missing-target", str(missing)) is not None
    assert authorizer.approve_exact("read_file", "missing-target") is not None
    assert "does not exist" in invoke(tool, missing_arguments, call_id="missing-target")


def test_external_traversal_and_symlink_paths_are_proposed_and_require_approval(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external.txt"
    outside.write_text("approved contents", encoding="utf-8")
    authorizer = ExternalPathAuthorizer(tmp_path)
    tool = make_read_file_tool(tmp_path, authorizer=authorizer)

    traversal = f"../{outside.name}"
    symlink = tmp_path / "external-link.txt"
    symlink.symlink_to(outside)
    for call_id, entered_path in (("traversal", traversal), ("symlink", symlink.name)):
        arguments = json.dumps({"path": entered_path})
        assert "External path requires approval" in invoke(tool, arguments, call_id=call_id)
        proposal = authorizer.propose("read_file", call_id, entered_path)
        assert proposal is not None
        assert proposal.path == outside.resolve()
        assert proposal.display_path == entered_path
        assert authorizer.approve_exact("read_file", call_id) == proposal
        assert invoke(tool, arguments, call_id=call_id) == "approved contents"


def test_every_filesystem_tool_requires_an_exact_external_grant(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external"
    outside.mkdir()
    (outside / "read.txt").write_text("needle", encoding="utf-8")
    (outside / "replace.txt").write_text("old", encoding="utf-8")
    authorizer = ExternalPathAuthorizer(tmp_path)
    cases = (
        ("read_file", make_read_file_tool(tmp_path, authorizer=authorizer), {"path": str(outside / "read.txt")}, "needle"),
        (
            "list_directory",
            make_list_directory_tool(tmp_path, authorizer=authorizer),
            {"path": str(outside)},
            "file: read.txt",
        ),
        (
            "search_files",
            make_search_files_tool(tmp_path, authorizer=authorizer),
            {"path": str(outside), "query": "needle"},
            "read.txt:1: needle",
        ),
        (
            "write_file",
            make_write_file_tool(tmp_path, authorizer=authorizer),
            {"path": str(outside / "written.txt"), "content": "written"},
            "Wrote",
        ),
        (
            "replace_in_file",
            make_replace_in_file_tool(tmp_path, authorizer=authorizer),
            {"path": str(outside / "replace.txt"), "old_text": "old", "new_text": "new"},
            "Replaced",
        ),
    )

    for tool_name, tool, arguments, expected in cases:
        call_id = f"external-{tool_name}"
        encoded = json.dumps(arguments)
        assert "External path requires approval" in invoke(tool, encoded, call_id=call_id)
        if tool_name == "write_file":
            assert not (outside / "written.txt").exists()
        if tool_name == "replace_in_file":
            assert (outside / "replace.txt").read_text(encoding="utf-8") == "old"
        proposal = authorizer.propose(
            tool_name, call_id, arguments["path"], mutation=tool_name in {"write_file", "replace_in_file"}
        )
        assert proposal is not None
        assert authorizer.approve_exact(tool_name, call_id) == proposal
        assert expected in invoke(tool, encoded, call_id=call_id)

    assert (outside / "written.txt").read_text(encoding="utf-8") == "written"
    assert (outside / "replace.txt").read_text(encoding="utf-8") == "new"


def test_external_write_is_excluded_from_the_task_journal(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-external-write.txt"
    journal = TaskMutationJournal(tmp_path)
    authorizer = ExternalPathAuthorizer(tmp_path)
    tool = make_write_file_tool(tmp_path, journal, authorizer=authorizer)
    arguments = json.dumps({"path": str(outside), "content": "external change"})

    assert "External path requires approval" in invoke(tool, arguments, call_id="external-write")
    assert not outside.exists()
    assert authorizer.propose("write_file", "external-write", str(outside), mutation=True) is not None
    assert authorizer.approve_exact("write_file", "external-write") is not None

    assert "Wrote" in invoke(tool, arguments, call_id="external-write")
    assert outside.read_text(encoding="utf-8") == "external change"
    assert journal.summary()["changed_files"] == []
    assert journal.unified_diff() == ""


def test_write_file_creates_file(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    result = invoke(tool, '{"path":"main.go","content":"package main\\n"}')
    assert "Wrote main.go" in result
    assert (tmp_path / "main.go").read_text(encoding="utf-8") == "package main\n"


def test_write_file_rejects_traversal(tmp_path: Path) -> None:
    tool = make_write_file_tool(tmp_path)
    result = invoke(tool, '{"path":"../secret.txt","content":"nope"}')
    assert "outside the workspace" in result
    assert not (tmp_path.parent / "secret.txt").exists()


def test_list_directory_is_typed_nonrecursive_sorted_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "beta.txt").write_text("z", encoding="utf-8")
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "nested.txt").write_text("nested", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(tmp_path / "beta.txt")
    for index in range(100):
        (tmp_path / f"long-entry-{index:03d}.txt").write_text("x", encoding="utf-8")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")

    result = invoke(make_list_directory_tool(tmp_path), '{"path":"."}')

    lines = result.splitlines()
    entries = [line for line in lines if not line.startswith("...")]
    assert entries == sorted(entries, key=lambda entry: entry.split(": ", 1)[1])
    assert "directory: alpha" in entries
    assert "file: beta.txt" in entries
    assert "symlink: link.txt" in entries
    assert any(line.startswith("file: long-entry-") for line in entries)
    assert "nested.txt" not in result
    assert estimate_tokens(result) <= tool_result_token_budget()


def test_list_and_read_file_give_wrong_kind_guidance(tmp_path: Path) -> None:
    (tmp_path / "directory").mkdir()
    (tmp_path / "file.txt").write_text("contents", encoding="utf-8")

    listed_file = invoke(make_list_directory_tool(tmp_path), '{"path":"file.txt"}')
    read_directory = invoke(make_read_file_tool(tmp_path), '{"path":"directory"}')

    assert listed_file == "Error listing 'file.txt': path is a file; use read_file instead"
    assert read_directory == "Error reading 'directory': path is a directory; use list_directory instead"


def test_list_directory_rejects_workspace_escape(tmp_path: Path) -> None:
    result = invoke(make_list_directory_tool(tmp_path), '{"path":"../"}')

    assert "outside the workspace" in result


def test_search_files_returns_sorted_matches_and_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("needle beta\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("needle alpha\nother needle\n", encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"needle\x00hidden")

    result = invoke(make_search_files_tool(tmp_path), '{"query":"needle","max_results":2}')

    assert result.splitlines() == ["a.txt:1: needle alpha", "a.txt:2: other needle"]
    assert "binary.bin" not in result


def test_search_files_preserves_escape_and_context_protections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for index in range(100):
        (tmp_path / f"file-{index:03d}.txt").write_text("needle " + "x" * 30, encoding="utf-8")
    outside = tmp_path.parent / "outside-search.txt"
    outside.write_text("needle outside\n", encoding="utf-8")
    (tmp_path / "outside-link.txt").symlink_to(outside)
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")
    tool = make_search_files_tool(tmp_path)

    result = invoke(tool, '{"query":"needle","max_results":100}')
    escaped = invoke(tool, '{"query":"needle","path":"../"}')
    linked = invoke(tool, '{"query":"needle","path":"outside-link.txt"}')

    assert estimate_tokens(result) <= tool_result_token_budget()
    assert "outside-link.txt" not in result
    assert "outside the workspace" in escaped
    assert "outside the workspace" in linked


def test_search_files_skips_generated_and_dependency_directories_but_allows_explicit_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "visible.txt").write_text("needle visible\n", encoding="utf-8")
    ignored_files = []
    for directory in (".git", ".hans-tmp", "__pycache__", ".venv", "build", "dist", "node_modules", "target"):
        ignored = tmp_path / directory
        ignored.mkdir()
        ignored_file = ignored / "ignored.txt"
        ignored_file.write_text(f"needle {directory}\n", encoding="utf-8")
        ignored_files.append(ignored_file)

    def read_bytes(*_args, **_kwargs):
        raise AssertionError("search_files must read files line by line")

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    tool = make_search_files_tool(tmp_path)
    recursive = invoke(tool, '{"query":"needle"}')
    explicit = invoke(tool, '{"query":"needle","path":"node_modules/ignored.txt"}')

    assert recursive == "visible.txt:1: needle visible"
    assert explicit == "node_modules/ignored.txt:1: needle node_modules"
    assert all(str(path.relative_to(tmp_path)) not in recursive for path in ignored_files)


def test_search_files_truncates_a_huge_matching_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "huge.txt").write_text("needle " + "x" * 10_000, encoding="utf-8")
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")

    result = invoke(make_search_files_tool(tmp_path), '{"query":"needle"}')

    assert result.startswith("huge.txt:1: needle ")
    assert "[matching line truncated]" in result
    assert estimate_tokens(result) <= tool_result_token_budget()


def test_bounded_lines_reports_an_oversized_omitted_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_CONTEXT_TOKENS", "1024")

    result = _bounded_lines(["x" * 10_000])

    assert result == "... 1 result(s) omitted to fit the tool result budget."
    assert estimate_tokens(result) <= tool_result_token_budget()


def test_replace_in_file_requires_one_match_and_replaces_atomically(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_text("before\nold value\nafter\n", encoding="utf-8")
    target.chmod(0o755)
    original_stat = target.stat()
    original_inode = original_stat.st_ino
    original_mode = original_stat.st_mode & 0o777
    original_mtime_ns = original_stat.st_mtime_ns
    tool = make_replace_in_file_tool(tmp_path)

    result = invoke(tool, '{"path":"main.py","old_text":"old value","new_text":"new value"}')

    assert result == "Replaced text in main.py"
    assert target.read_text(encoding="utf-8") == "before\nnew value\nafter\n"
    assert target.stat().st_ino != original_inode
    assert target.stat().st_mode & 0o777 == original_mode
    assert target.stat().st_mtime_ns >= original_mtime_ns + 1_000_000_000
    (tmp_path / "duplicate.txt").write_text("old\nold\n", encoding="utf-8")
    duplicate = invoke(tool, '{"path":"duplicate.txt","old_text":"old","new_text":"other"}')
    missing = invoke(tool, '{"path":"main.py","old_text":"absent","new_text":"other"}')
    assert "exactly once (found 2)" in duplicate
    assert "exactly once (found 0)" in missing


def test_replace_in_file_rejects_symlink_outside_workspace(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-replace.txt"
    outside.write_text("old", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)

    result = invoke(make_replace_in_file_tool(tmp_path), '{"path":"link.txt","old_text":"old","new_text":"new"}')

    assert "outside the workspace" in result
    assert outside.read_text(encoding="utf-8") == "old"


def test_run_command_returns_stdout_and_defaults_to_inspect(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"printf HELLO_HANS"}')
    assert "exit_code=0" in result
    assert "HELLO_HANS" in result


def test_run_command_rejects_an_invalid_purpose(tmp_path: Path) -> None:
    result = invoke(make_run_command_tool(tmp_path), '{"command":"printf HELLO_HANS","purpose":"plan"}')

    assert result == "Error: purpose must be either 'inspect' or 'verify'"


def test_run_command_allows_literal_go_package_pattern_without_running_go() -> None:
    assert reject_shell_syntax("go test ./...") is None


@pytest.mark.parametrize(
    "command",
    [
        "go test ./... 2>&1",
        "go test ./... && gofmt",
        "cat a | head",
        "echo $(pwd)",
        "ls *.go",
    ],
)
def test_run_command_rejects_shell_syntax(tmp_path: Path, command: str) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, json.dumps({"command": command}))
    assert "not a shell" in result
    assert "exit_code" not in result


def test_run_command_routes_home_paths_to_approved_filesystem_tools(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)

    result = invoke(tool, json.dumps({"command": "find ~/Desktop -maxdepth 1"}))

    assert "does not expand `~`" in result
    assert "list_directory, search_files, or read_file" in result
    assert "external access follows the current external-path policy" in result
    assert "exit_code" not in result


def test_run_command_rejects_parent_path(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"ls .."}')
    assert "escapes the workspace" in result
    assert "exit_code" not in result


def test_run_command_rejects_absolute_path_outside_workspace(tmp_path: Path) -> None:
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"cat /etc/passwd"}')
    assert "outside the workspace" in result
    assert "root:" not in result


def test_run_command_hides_process_secrets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BOLT_MODEL_API_KEY", "super-secret-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    tool = make_run_command_tool(tmp_path)
    result = invoke(tool, '{"command":"env"}')
    assert "exit_code=0" in result
    assert "super-secret-key" not in result
    assert "openai-secret" not in result
    assert f"HOME={tmp_path}" in result


def test_symlink_outside_workspace_rejected(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (tmp_path / "link.txt").symlink_to(outside)
    with pytest.raises(WorkspaceError):
        resolve_workspace_path(tmp_path, "link.txt")


def test_mutation_journal_tracks_original_once_across_write_and_replace(tmp_path: Path) -> None:
    target = tmp_path / "main.py"
    target.write_bytes(b"before\nold\n")
    target.chmod(0o751)
    journal = TaskMutationJournal(tmp_path)

    assert "Wrote main.py" in invoke(
        make_write_file_tool(tmp_path, journal), '{"path":"main.py","content":"intermediate\\nold\\n"}'
    )
    assert invoke(
        make_replace_in_file_tool(tmp_path, journal),
        '{"path":"main.py","old_text":"old","new_text":"new"}',
    ) == "Replaced text in main.py"

    assert journal.summary() == {
        "changed_files": ["main.py"],
        "created_files": [],
        "preexisting_git_worktree_changes": [],
        "git_available": False,
    }
    assert "-before\n" in journal.unified_diff()
    assert "+intermediate\n" in journal.unified_diff()
    assert "+new\n" in journal.unified_diff()
    assert journal.undo() == {"restored": ["main.py"], "removed": [], "conflicts": []}
    assert target.read_bytes() == b"before\nold\n"
    assert target.stat().st_mode & 0o7777 == 0o751


def test_mutation_journal_removes_created_files_and_resets(tmp_path: Path) -> None:
    journal = TaskMutationJournal(tmp_path)
    assert "Wrote nested/new.txt" in invoke(
        make_write_file_tool(tmp_path, journal), '{"path":"nested/new.txt","content":"created"}'
    )
    assert journal.summary()["created_files"] == ["nested/new.txt"]
    assert journal.undo() == {"restored": [], "removed": ["nested/new.txt"], "conflicts": []}
    assert not (tmp_path / "nested" / "new.txt").exists()

    journal.begin_task()
    assert journal.summary()["changed_files"] == []


def test_mutation_journal_refuses_undo_after_external_change(tmp_path: Path) -> None:
    journal = TaskMutationJournal(tmp_path)
    tool = make_write_file_tool(tmp_path, journal)
    invoke(tool, '{"path":"note.txt","content":"by HANS"}')
    (tmp_path / "note.txt").write_text("external", encoding="utf-8")

    assert journal.undo() == {"restored": [], "removed": [], "conflicts": ["note.txt"]}
    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "external"


def test_mutation_journal_reports_git_state_and_bounds_hans_only_diff(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "tracked-later.txt").write_text("original\n", encoding="utf-8")
    journal = TaskMutationJournal(tmp_path, max_diff_chars=90)
    invoke(
        make_write_file_tool(tmp_path, journal),
        json.dumps({"path": "tracked-later.txt", "content": "changed " + "x" * 1_000}),
    )

    summary = journal.summary()
    assert summary["git_available"] is True
    assert summary["preexisting_git_worktree_changes"] == ["tracked-later.txt"]
    diff = journal.unified_diff()
    assert len(diff) <= 90
    assert "HANS-only diff truncated" in diff
    assert diff.startswith("--- a/tracked-later.txt")
