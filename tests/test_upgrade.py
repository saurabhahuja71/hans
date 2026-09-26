import subprocess
from pathlib import Path
from urllib.error import URLError

import pytest

import bolt_next.main as cli
from bolt_next import upgrade


class _Response:
    def __init__(self, installer: bytes) -> None:
        self.installer = installer

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.installer


def test_upgrade_downloads_installer_and_returns_bash_status(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_urlopen(url: str, *, timeout: int) -> _Response:
        seen["url"] = url
        seen["timeout"] = timeout
        return _Response(b"#!/usr/bin/env bash\nexit 17\n")

    def fake_run(command: list[str], *, check: bool) -> subprocess.CompletedProcess[object]:
        seen["command"] = command
        seen["check"] = check
        assert command[0] == "bash"
        assert Path(command[1]).read_bytes() == b"#!/usr/bin/env bash\nexit 17\n"
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setattr(upgrade, "urlopen", fake_urlopen)
    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)

    assert upgrade.run_upgrade() == 17
    assert seen["url"] == upgrade.CANONICAL_INSTALLER_URL
    assert seen["timeout"] == 30
    assert seen["check"] is False


def test_installer_url_uses_override(monkeypatch: pytest.MonkeyPatch) -> None:
    override = "https://mirror.example.test/hans/install.sh"
    monkeypatch.setenv(upgrade.INSTALLER_URL_ENV, override)

    assert upgrade.installer_url() == override


def test_upgrade_reports_download_error_without_running_installer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_urlopen(_url: str, *, timeout: int) -> _Response:
        raise URLError("offline")

    monkeypatch.setattr(upgrade, "urlopen", fake_urlopen)
    monkeypatch.setattr(upgrade.subprocess, "run", lambda *_args, **_kwargs: pytest.fail("installer ran"))

    assert upgrade.run_upgrade() == 1
    assert capsys.readouterr().err == "error: unable to download HANS installer: <urlopen error offline>\n"


def test_upgrade_rejects_non_https_override(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(upgrade.INSTALLER_URL_ENV, "http://example.test/install.sh")

    assert upgrade.run_upgrade() == 1
    assert capsys.readouterr().err == "error: HANS_INSTALLER_URL must be an HTTPS URL\n"


def test_main_without_arguments_starts_tui(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(cli, "run_tui", lambda: calls.append("tui"))

    assert cli.main([]) == 0
    assert calls == ["tui"]


@pytest.mark.parametrize("command", ["upgrade", "update"])
def test_main_dispatches_upgrade_aliases(monkeypatch: pytest.MonkeyPatch, command: str) -> None:
    monkeypatch.setattr(cli, "run_upgrade", lambda: 23)

    assert cli.main([command]) == 23


def test_main_reports_invalid_usage(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["other"]) == 2
    assert capsys.readouterr().err == "usage: hans [upgrade|update]\n"
