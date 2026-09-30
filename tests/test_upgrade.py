import json
import os
import subprocess
from pathlib import Path
from urllib.error import URLError

import pytest

import bolt_next.main as cli
from bolt_next import upgrade


class _Response:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


def test_upgrade_downloads_tagged_installer_and_sets_archive_url(monkeypatch: pytest.MonkeyPatch) -> None:
    tag = "v1.2.3"
    installer_url = upgrade._installer_url(tag)
    seen: dict[str, object] = {"urls": []}

    def fake_urlopen(url: str, *, timeout: int) -> _Response:
        seen["urls"].append(url)
        assert timeout == 30
        if url == upgrade.LATEST_RELEASE_URL:
            return _Response(json.dumps({"tag_name": tag}).encode())
        assert url == installer_url
        return _Response(b"#!/usr/bin/env bash\nexit 17\n")

    def fake_run(
        command: list[str], *, check: bool, env: dict[str, str] | None
    ) -> subprocess.CompletedProcess[object]:
        seen["command"] = command
        seen["check"] = check
        seen["env"] = env
        assert command[0] == "bash"
        assert Path(command[1]).read_bytes() == b"#!/usr/bin/env bash\nexit 17\n"
        return subprocess.CompletedProcess(command, 17)

    monkeypatch.setenv("HANS_TEST_INHERITED", "present")
    monkeypatch.setattr(upgrade, "urlopen", fake_urlopen)
    monkeypatch.setattr(upgrade.metadata, "version", lambda _name: "1.2.2")
    monkeypatch.setattr(upgrade.subprocess, "run", fake_run)

    assert upgrade.run_upgrade() == 17
    assert seen["urls"] == [upgrade.LATEST_RELEASE_URL, installer_url]
    assert seen["check"] is False
    environment = seen["env"]
    assert isinstance(environment, dict)
    assert environment[upgrade.ARCHIVE_URL_ENV] == upgrade._archive_url(tag)
    assert environment["HANS_TEST_INHERITED"] == "present"


def test_upgrade_exits_successfully_when_latest_release_is_installed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        upgrade,
        "urlopen",
        lambda _url, *, timeout: _Response(b'{"tag_name": "v1.2.3"}'),
    )
    monkeypatch.setattr(upgrade.metadata, "version", lambda _name: "1.2.3")
    monkeypatch.setattr(upgrade, "_run_installer", lambda *_args: pytest.fail("installer ran"))

    assert upgrade.run_upgrade() == 0
    assert capsys.readouterr().out == "HANS 1.2.3 is already up to date (v1.2.3).\n"


def test_upgrade_does_not_downgrade_a_newer_installed_release(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        upgrade,
        "urlopen",
        lambda _url, *, timeout: _Response(b'{"tag_name": "v1.2.3"}'),
    )
    monkeypatch.setattr(upgrade.metadata, "version", lambda _name: "1.3.0")
    monkeypatch.setattr(upgrade, "_run_installer", lambda *_args: pytest.fail("installer ran"))

    assert upgrade.run_upgrade() == 0
    assert capsys.readouterr().out == "HANS 1.3.0 is newer than the latest release (v1.2.3); not downgrading.\n"


@pytest.mark.parametrize("tag", ["1.2.3", "v1.2", "v01.2.3", "v1.2.3rc1"])
def test_upgrade_rejects_invalid_latest_release_tag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tag: str
) -> None:
    monkeypatch.setattr(
        upgrade,
        "urlopen",
        lambda _url, *, timeout: _Response(json.dumps({"tag_name": tag}).encode()),
    )

    assert upgrade.run_upgrade() == 1
    assert capsys.readouterr().err == "error: latest HANS release tag must be a vX.Y.Z version\n"


def test_upgrade_keeps_https_installer_override_as_direct_path(monkeypatch: pytest.MonkeyPatch) -> None:
    override = "https://mirror.example.test/hans/install.sh"
    seen: dict[str, object] = {}
    monkeypatch.setenv(upgrade.INSTALLER_URL_ENV, override)
    monkeypatch.setattr(
        upgrade,
        "_run_installer",
        lambda url, environment=None: seen.update(url=url, environment=environment) or 19,
    )
    monkeypatch.setattr(upgrade, "_latest_tag", lambda: pytest.fail("release lookup ran"))

    assert upgrade.run_upgrade() == 19
    assert seen == {"url": override, "environment": None}


def test_upgrade_reports_latest_release_download_error_without_running_installer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_urlopen(_url: str, *, timeout: int) -> _Response:
        raise URLError("offline")

    monkeypatch.setattr(upgrade, "urlopen", fake_urlopen)
    monkeypatch.setattr(upgrade, "_run_installer", lambda *_args: pytest.fail("installer ran"))

    assert upgrade.run_upgrade() == 1
    assert capsys.readouterr().err == (
        "error: unable to download latest HANS release metadata: <urlopen error offline>\n"
    )


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


def test_install_script_rejects_non_https_archive_url() -> None:
    result = subprocess.run(
        ["bash", str(Path(__file__).parents[1] / "install.sh")],
        check=False,
        capture_output=True,
        env={**os.environ, "HANS_ARCHIVE_URL": "http://example.test/hans.tar.gz"},
        text=True,
    )

    assert result.returncode == 1
    assert result.stderr == "error: HANS_ARCHIVE_URL must be an HTTPS URL\n"


def test_readme_current_release_matches_project_version() -> None:
    root = Path(__file__).parents[1]
    project = (root / "pyproject.toml").read_text()
    readme = (root / "README.md").read_text()
    version_line = next(line for line in project.splitlines() if line.startswith("version = "))
    version = version_line.removeprefix('version = "').removesuffix('"')

    assert f"Current release: **{version}**." in readme
