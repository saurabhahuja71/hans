import importlib.util
from pathlib import Path

import pytest


RELEASE_SCRIPT = Path(__file__).parents[1] / "scripts" / "release.py"
SPEC = importlib.util.spec_from_file_location("release", RELEASE_SCRIPT)
assert SPEC is not None and SPEC.loader is not None
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


def _write_release_files(root: Path, version: str = "1.2.3") -> None:
    (root / "pyproject.toml").write_text(f'[project]\nversion = "{version}"\n', encoding="utf-8")
    (root / "README.md").write_text(
        f"HANS. Current release: **{version}**.\n\n## What {version} provides\n",
        encoding="utf-8",
    )


def test_bump_version_supports_minor_and_patch() -> None:
    assert release.bump_version("1.2.3", "minor") == "1.3.0"
    assert release.bump_version("1.2.3", "patch") == "1.2.4"


@pytest.mark.parametrize("version", ["1.2", "01.2.3", "1.2.3rc1"])
def test_bump_version_rejects_non_release_versions(version: str) -> None:
    with pytest.raises(ValueError, match="X.Y.Z"):
        release.bump_version(version, "patch")


def test_update_release_files_bumps_minor_versions_together(tmp_path: Path) -> None:
    _write_release_files(tmp_path)

    assert release.update_release_files(tmp_path, "minor") == "1.3.0"
    assert 'version = "1.3.0"' in (tmp_path / "pyproject.toml").read_text()
    readme = (tmp_path / "README.md").read_text()
    assert "Current release: **1.3.0**." in readme
    assert "## What 1.3.0 provides" in readme


def test_update_release_files_refuses_inconsistent_readme_without_mutating_project(tmp_path: Path) -> None:
    _write_release_files(tmp_path)
    (tmp_path / "README.md").write_text("Current release: **9.9.9**.\n", encoding="utf-8")

    with pytest.raises(ValueError, match="current release declaration"):
        release.update_release_files(tmp_path, "patch")

    assert (tmp_path / "pyproject.toml").read_text() == '[project]\nversion = "1.2.3"\n'


def test_main_prints_manual_release_instructions(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(release, "update_release_files", lambda _root, _kind: "1.2.4")

    assert release.main(["patch"]) == 0
    output = capsys.readouterr().out
    assert "git tag v1.2.4" in output
    assert "git push origin HEAD v1.2.4" in output


def test_main_rejects_invalid_release_kind(capsys: pytest.CaptureFixture[str]) -> None:
    assert release.main(["major"]) == 2
    assert capsys.readouterr().err == "usage: python scripts/release.py {minor|patch}\n"
