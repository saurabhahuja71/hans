#!/usr/bin/env python3
"""Prepare the project files for a minor or patch release without using Git."""

from __future__ import annotations

import re
import sys
from pathlib import Path

VERSION_PATTERN = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
PROJECT_VERSION_PATTERN = re.compile(r'(?m)^version = "([^"]+)"$')


def bump_version(version: str, kind: str) -> str:
    match = VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError(f"project version must be X.Y.Z, got {version!r}")
    major, minor, patch = (int(component) for component in match.groups())
    if kind == "minor":
        return f"{major}.{minor + 1}.0"
    if kind == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError("release kind must be 'minor' or 'patch'")


def project_version(project: Path) -> str:
    matches = PROJECT_VERSION_PATTERN.findall(project.read_text(encoding="utf-8"))
    if len(matches) != 1:
        raise ValueError("pyproject.toml must contain exactly one project version")
    version = matches[0]
    if VERSION_PATTERN.fullmatch(version) is None:
        raise ValueError(f"project version must be X.Y.Z, got {version!r}")
    return version


def update_release_files(root: Path, kind: str) -> str:
    project = root / "pyproject.toml"
    readme = root / "README.md"
    current = project_version(project)
    next_version = bump_version(current, kind)
    project_text = project.read_text(encoding="utf-8")
    readme_text = readme.read_text(encoding="utf-8")
    current_release = f"Current release: **{current}**."
    release_heading = f"## What {current} provides"
    if readme_text.count(current_release) != 1:
        raise ValueError("README.md must contain exactly one current release declaration")
    if readme_text.count(release_heading) != 1:
        raise ValueError("README.md must contain exactly one current release heading")

    updated_project, replacements = PROJECT_VERSION_PATTERN.subn(
        f'version = "{next_version}"', project_text
    )
    if replacements != 1:
        raise ValueError("pyproject.toml must contain exactly one project version")
    updated_readme = readme_text.replace(current_release, f"Current release: **{next_version}**.")
    updated_readme = updated_readme.replace(release_heading, f"## What {next_version} provides")
    project.write_text(updated_project, encoding="utf-8")
    readme.write_text(updated_readme, encoding="utf-8")
    return next_version


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1 or arguments[0] not in {"minor", "patch"}:
        print("usage: python scripts/release.py {minor|patch}", file=sys.stderr)
        return 2
    try:
        version = update_release_files(Path(__file__).resolve().parents[1], arguments[0])
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    tag = f"v{version}"
    print(f"Updated pyproject.toml and README.md for {tag}.")
    print("Review the changes, then run these commands manually:")
    print(f'  git commit -am "Release {tag}"')
    print(f"  git tag {tag}")
    print(f"  git push origin HEAD {tag}")
    print("Pushing the tag starts the GitHub release workflow.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
