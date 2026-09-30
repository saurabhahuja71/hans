import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

REPOSITORY = "saurabhahuja71/hans"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"
INSTALLER_URL_ENV = "HANS_INSTALLER_URL"
ARCHIVE_URL_ENV = "HANS_ARCHIVE_URL"
TAG_PATTERN = re.compile(r"^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def _version_parts(version: str) -> tuple[int, int, int] | None:
    match = TAG_PATTERN.fullmatch(f"v{version}")
    if match is None:
        return None
    return tuple(int(component) for component in match.groups())


def _is_https_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(parsed.netloc)


def _download(url: str, description: str) -> bytes | None:
    try:
        with urlopen(url, timeout=30) as response:
            return response.read()
    except (HTTPError, URLError, OSError, ValueError) as error:
        print(f"error: unable to download {description}: {error}", file=sys.stderr)
        return None


def _latest_tag() -> str | None:
    release = _download(LATEST_RELEASE_URL, "latest HANS release metadata")
    if release is None:
        return None
    try:
        payload = json.loads(release.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("error: latest HANS release metadata is not valid JSON", file=sys.stderr)
        return None
    tag = payload.get("tag_name") if isinstance(payload, dict) else None
    if not isinstance(tag, str) or TAG_PATTERN.fullmatch(tag) is None:
        print("error: latest HANS release tag must be a vX.Y.Z version", file=sys.stderr)
        return None
    return tag


def _installed_version() -> str | None:
    try:
        return metadata.version("hans")
    except metadata.PackageNotFoundError:
        print("error: unable to determine the installed HANS version", file=sys.stderr)
        return None


def _installer_url(tag: str) -> str:
    return f"https://raw.githubusercontent.com/{REPOSITORY}/{tag}/install.sh"


def _archive_url(tag: str) -> str:
    return f"https://github.com/{REPOSITORY}/archive/refs/tags/{tag}.tar.gz"


def _run_installer(url: str, environment: dict[str, str] | None = None) -> int:
    installer = _download(url, "HANS installer")
    if installer is None:
        return 1
    try:
        with tempfile.TemporaryDirectory(prefix="hans-upgrade-") as directory:
            installer_path = Path(directory) / "install.sh"
            installer_path.write_bytes(installer)
            return subprocess.run(["bash", str(installer_path)], check=False, env=environment).returncode
    except OSError as error:
        print(f"error: unable to run HANS installer: {error}", file=sys.stderr)
        return 1


def run_upgrade() -> int:
    override = os.environ.get(INSTALLER_URL_ENV)
    if override is not None:
        if not _is_https_url(override):
            print(f"error: {INSTALLER_URL_ENV} must be an HTTPS URL", file=sys.stderr)
            return 1
        return _run_installer(override)

    tag = _latest_tag()
    if tag is None:
        return 1
    installed_version = _installed_version()
    if installed_version is None:
        return 1
    release_version = tag[1:]
    installed_parts = _version_parts(installed_version)
    release_parts = _version_parts(release_version)
    if installed_parts is not None and release_parts is not None and installed_parts >= release_parts:
        if installed_parts == release_parts:
            print(f"HANS {installed_version} is already up to date ({tag}).")
        else:
            print(f"HANS {installed_version} is newer than the latest release ({tag}); not downgrading.")
        return 0

    archive_url = _archive_url(tag)
    return _run_installer(
        _installer_url(tag),
        {**os.environ, ARCHIVE_URL_ENV: archive_url},
    )
