import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import urlopen

CANONICAL_INSTALLER_URL = "https://raw.githubusercontent.com/saurabhahuja71/hans/main/install.sh"
INSTALLER_URL_ENV = "HANS_INSTALLER_URL"


def installer_url() -> str:
    return os.environ.get(INSTALLER_URL_ENV, CANONICAL_INSTALLER_URL)


def run_upgrade() -> int:
    url = installer_url()
    try:
        parsed = urlparse(url)
    except ValueError:
        print(f"error: {INSTALLER_URL_ENV} must be an HTTPS URL", file=sys.stderr)
        return 1
    if parsed.scheme != "https" or not parsed.netloc:
        print(f"error: {INSTALLER_URL_ENV} must be an HTTPS URL", file=sys.stderr)
        return 1

    try:
        with urlopen(url, timeout=30) as response:
            installer = response.read()
    except (HTTPError, URLError, OSError, ValueError) as error:
        print(f"error: unable to download HANS installer: {error}", file=sys.stderr)
        return 1

    try:
        with tempfile.TemporaryDirectory(prefix="hans-upgrade-") as directory:
            installer_path = Path(directory) / "install.sh"
            installer_path.write_bytes(installer)
            return subprocess.run(["bash", str(installer_path)], check=False).returncode
    except OSError as error:
        print(f"error: unable to run HANS installer: {error}", file=sys.stderr)
        return 1
