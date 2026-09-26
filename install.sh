#!/usr/bin/env bash
# Install the hans command for the current user.
# curl -LsSf https://raw.githubusercontent.com/saurabhahuja71/hans/main/install.sh | bash
set -euo pipefail

ARCHIVE_URL="${HANS_ARCHIVE_URL:-https://github.com/saurabhahuja71/hans/archive/refs/heads/main.tar.gz}"
INSTALL_ROOT="${HANS_HOME:-${HOME}/.hans}"
BIN_DIR="${HOME}/.local/bin"

info() { printf '==> %s\n' "$1"; }
die() { printf 'error: %s\n' "$1" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"
command -v tar >/dev/null 2>&1 || die "tar is required"

PY=""
for candidate in python3.12 python3; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)'; then
            PY="$candidate"
            break
        fi
    fi
done
[[ -n "$PY" ]] || die "Python 3.12 or newer is required"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

info "Downloading HANS"
curl -fsSL -L "$ARCHIVE_URL" | tar -xz -C "$tmp"
source_dir="$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
[[ -n "$source_dir" && -f "$source_dir/pyproject.toml" ]] || die "Downloaded archive did not contain HANS"

info "Installing into ${INSTALL_ROOT}"
mkdir -p "$INSTALL_ROOT" "$BIN_DIR"
"$PY" -m venv "$INSTALL_ROOT/venv"
"$INSTALL_ROOT/venv/bin/python" -m pip install -U pip
"$INSTALL_ROOT/venv/bin/python" -m pip install --upgrade "$source_dir"
ln -sfn "$INSTALL_ROOT/venv/bin/hans" "$BIN_DIR/hans"

info "Installed ${BIN_DIR}/hans"
case ":$PATH:" in
    *":${BIN_DIR}:"*) ;;
    *)
        printf '\nAdd this to your shell profile, then open a new terminal:\n'
        printf '  export PATH="%s:$PATH"\n' "$BIN_DIR"
        ;;
esac
printf '\nStart HANS after exporting BOLT_MODEL_BASE_URL and BOLT_MODEL_API_KEY:\n'
printf '  hans\n'
printf '\nUpgrade later with:\n'
printf '  hans upgrade\n'
