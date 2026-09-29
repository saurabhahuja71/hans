from __future__ import annotations

import os


DEFAULT_MODEL = "qwen3.6-27b"


def configured_model() -> str:
    return os.environ.get("BOLT_MODEL", DEFAULT_MODEL)
