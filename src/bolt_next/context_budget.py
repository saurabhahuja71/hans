"""Keep model requests inside the configured context size.

The token estimate is a character budget derived from one configured limit.
It is a guard, not a tokenizer, and it does not replace the Agents SDK session.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from copy import deepcopy

from agents.run_config import CallModelData, ModelInputData

from bolt_next.errors import ConfigurationError

DEFAULT_CONTEXT_TOKENS = 16384


def context_token_limit(context_tokens: int | None = None) -> int:
    """Return an explicit capacity or the legacy environment-backed capacity."""
    if context_tokens is None:
        raw = os.environ.get("BOLT_MODEL_CONTEXT_TOKENS", "").strip()
        if not raw:
            return DEFAULT_CONTEXT_TOKENS
        try:
            limit = int(raw)
        except ValueError as exc:
            raise ConfigurationError("BOLT_MODEL_CONTEXT_TOKENS must be an integer") from exc
    else:
        limit = context_tokens
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise ConfigurationError("context_tokens must be an integer")
    if limit < 1024:
        variable = "BOLT_MODEL_CONTEXT_TOKENS" if context_tokens is None else "context_tokens"
        raise ConfigurationError(f"{variable} must be at least 1024")
    return limit


def completion_token_reserve(context_tokens: int | None = None) -> int:
    return context_token_limit(context_tokens) - max_input_tokens(context_tokens)


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    # Tool payloads are mostly source code, JSON, and paths. Those tokenize
    # more densely than ordinary prose, so 4 characters/token is unsafe for
    # smaller-context deployments. Keep a conservative estimate here; it is a
    # safety guard, not an attempt to replace the provider's
    # tokenizer.
    return (len(text) + 2) // 3


def max_input_tokens(context_tokens: int | None = None) -> int:
    """Input budget, leaving a quarter of the context for the model reply."""
    return max(256, context_token_limit(context_tokens) * 3 // 4)


def tool_result_token_budget(context_tokens: int | None = None) -> int:
    """Largest single tool result that can still leave room for the rest of the turn."""
    return max(128, max_input_tokens(context_tokens) // 4)


def request_tokens(instructions: str | None, items: list) -> int:
    payload = json.dumps(items, default=str, ensure_ascii=False)
    return estimate_tokens(instructions or "") + estimate_tokens(payload)


def _output_text(item) -> str | None:
    if isinstance(item, dict):
        output = item.get("output")
        return output if isinstance(output, str) else None
    output = getattr(item, "output", None)
    return output if isinstance(output, str) else None


def _set_output(item, text: str):
    if isinstance(item, dict):
        updated = dict(item)
        updated["output"] = text
        return updated
    cloned = deepcopy(item)
    setattr(cloned, "output", text)
    return cloned


def _omitted_notice(context_tokens: int) -> str:
    return (
        "Tool output omitted from this model request because including it would "
        f"exceed the context budget of {context_tokens} tokens. "
        "The authoritative output is unchanged in the session and on disk. "
        "This is not a summary. Request a smaller read_file range or a narrower command."
    )


def _fit_model_input(data: CallModelData, context_tokens: int) -> ModelInputData:
    instructions = data.model_data.instructions
    items = list(data.model_data.input)
    limit = max_input_tokens(context_tokens)
    if request_tokens(instructions, items) <= limit:
        return ModelInputData(input=items, instructions=instructions)

    notice = _omitted_notice(context_tokens)
    fitted = []
    for item in items:
        text = _output_text(item)
        if text is not None and estimate_tokens(text) > tool_result_token_budget(context_tokens):
            fitted.append(_set_output(item, notice))
        else:
            fitted.append(item)
    if request_tokens(instructions, fitted) <= limit:
        return ModelInputData(input=fitted, instructions=instructions)

    # Last resort: keep the newest items that fit, plus the omission notice.
    kept: list = []
    for item in reversed(fitted):
        trial = [item, *kept]
        if request_tokens(instructions, trial) > limit:
            break
        kept = trial
    if not kept:
        kept = [{"role": "user", "content": notice}]
    elif request_tokens(instructions, kept) > limit:
        kept = [{"role": "user", "content": notice}]
    return ModelInputData(input=kept, instructions=instructions)


def make_fit_model_input(context_tokens: int) -> Callable[[CallModelData], ModelInputData]:
    """Build a model-input filter bound to one selected model capacity."""
    selected_context_tokens = context_token_limit(context_tokens)

    def fit(data: CallModelData) -> ModelInputData:
        return _fit_model_input(data, selected_context_tokens)

    return fit


def fit_model_input(data: CallModelData) -> ModelInputData:
    """Return model input that fits the legacy configured context budget."""
    return _fit_model_input(data, context_token_limit())
