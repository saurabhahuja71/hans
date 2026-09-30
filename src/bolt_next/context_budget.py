"""Keep model requests inside the configured context size.

The token estimate is a character budget derived from one configured limit.
It is a guard, not a tokenizer, and it does not replace the Agents SDK session.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

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


def _item_value(item: object, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _input_image_mime(image_url: str) -> str | None:
    if not image_url.startswith("data:"):
        return None
    header, separator, _payload = image_url[5:].partition(",")
    if not separator:
        return None
    mime_type, separator, _parameters = header.partition(";")
    if not separator or not mime_type.startswith("image/") or len(mime_type) > 127:
        return None
    return mime_type


def _budget_payload(value: object) -> object:
    if isinstance(value, Mapping):
        item_type = value.get("type")
        result: dict[object, object] = {}
        for key, child in value.items():
            if key == "image_url" and item_type == "input_image" and isinstance(child, str) and child.startswith("data:"):
                mime_type = _input_image_mime(child)
                result[key] = f"data:{mime_type};base64,[image payload omitted]" if mime_type else "data:[image payload omitted]"
                continue
            result[key] = _budget_payload(child)
        return result
    if isinstance(value, list | tuple):
        return [_budget_payload(child) for child in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _budget_payload(model_dump(mode="json", exclude_none=True))
        except (TypeError, ValueError):
            pass
    return value


def _contains_input_image(value: object) -> bool:
    if isinstance(value, Mapping):
        return value.get("type") == "input_image" or any(_contains_input_image(child) for child in value.values())
    if isinstance(value, list | tuple):
        return any(_contains_input_image(child) for child in value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _contains_input_image(model_dump(mode="json", exclude_none=True))
        except (TypeError, ValueError):
            return False
    return False


def _image_mime_types(value: object) -> set[str]:
    if isinstance(value, Mapping):
        mime_types = set()
        if value.get("type") == "input_image":
            image_url = value.get("image_url")
            if isinstance(image_url, str):
                mime_type = _input_image_mime(image_url)
                if mime_type is not None:
                    mime_types.add(mime_type)
        for child in value.values():
            mime_types.update(_image_mime_types(child))
        return mime_types
    if isinstance(value, list | tuple):
        return set().union(*(_image_mime_types(child) for child in value)) if value else set()
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _image_mime_types(model_dump(mode="json", exclude_none=True))
        except (TypeError, ValueError):
            return set()
    return set()


def model_input_propagation_summary(items: list) -> dict[str, object]:
    """Return safe image propagation metadata without rendering model payloads."""
    image_items = [item for item in items if _contains_input_image(item)]
    return {
        "input_image_item_count": len(image_items),
        "image_mime_types": tuple(sorted(set().union(*(_image_mime_types(item) for item in image_items)))) if image_items else (),
        "image_function_call_output_count": sum(
            _item_value(item, "type") == "function_call_output" for item in image_items
        ),
    }


def request_tokens(instructions: str | None, items: list) -> int:
    payload = json.dumps(_budget_payload(items), default=str, ensure_ascii=False)
    return estimate_tokens(instructions or "") + estimate_tokens(payload)


def _output_text(item: object) -> str | None:
    output = _item_value(item, "output")
    return output if isinstance(output, str) else None


def _set_output(item: object, text: str) -> object:
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


def _current_image_group(items: list) -> set[int]:
    image_output_indices = [
        index
        for index, item in enumerate(items)
        if _item_value(item, "type") == "function_call_output" and _contains_input_image(item)
    ]
    if not image_output_indices:
        return set()
    output_index = image_output_indices[-1]
    group = {output_index}
    call_id = _item_value(items[output_index], "call_id")
    if call_id is None:
        return group
    for index in range(output_index - 1, -1, -1):
        item = items[index]
        if _item_value(item, "type") == "function_call" and _item_value(item, "call_id") == call_id:
            group.add(index)
            break
    return group


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

    required_indices = _current_image_group(fitted)
    required = [fitted[index] for index in sorted(required_indices)]
    if required and request_tokens(instructions, required) > limit:
        raise ConfigurationError(
            "Current image tool output and its function-call continuation cannot fit within the configured context budget."
        )

    kept_indices = set(required_indices)
    for index in range(len(fitted) - 1, -1, -1):
        if index in kept_indices:
            continue
        trial_indices = kept_indices | {index}
        trial = [fitted[position] for position in sorted(trial_indices)]
        if request_tokens(instructions, trial) <= limit:
            kept_indices = trial_indices
    if kept_indices:
        kept = [fitted[index] for index in sorted(kept_indices)]
    else:
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
