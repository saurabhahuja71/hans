"""Local slash-command registry and deterministic composer completion."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    category: str
    usage: str
    summary: str


COMMANDS = (
    CommandSpec("/help", "General", "/help", "Show local command help."),
    CommandSpec("/clear", "General", "/clear", "Clear conversation history."),
    CommandSpec("/models", "Models", "/models [use <model>]", "Show or select a configured model."),
    CommandSpec("/mode", "Models", "/mode [mode]", "Show or set the reasoning mode."),
    CommandSpec(
        "/permissions",
        "Permissions",
        "/permissions [read|write|execute] [allow|deny|ask]",
        "Show or set local tool permissions.",
    ),
    CommandSpec("/todo", "Workspace / productivity", "/todo [list|add|done|remove|clear]", "Manage the local TODO list."),
    CommandSpec("/theme", "Workspace / productivity", "/theme [theme]", "Show or set the session theme."),
)

COMMAND_BY_NAME = {command.name: command for command in COMMANDS}
TODO_VERBS = ("list", "add", "done", "remove", "clear")
PERMISSION_CATEGORIES = ("read", "write", "execute")
PERMISSION_POLICIES = ("allow", "deny", "ask")
KEYBOARD_SHORTCUTS = (
    ("Enter", "Send"),
    ("Shift+Enter", "New line"),
    ("Ctrl-D", "Send / exit when empty"),
    ("Ctrl-C", "Cancel"),
    ("Ctrl-G", "Diff"),
    ("Ctrl-Z", "Undo"),
    ("Ctrl-O", "Output"),
    ("Ctrl-Q", "Quit"),
)
MAX_SUGGESTIONS = len(COMMANDS)


@dataclass(frozen=True)
class CompletionContext:
    model_ids: tuple[str, ...] = ()
    reasoning_modes: tuple[str, ...] = ()
    permission_categories: tuple[str, ...] = PERMISSION_CATEGORIES
    permission_policies: tuple[str, ...] = PERMISSION_POLICIES
    themes: tuple[str, ...] = ()
    todo_verbs: tuple[str, ...] = TODO_VERBS


def command_names() -> tuple[str, ...]:
    return tuple(command.name for command in COMMANDS)


def known_command(token: str) -> CommandSpec | None:
    return COMMAND_BY_NAME.get(token.lower())


def is_complete_command(text: str) -> bool:
    """Whether text is an executable no-argument local command."""
    if "\n" in text:
        return False
    command, _, argument = text.strip().partition(" ")
    return bool(known_command(command)) and not argument.strip()


def format_help() -> str:
    lines = ["HANS commands"]
    category = ""
    for command in COMMANDS:
        if command.category != category:
            category = command.category
            lines.extend(("", category))
        lines.append(f"  {command.usage:<62} {command.summary}")
    lines.extend(("", "Keyboard"))
    lines.extend(f"  {key:<12} {description}" for key, description in KEYBOARD_SHORTCUTS)
    lines.extend(("", "Type / in an idle composer to discover commands and arguments."))
    return "\n".join(lines)


def command_suggestions(text: str, context: CompletionContext, *, limit: int = MAX_SUGGESTIONS) -> tuple[str, ...]:
    """Return local, bounded full-text completions for an eligible composer value."""
    if limit < 1 or "\n" in text or not text.startswith("/"):
        return ()
    command, separator, argument = text.partition(" ")
    spec = known_command(command)
    if spec is None:
        if separator:
            return ()
        return _matching(command, command_names(), limit)
    return _argument_suggestions(spec.name, argument if separator else "", context, limit)


def _argument_suggestions(command: str, argument: str, context: CompletionContext, limit: int) -> tuple[str, ...]:
    if command == "/models":
        values = argument.split()
        if not values:
            return ("/models use",)
        if len(values) == 1:
            if values[0].lower() == "use":
                return _prefixed("/models use ", "", context.model_ids, limit)
            if "use".startswith(values[0].lower()):
                return ("/models use",)
        if len(values) == 2 and values[0].lower() == "use":
            return _prefixed("/models use ", values[1], context.model_ids, limit)
        return ()
    if command == "/mode":
        if len(argument.split()) > 1:
            return ()
        return _prefixed("/mode ", argument.strip(), context.reasoning_modes, limit)
    if command == "/permissions":
        values = argument.split()
        if not values:
            return _prefixed("/permissions ", "", context.permission_categories, limit)
        if len(values) == 1:
            category = values[0].lower()
            if category in context.permission_categories:
                return _prefixed(f"/permissions {category} ", "", context.permission_policies, limit)
            return _prefixed("/permissions ", values[0], context.permission_categories, limit)
        if len(values) == 2 and values[0].lower() in context.permission_categories:
            return _prefixed(
                f"/permissions {values[0].lower()} ", values[1], context.permission_policies, limit
            )
        return ()
    if command == "/theme":
        if len(argument.split()) > 1:
            return ()
        return _prefixed("/theme ", argument.strip(), context.themes, limit)
    if command == "/todo":
        if len(argument.split()) > 1:
            return ()
        return _prefixed("/todo ", argument.strip(), context.todo_verbs, limit)
    return ()


def _matching(prefix: str, values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    normalized = prefix.lower()
    return tuple(value for value in values if value.lower().startswith(normalized))[:limit]


def _prefixed(prefix: str, typed: str, values: tuple[str, ...], limit: int) -> tuple[str, ...]:
    normalized = typed.lower()
    return tuple(
        prefix + value
        for value in values
        if value.lower().startswith(normalized) and (not typed or value.lower() != normalized)
    )[:limit]
