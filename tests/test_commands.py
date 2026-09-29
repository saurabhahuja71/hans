from bolt_next.commands import (
    COMMANDS,
    KEYBOARD_SHORTCUTS,
    CompletionContext,
    command_names,
    command_suggestions,
    format_help,
    is_complete_command,
    known_command,
)
from bolt_next.tui_screen import TodoList, handle_local_command


def test_registry_contains_the_supported_local_commands_once() -> None:
    names = command_names()

    assert names == ("/help", "/clear", "/models", "/mode", "/permissions", "/todo", "/theme")
    assert len(names) == len(set(names)) == len(COMMANDS)
    assert known_command("/MODELS") is not None
    assert known_command("/unknown") is None
    assert is_complete_command("/MODELS")
    assert is_complete_command("/permissions   ")
    assert not is_complete_command("/permissions write")
    assert not is_complete_command("explain /mode")


def test_parser_recognizes_every_registry_command_and_keeps_unknown_commands_local() -> None:
    todos = TodoList()

    assert all(handle_local_command(command.name, todos).handled for command in COMMANDS)
    assert handle_local_command("/unknown", todos).text == "Unknown command: /unknown"


def test_registry_generated_help_covers_commands_categories_and_keyboard_shortcuts() -> None:
    help_text = format_help()

    assert help_text.startswith("HANS commands")
    for command in COMMANDS:
        assert command.category in help_text
        assert command.usage in help_text
        assert command.summary in help_text
    assert "Keyboard" in help_text
    for key, description in KEYBOARD_SHORTCUTS:
        assert key in help_text
        assert description in help_text
    assert "/compact" not in help_text
    assert "idle composer" in help_text


def test_command_suggestions_are_canonical_bounded_and_contextual() -> None:
    context = CompletionContext(
        model_ids=("configured-model", "large"),
        reasoning_modes=("none", "high"),
        themes=("dark", "light"),
    )

    assert command_suggestions("/", context) == command_names()
    assert command_suggestions("/mo", context) == ("/models", "/mode")
    assert command_suggestions("/models", context) == ("/models use",)
    assert command_suggestions("/models ", context) == ("/models use",)
    assert command_suggestions("/models use", context) == (
        "/models use configured-model",
        "/models use large",
    )
    assert command_suggestions("/models use l", context) == ("/models use large",)
    assert command_suggestions("/mode", context) == ("/mode none", "/mode high")
    assert command_suggestions("/mode h", context) == ("/mode high",)
    assert command_suggestions("/permissions", context) == (
        "/permissions read",
        "/permissions write",
        "/permissions execute",
    )
    assert command_suggestions("/permissions write", context) == (
        "/permissions write allow",
        "/permissions write deny",
        "/permissions write ask",
    )
    assert command_suggestions("/permissions write a", context) == (
        "/permissions write allow",
        "/permissions write ask",
    )
    assert command_suggestions("/theme", context) == ("/theme dark", "/theme light")
    assert command_suggestions("/theme l", context) == ("/theme light",)
    assert command_suggestions("/todo", context) == (
        "/todo list",
        "/todo add",
        "/todo done",
        "/todo remove",
        "/todo clear",
    )
    assert command_suggestions("/todo d", context) == ("/todo done",)


def test_command_suggestions_require_a_single_leading_slash_command() -> None:
    context = CompletionContext(themes=("dark",))

    assert command_suggestions("write /mode", context) == ()
    assert command_suggestions("explain /tmp/config.yaml", context) == ()
    assert command_suggestions("/tmp/config.yaml", context) == ()
    assert command_suggestions(" /mode", context) == ()
    assert command_suggestions("/mode h\n", context) == ()
    assert command_suggestions("/mode", context, limit=0) == ()
