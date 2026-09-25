from bolt_next.tui import read_user_message


class _Lines:
    def __init__(self, lines: list[object]) -> None:
        self._lines = iter(lines)
        self._eof = False

    def __call__(self, _prompt: str) -> str:
        if self._eof:
            raise EOFError
        item = next(self._lines, None)
        if item is None:
            self._eof = True
            raise EOFError
        return str(item)


def test_multiline_text_is_one_message() -> None:
    message = read_user_message(
        _Lines(
            [
                "Inspect this repository.",
                "",
                "Run the tests.",
                None,
            ]
        )
    )
    assert message == "Inspect this repository.\n\nRun the tests."


def test_embedded_newlines_are_preserved() -> None:
    message = read_user_message(_Lines(["alpha", "beta", "gamma", None]))
    assert message is not None
    assert message.split("\n") == ["alpha", "beta", "gamma"]


def test_one_pasted_block_is_one_message_and_then_stop() -> None:
    read_line = _Lines(
        [
            "Inspect this repository.",
            "Find the relevant Go implementation and tests.",
            "Diagnose the problem.",
            None,
        ]
    )
    first = read_user_message(read_line)
    second = read_user_message(read_line)
    assert first == (
        "Inspect this repository.\n"
        "Find the relevant Go implementation and tests.\n"
        "Diagnose the problem."
    )
    assert second is None


def test_lines_are_not_separate_messages() -> None:
    read_line = _Lines(["one", "two", None])
    messages = []
    while True:
        message = read_user_message(read_line)
        if message is None:
            break
        messages.append(message)
    assert messages == ["one\ntwo"]
