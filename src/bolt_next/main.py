import sys
from collections.abc import Sequence

from bolt_next.tui import run_tui
from bolt_next.upgrade import run_upgrade


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        run_tui()
        return 0
    if len(arguments) == 1 and arguments[0] in {"upgrade", "update"}:
        return run_upgrade()
    print("usage: hans [upgrade|update]", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
