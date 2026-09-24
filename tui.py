import asyncio

from bolt_next.agent import run_agent


def run_tui() -> None:
    print("╭──────────────────────────────╮")
    print("│          Bolt Next           │")
    print("│   OpenAI Agents SDK runtime  │")
    print("╰──────────────────────────────╯")

    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return

        if prompt.strip().lower() in {"exit", "quit"}:
            return

        if not prompt.strip():
            continue

        try:
            response = asyncio.run(run_agent(prompt))
            print(f"\n{response}")
        except Exception as exc:
            print(f"\nError: {exc}")
