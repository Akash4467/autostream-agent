"""
main.py

CLI entry point for the AutoStream AI agent.
Useful for quick testing without the Streamlit UI.

Run:
    python main.py
"""

from __future__ import annotations

import sys
from agent.graph import chat, AgentState

BANNER = """
╔══════════════════════════════════════════════════════╗
║      AutoStream AI Assistant  (type 'quit' to exit)  ║
╚══════════════════════════════════════════════════════╝
"""


def run_cli() -> None:
    print(BANNER)
    state: AgentState | None = None

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye"}:
            print("Agent: Thanks for chatting with AutoStream! Have a great day. 👋")
            sys.exit(0)

        try:
            reply, state = chat(user_input, state)
            print(f"\nAgent: {reply}\n")
            if state.get("lead_captured"):
                print(f"  ✅  Lead {state.get('lead_id')} captured and saved to leads.json\n")
        except EnvironmentError as exc:
            print(f"\n[Configuration Error]\n{exc}\n")
            sys.exit(1)
        except RuntimeError as exc:
            print(f"\n[Agent Error] {exc}\n")
        except Exception as exc:
            print(f"\n[Error] {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    run_cli()
