"""Interactive chat with context management.

A full interactive REPL that demonstrates:
- Real-time usage bar
- Auto-compression at threshold
- Manual /compact [focus] command
- /context command to show stats
- /history to show raw message count
- Session persistence to messages.json

Run:
    uv run python examples/05_interactive_chat.py
"""

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_middleware import MiddlewareAgent

from pydantic_ai_summarization import create_context_manager_middleware

MODEL = "openai:gpt-4.1-mini"
MAX_TOKENS = 5_000  # low budget for quick demo


class UsageTracker:
    """Tracks latest usage stats for display."""

    def __init__(self) -> None:
        self.pct = 0.0
        self.current = 0
        self.maximum = 0

    def update(self, pct: float, current: int, maximum: int) -> None:
        self.pct = pct
        self.current = current
        self.maximum = maximum


def print_usage_bar(tracker: UsageTracker) -> None:
    filled = int(20 * tracker.pct)
    bar = "█" * filled + "░" * (20 - filled)
    print(f"  [{bar}] {tracker.pct:.0%}  ({tracker.current:,} / {tracker.maximum:,} tokens)")


async def main() -> None:
    tmpdir = tempfile.mkdtemp()
    messages_path = str(Path(tmpdir) / "messages.json")

    tracker = UsageTracker()

    middleware = create_context_manager_middleware(
        max_tokens=MAX_TOKENS,
        compress_threshold=0.8,
        summarization_model=MODEL,
        messages_path=messages_path,
        on_usage_update=tracker.update,
    )

    agent = Agent(
        MODEL,
        instructions="You are a helpful assistant. Keep answers short (1-2 sentences).",
        history_processors=[middleware],
    )
    wrapped = MiddlewareAgent(agent, middleware=[middleware])

    history: list[ModelMessage] = []

    print("Interactive Chat (context managed)")
    print(f"  Model: {MODEL}")
    print(f"  Budget: {MAX_TOKENS:,} tokens")
    print("  Threshold: 80%")
    print(f"  Persistence: {messages_path}")
    print()
    print("Commands:")
    print("  /compact [focus]  — force compression (optional focus)")
    print("  /context          — show usage stats")
    print("  /quit             — exit")
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue

        if user_input == "/quit":
            print("Bye!")
            break

        if user_input == "/context":
            print_usage_bar(tracker)
            print(f"  Messages: {len(history)}")
            print(f"  Compressions: {middleware.compression_count}")
            p = Path(messages_path)
            if p.exists():
                print(f"  File: {messages_path} ({p.stat().st_size / 1024:.1f} KB)")
            continue

        if user_input.startswith("/compact"):
            focus = user_input[8:].strip() or None
            print(f"  Compacting{' (focus: ' + focus + ')' if focus else ''}...")
            history = await middleware.compact(history, focus=focus)
            print(f"  Done. Messages: {len(history)}")
            print_usage_bar(tracker)
            continue

        # Regular message
        compressions_before = middleware.compression_count
        result = await wrapped.run(user_input, message_history=history)
        history = result.all_messages()

        if middleware.compression_count > compressions_before:
            print(f"  [auto-compressed — compression #{middleware.compression_count}]")

        print(f"Assistant: {result.output}")
        print_usage_bar(tracker)


if __name__ == "__main__":
    asyncio.run(main())
