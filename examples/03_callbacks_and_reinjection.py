"""Callbacks: on_before_compress, on_after_compress, on_usage_update.

Shows:
- on_before_compress: notification before messages are summarized
- on_after_compress: re-inject critical instructions after compression
- on_usage_update: real-time usage tracking
- Focus-based compaction via request_compact(focus=...)

Run:
    uv run python examples/03_callbacks_and_reinjection.py
"""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_middleware import MiddlewareAgent

from pydantic_ai_summarization import create_context_manager_middleware

MODEL = "openai:gpt-4.1-mini"
MAX_TOKENS = 5_000

# Critical instructions that must survive compression
# (like Claude Code's SessionStart hook with compact matcher)
CRITICAL_INSTRUCTIONS = (
    "IMPORTANT RULES (re-injected after compression):\n"
    "- Always respond in English\n"
    "- Never use markdown formatting\n"
    "- Keep responses under 2 sentences"
)


def on_usage(pct: float, current: int, maximum: int) -> None:
    filled = int(20 * pct)
    bar = "█" * filled + "░" * (20 - filled)
    color = "🔴" if pct > 0.8 else "🟡" if pct > 0.5 else "🟢"
    print(f"  {color} [{bar}] {pct:.0%}  ({current:,} / {maximum:,})")


def on_before_compress(messages: list[ModelMessage], cutoff_index: int) -> None:
    """Called BEFORE compression — messages are about to be summarized."""
    print("\n  ⚡ COMPRESSION STARTING")
    print(f"     Summarizing {cutoff_index} messages (out of {len(messages)} total)")
    print("     These messages will be replaced by a summary.")


def on_after_compress(messages: list[ModelMessage]) -> str | None:
    """Called AFTER compression — return a string to re-inject into context.

    This is how you ensure critical instructions survive compression.
    The returned string becomes a SystemPromptPart inserted after the summary.
    """
    print(f"  ✅ COMPRESSION DONE — {len(messages)} messages remain")
    print("     Re-injecting critical instructions...")
    return CRITICAL_INSTRUCTIONS


async def main() -> None:
    middleware = create_context_manager_middleware(
        max_tokens=MAX_TOKENS,
        compress_threshold=0.7,  # lower threshold for quicker demo
        summarization_model=MODEL,
        on_usage_update=on_usage,
        on_before_compress=on_before_compress,
        on_after_compress=on_after_compress,
    )

    agent = Agent(
        MODEL,
        instructions=(
            "You are a programming tutor. Keep responses short.\n\n" + CRITICAL_INSTRUCTIONS
        ),
        history_processors=[middleware],
    )
    wrapped = MiddlewareAgent(agent, middleware=[middleware])

    history: list[ModelMessage] = []
    prompts = [
        "Explain variables in Python.",
        "Explain functions.",
        "Explain classes.",
        "Explain decorators.",
        # This should trigger compression. After it, the agent should still
        # follow the re-injected rules (English, no markdown, short)
        "What topics have we covered so far? Do you remember the rules?",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"Turn {i}: {prompt}")
        print(f"{'=' * 60}")

        result = await wrapped.run(prompt, message_history=history)
        history = result.all_messages()

        print(f"\n  Assistant: {result.output}")

    # --- Manual compaction with focus ---
    print(f"\n{'=' * 60}")
    print("MANUAL COMPACTION with focus")
    print("=" * 60)

    print("Requesting focused compaction on 'decorators'...")
    middleware.request_compact(focus="decorators and how they work")

    result = await wrapped.run(
        "What do you know about decorators?",
        message_history=history,
    )
    history = result.all_messages()
    print(f"\n  Assistant: {result.output}")
    print(f"\nTotal compressions: {middleware.compression_count}")


if __name__ == "__main__":
    asyncio.run(main())
