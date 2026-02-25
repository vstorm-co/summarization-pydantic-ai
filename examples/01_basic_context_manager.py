"""Basic ContextManagerMiddleware example.

Shows: auto-compression when approaching token limit.
Uses max_tokens=500 so compression triggers after a few messages.

Run:
    uv run python examples/01_basic_context_manager.py
"""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage
from pydantic_ai_middleware import MiddlewareAgent

from pydantic_ai_summarization import create_context_manager_middleware

# Very low budget so compression triggers quickly
MAX_TOKENS = 500
MODEL = "openai:gpt-4.1-mini"


def on_usage_update(pct: float, current: int, maximum: int) -> None:
    bar_width = 30
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"  [{bar}] {pct:.0%}  ({current:,} / {maximum:,} tokens)")


async def main() -> None:
    middleware = create_context_manager_middleware(
        max_tokens=MAX_TOKENS,
        compress_threshold=0.7,  # compress at 70% (= 350 tokens)
        summarization_model=MODEL,
        on_usage_update=on_usage_update,
    )

    agent = Agent(
        MODEL,
        instructions="You are a helpful assistant. Answer in 3-5 sentences with details.",
        history_processors=[middleware],
    )
    wrapped = MiddlewareAgent(agent, middleware=[middleware])

    history: list[ModelMessage] = []
    prompts = [
        "What is Python and what are its main use cases?",
        "Explain how asyncio works in Python.",
        "What is pydantic and why is it useful?",
        "How does FastAPI use pydantic?",
        "What is Docker and how does containerization work?",
        "Now list ALL topics we discussed so far.",
    ]

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'=' * 60}")
        print(f"Turn {i}: {prompt}")
        print(f"{'=' * 60}")

        compressions_before = middleware.compression_count
        result = await wrapped.run(prompt, message_history=history)
        history = result.all_messages()

        if middleware.compression_count > compressions_before:
            print(f"\n  ** AUTO-COMPRESSED (compression #{middleware.compression_count}) **")

        print(f"\nAssistant: {result.output}")
        print(f"Messages in context: {len(history)}")
        print(f"Compressions so far: {middleware.compression_count}")

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"Total compressions: {middleware.compression_count}")
    print(f"Final messages in context: {len(history)}")


if __name__ == "__main__":
    asyncio.run(main())
