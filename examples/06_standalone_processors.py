"""Standalone processors: SummarizationProcessor and SlidingWindowProcessor.

Shows the two simpler alternatives that DON'T require pydantic-ai-middleware:
- SummarizationProcessor: LLM-based compression (trigger → summarize → keep N)
- SlidingWindowProcessor: Zero-cost trimming (just drops old messages)

These are pure history_processors — no middleware wrapping needed.

Run:
    uv run python examples/06_standalone_processors.py
"""

import asyncio

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from pydantic_ai_summarization import (
    count_tokens_approximately,
    create_sliding_window_processor,
    create_summarization_processor,
)

MODEL = "openai:gpt-4.1-mini"

PROMPTS = [
    "What is Python?",
    "What is Rust?",
    "What is Go?",
    "What is Java?",
    "What is C++?",
    "List ALL programming languages we discussed.",
]


async def demo_summarization() -> None:
    """SummarizationProcessor: LLM compresses old messages into a summary."""
    print("=" * 60)
    print("DEMO 1: SummarizationProcessor")
    print("  Trigger: 8 messages → summarize, keep last 2")
    print("=" * 60)

    processor = create_summarization_processor(
        model=MODEL,
        trigger=("messages", 8),  # trigger after 8 messages
        keep=("messages", 2),     # keep last 2 after summarization
    )

    agent = Agent(
        MODEL,
        instructions="Keep responses to 1 sentence.",
        history_processors=[processor],
    )

    history: list[ModelMessage] = []
    for prompt in PROMPTS:
        print(f"\n> {prompt}")
        result = await agent.run(prompt, message_history=history)
        history = result.all_messages()
        tokens = count_tokens_approximately(history)
        print(f"  Assistant: {result.output}")
        print(f"  Messages: {len(history)}, ~{tokens} tokens")


async def demo_sliding_window() -> None:
    """SlidingWindowProcessor: just drops old messages, no LLM cost."""
    print(f"\n\n{'=' * 60}")
    print("DEMO 2: SlidingWindowProcessor")
    print("  Trigger: 8 messages → trim to last 4 (no LLM cost)")
    print("=" * 60)

    processor = create_sliding_window_processor(
        trigger=("messages", 8),  # trigger after 8 messages
        keep=("messages", 4),     # keep last 4
    )

    agent = Agent(
        MODEL,
        instructions="Keep responses to 1 sentence.",
        history_processors=[processor],
    )

    history: list[ModelMessage] = []
    for prompt in PROMPTS:
        print(f"\n> {prompt}")
        result = await agent.run(prompt, message_history=history)
        history = result.all_messages()
        tokens = count_tokens_approximately(history)
        print(f"  Assistant: {result.output}")
        print(f"  Messages: {len(history)}, ~{tokens} tokens")


async def main() -> None:
    await demo_summarization()
    await demo_sliding_window()

    print(f"\n\n{'=' * 60}")
    print("COMPARISON:")
    print("  SummarizationProcessor — costs LLM tokens but preserves meaning")
    print("  SlidingWindowProcessor — free but loses old context completely")
    print("  ContextManagerMiddleware — best of both + persistence + callbacks")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
