"""Message persistence and session resume example.

Shows:
- Continuous message saving to messages.json
- Session resume from the same file
- How compression summaries are also persisted

Run:
    uv run python examples/02_persistence_and_resume.py
"""

import asyncio
import tempfile
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_middleware import MiddlewareAgent

from pydantic_ai_summarization import create_context_manager_middleware

MODEL = "openai:gpt-4.1-mini"
MAX_TOKENS = 5_000


def on_usage(pct: float, current: int, maximum: int) -> None:
    filled = int(20 * pct)
    print(f"  [{'█' * filled}{'░' * (20 - filled)}] {pct:.0%}")


async def run_session(messages_path: str, history: list[ModelMessage], prompts: list[str]) -> list[ModelMessage]:
    """Run a series of prompts with persistence."""
    middleware = create_context_manager_middleware(
        max_tokens=MAX_TOKENS,
        compress_threshold=0.8,
        summarization_model=MODEL,
        messages_path=messages_path,
        on_usage_update=on_usage,
    )

    agent = Agent(
        MODEL,
        instructions="Keep responses to 1 sentence.",
        history_processors=[middleware],
    )
    wrapped = MiddlewareAgent(agent, middleware=[middleware])

    for prompt in prompts:
        print(f"\n> {prompt}")
        result = await wrapped.run(prompt, message_history=history)
        history = result.all_messages()
        print(f"  Assistant: {result.output}")

    return history


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        messages_path = str(Path(tmpdir) / "messages.json")

        # --- Session 1: Start a conversation ---
        print("=" * 60)
        print("SESSION 1 — Starting fresh")
        print("=" * 60)

        history = await run_session(messages_path, [], [
            "What is Python?",
            "What is Rust?",
            "What is Go?",
        ])

        # Check what's on disk
        raw = Path(messages_path).read_bytes()
        saved = list(ModelMessagesTypeAdapter.validate_json(raw))
        print(f"\nSession 1 done. Messages on disk: {len(saved)}")
        print(f"Messages in context: {len(history)}")

        # --- Session 2: Resume from messages.json ---
        print(f"\n{'=' * 60}")
        print("SESSION 2 — Resuming from messages.json")
        print("=" * 60)

        # Load history from file (this is what --resume does)
        loaded = list(ModelMessagesTypeAdapter.validate_json(Path(messages_path).read_bytes()))
        print(f"Loaded {len(loaded)} messages from disk")

        history = await run_session(messages_path, loaded, [
            "What is JavaScript?",
            "Now list ALL languages we discussed.",
        ])

        # Final state
        raw2 = Path(messages_path).read_bytes()
        saved2 = list(ModelMessagesTypeAdapter.validate_json(raw2))
        print(f"\nSession 2 done. Messages on disk: {len(saved2)}")
        print(f"Messages in context: {len(history)}")

        # Show the file size
        size_kb = Path(messages_path).stat().st_size / 1024
        print(f"File size: {size_kb:.1f} KB")


if __name__ == "__main__":
    asyncio.run(main())
