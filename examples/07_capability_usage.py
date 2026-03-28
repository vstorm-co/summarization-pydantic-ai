"""Using capabilities — the recommended approach (no middleware needed)."""

import asyncio

from pydantic_ai import Agent

from pydantic_ai_summarization import (
    ContextManagerCapability,
    LimitWarnerCapability,
    SlidingWindowCapability,
    SummarizationCapability,
)


async def main() -> None:
    # === ContextManagerCapability (recommended for production) ===
    # Auto-detects max_tokens, compresses at 90%, truncates large tool outputs
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[ContextManagerCapability(max_tokens=100_000)],
    )

    result = await agent.run("Hello!")
    print(result.output)

    # === Combine with LimitWarner ===
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            LimitWarnerCapability(max_iterations=40, max_context_tokens=100_000),
            ContextManagerCapability(max_tokens=100_000),
        ],
    )

    result = await agent.run("What is Python?")
    print(result.output)

    # === SummarizationCapability (standalone) ===
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            SummarizationCapability(
                trigger=("messages", 50),
                keep=("messages", 10),
            )
        ],
    )

    # === SlidingWindowCapability (zero-cost) ===
    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[
            SlidingWindowCapability(
                trigger=("messages", 100),
                keep=("messages", 50),
            )
        ],
    )


if __name__ == "__main__":
    asyncio.run(main())
