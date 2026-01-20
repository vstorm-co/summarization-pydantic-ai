"""Automatic conversation summarization for pydantic-ai agents.

This package provides history processors for automatic conversation summarization,
helping manage context window limits in long-running agent conversations.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_summarization import create_summarization_processor

    # Create a processor that triggers at 100k tokens and keeps 20 messages
    processor = create_summarization_processor(
        trigger=("tokens", 100000),
        keep=("messages", 20),
    )

    agent = Agent(
        "openai:gpt-4.1",
        history_processors=[processor],
    )

    # The processor will automatically summarize older messages
    # when the conversation grows too long
    result = await agent.run("Hello!")
    ```
"""

from importlib.metadata import PackageNotFoundError, version

from pydantic_ai_summarization.processor import (
    DEFAULT_SUMMARY_PROMPT,
    SummarizationProcessor,
    count_tokens_approximately,
    create_summarization_processor,
    format_messages_for_summary,
)
from pydantic_ai_summarization.sliding_window import (
    SlidingWindowProcessor,
    create_sliding_window_processor,
)
from pydantic_ai_summarization.types import (
    ContextFraction,
    ContextMessages,
    ContextSize,
    ContextTokens,
    TokenCounter,
)

try:
    __version__ = version("summarization-pydantic-ai")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    # Main exports - Summarization
    "SummarizationProcessor",
    "create_summarization_processor",
    # Main exports - Sliding Window
    "SlidingWindowProcessor",
    "create_sliding_window_processor",
    # Utilities
    "count_tokens_approximately",
    "format_messages_for_summary",
    # Types
    "ContextSize",
    "ContextFraction",
    "ContextTokens",
    "ContextMessages",
    "TokenCounter",
    # Constants
    "DEFAULT_SUMMARY_PROMPT",
    # Version
    "__version__",
]
