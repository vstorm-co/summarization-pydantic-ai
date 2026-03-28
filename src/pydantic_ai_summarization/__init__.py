"""summarization-pydantic-ai — Context management for Pydantic AI agents.

Automatic conversation summarization, sliding window trimming, limit warnings,
and full context management via pydantic-ai capabilities.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_summarization import ContextManagerCapability

    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[ContextManagerCapability(max_tokens=100_000)],
    )
    ```
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pydantic_ai_summarization._cutoff import async_count_tokens
from pydantic_ai_summarization.capability import (
    ContextManagerCapability,
    LimitWarnerCapability,
    SlidingWindowCapability,
    SummarizationCapability,
)
from pydantic_ai_summarization.limit_warner import (
    LimitWarnerProcessor,
    create_limit_warner_processor,
)
from pydantic_ai_summarization.processor import (
    DEFAULT_CONTINUATION_PROMPT,
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
    ModelType,
    TokenCounter,
    WarningOn,
)

try:
    __version__ = version("summarization-pydantic-ai")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    # Capabilities (recommended)
    "SummarizationCapability",
    "SlidingWindowCapability",
    "LimitWarnerCapability",
    "ContextManagerCapability",
    # Processors (standalone)
    "SummarizationProcessor",
    "create_summarization_processor",
    "SlidingWindowProcessor",
    "create_sliding_window_processor",
    "LimitWarnerProcessor",
    "create_limit_warner_processor",
    # Utilities
    "count_tokens_approximately",
    "format_messages_for_summary",
    "async_count_tokens",
    # Types
    "ContextSize",
    "ContextFraction",
    "ContextTokens",
    "ContextMessages",
    "WarningOn",
    "ModelType",
    "TokenCounter",
    # Constants
    "DEFAULT_SUMMARY_PROMPT",
    "DEFAULT_CONTINUATION_PROMPT",
    # Version
    "__version__",
]
