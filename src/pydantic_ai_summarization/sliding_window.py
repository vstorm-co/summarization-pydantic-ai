"""Sliding window history processor for managing conversation context.

This module provides a simple, zero-cost strategy for managing context window limits
by keeping only the most recent messages without LLM-based summarization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.messages import ModelMessage

from pydantic_ai_summarization._cutoff import (
    determine_cutoff_index as _determine_cutoff,
)
from pydantic_ai_summarization._cutoff import (
    find_safe_cutoff as _find_safe,
)
from pydantic_ai_summarization._cutoff import (
    find_token_based_cutoff as _find_token,
)
from pydantic_ai_summarization._cutoff import (
    is_safe_cutoff_point as _is_safe,
)
from pydantic_ai_summarization._cutoff import (
    should_trigger as _should_trigger,
)
from pydantic_ai_summarization._cutoff import (
    validate_context_size as _validate_ctx,
)
from pydantic_ai_summarization._cutoff import (
    validate_triggers_and_keep as _validate_trig_keep,
)
from pydantic_ai_summarization.processor import count_tokens_approximately
from pydantic_ai_summarization.types import ContextSize, TokenCounter

_DEFAULT_WINDOW_SIZE = 50
_DEFAULT_TRIGGER_MESSAGES = 100


@dataclass
class SlidingWindowProcessor:
    """History processor that keeps only the most recent messages.

    This is the simplest and most efficient strategy for managing context limits.
    It has zero LLM cost and near-zero latency, making it ideal for scenarios
    where preserving exact conversation history is less important than performance.

    Unlike SummarizationProcessor, this processor simply discards old messages
    without creating a summary. This means some context may be lost, but the
    operation is instantaneous and free.

    Attributes:
        trigger: Threshold(s) that trigger window trimming.
        keep: How many messages to keep after trimming.
        token_counter: Function to count tokens (only used for token-based triggers/keep).
        max_input_tokens: Maximum input tokens (required for fraction-based config).

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization import SlidingWindowProcessor

        # Keep last 50 messages, trim when reaching 100
        processor = SlidingWindowProcessor(
            trigger=("messages", 100),
            keep=("messages", 50),
        )

        agent = Agent(
            "openai:gpt-4.1",
            history_processors=[processor],
        )
        ```
    """

    trigger: ContextSize | list[ContextSize] | None = None
    """Threshold(s) that trigger window trimming.

    Examples:
        - ("messages", 100) - trigger when 100+ messages
        - ("tokens", 100000) - trigger when 100k+ tokens
        - ("fraction", 0.8) - trigger at 80% of max tokens (requires max_input_tokens)
        - [("messages", 100), ("tokens", 50000)] - trigger on either condition
    """

    keep: ContextSize = ("messages", _DEFAULT_WINDOW_SIZE)
    """How many messages to keep after trimming.

    Examples:
        - ("messages", 50) - keep last 50 messages
        - ("tokens", 10000) - keep last 10k tokens worth
        - ("fraction", 0.3) - keep last 30% of max_input_tokens
    """

    token_counter: TokenCounter = field(default=count_tokens_approximately)
    """Function to count tokens in messages. Only used for token-based triggers/keep."""

    max_input_tokens: int | None = None
    """Maximum input tokens for the model (required for fraction-based triggers)."""

    _trigger_conditions: list[ContextSize] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate configuration and set up trigger conditions."""
        self._trigger_conditions, self.keep = _validate_trig_keep(
            self.trigger, self.keep, self.max_input_tokens
        )

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        return _validate_ctx(context, parameter_name)

    def _should_trim(self, messages: list[ModelMessage], total_tokens: int) -> bool:
        """Determine whether window trimming should occur."""
        return _should_trigger(
            self._trigger_conditions, messages, total_tokens, self.max_input_tokens
        )

    def _determine_cutoff_index(self, messages: list[ModelMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        return _determine_cutoff(
            messages,
            self.keep,
            self.token_counter,
            self.max_input_tokens,
            _DEFAULT_WINDOW_SIZE,
        )

    def _find_token_based_cutoff(
        self, messages: list[ModelMessage], target_token_count: int
    ) -> int:
        """Find cutoff index based on target token retention."""
        return _find_token(messages, target_token_count, self.token_counter)

    def _find_safe_cutoff(self, messages: list[ModelMessage], messages_to_keep: int) -> int:
        """Find safe cutoff point that preserves tool call/response pairs."""
        return _find_safe(messages, messages_to_keep)

    def _is_safe_cutoff_point(self, messages: list[ModelMessage], cutoff_index: int) -> bool:
        """Check if cutting at index would separate tool call/response pairs."""
        return _is_safe(messages, cutoff_index)

    async def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Process messages and trim if needed.

        This is the main entry point called by pydantic-ai's history processor mechanism.

        Args:
            messages: Current message history.

        Returns:
            Trimmed message history if threshold was reached, otherwise unchanged.
        """
        total_tokens = self.token_counter(messages)

        if not self._should_trim(messages, total_tokens):
            return messages

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return messages

        # Simply discard old messages and keep recent ones
        return messages[cutoff_index:]


def create_sliding_window_processor(
    trigger: ContextSize | list[ContextSize] | None = ("messages", _DEFAULT_TRIGGER_MESSAGES),
    keep: ContextSize = ("messages", _DEFAULT_WINDOW_SIZE),
    max_input_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
) -> SlidingWindowProcessor:
    """Create a sliding window history processor.

    This is a convenience factory function for creating SlidingWindowProcessor
    instances with sensible defaults.

    Args:
        trigger: When to trigger window trimming. Can be:
            - ("messages", N) - trigger when N+ messages
            - ("tokens", N) - trigger when N+ tokens
            - ("fraction", F) - trigger at F fraction of max_input_tokens
            - List of tuples to trigger on any condition
            Defaults to ("messages", 100).
        keep: How many messages to keep after trimming. Defaults to ("messages", 50).
        max_input_tokens: Maximum input tokens (required for fraction-based triggers).
        token_counter: Custom token counting function. Defaults to approximate counter.

    Returns:
        Configured SlidingWindowProcessor.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization import create_sliding_window_processor

        # Simple: keep last 30 messages when reaching 60
        processor = create_sliding_window_processor(
            trigger=("messages", 60),
            keep=("messages", 30),
        )

        # Token-based: keep ~50k tokens when reaching 100k
        processor = create_sliding_window_processor(
            trigger=("tokens", 100000),
            keep=("tokens", 50000),
        )

        agent = Agent(
            "openai:gpt-4.1",
            history_processors=[processor],
        )
        ```
    """
    kwargs: dict[str, ContextSize | list[ContextSize] | int | TokenCounter | None] = {
        "trigger": trigger,
        "keep": keep,
    }

    if max_input_tokens is not None:
        kwargs["max_input_tokens"] = max_input_tokens

    if token_counter is not None:
        kwargs["token_counter"] = token_counter

    return SlidingWindowProcessor(**kwargs)  # type: ignore[arg-type]


__all__ = [
    "SlidingWindowProcessor",
    "create_sliding_window_processor",
]
