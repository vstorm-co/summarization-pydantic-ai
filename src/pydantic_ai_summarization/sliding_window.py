"""Sliding window history processor for managing conversation context.

This module provides a simple, zero-cost strategy for managing context window limits
by keeping only the most recent messages without LLM-based summarization.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from pydantic_ai_summarization.processor import count_tokens_approximately
from pydantic_ai_summarization.types import ContextSize, TokenCounter

_DEFAULT_WINDOW_SIZE = 50
_DEFAULT_TRIGGER_MESSAGES = 100
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


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
        if self.trigger is None:
            self._trigger_conditions = []
        elif isinstance(self.trigger, list):
            self._trigger_conditions = [
                self._validate_context_size(t, "trigger") for t in self.trigger
            ]
        else:
            self._trigger_conditions = [self._validate_context_size(self.trigger, "trigger")]

        self.keep = self._validate_context_size(self.keep, "keep")

        # Validate that fraction-based config has max_input_tokens
        requires_max_tokens = any(t[0] == "fraction" for t in self._trigger_conditions)
        if self.keep[0] == "fraction":
            requires_max_tokens = True

        if requires_max_tokens and self.max_input_tokens is None:
            raise ValueError(
                "max_input_tokens is required when using fraction-based triggers or keep. "
                "Please provide the model's maximum input token limit."
            )

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        kind, value = context
        if kind == "fraction":
            if not 0 < value <= 1:
                raise ValueError(
                    f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
                )
        elif kind in {"tokens", "messages"}:
            if value <= 0:
                raise ValueError(
                    f"{parameter_name} thresholds must be greater than 0, got {value}."
                )
        else:
            raise ValueError(f"Unsupported context size type {kind} for {parameter_name}.")
        return context

    def _should_trim(self, messages: list[ModelMessage], total_tokens: int) -> bool:
        """Determine whether window trimming should occur."""
        if not self._trigger_conditions:
            return False

        for kind, value in self._trigger_conditions:
            if kind == "messages" and len(messages) >= value:
                return True
            if kind == "tokens" and total_tokens >= value:
                return True
            if kind == "fraction" and self.max_input_tokens:
                threshold = int(self.max_input_tokens * value)
                if total_tokens >= threshold:
                    return True
        return False

    def _determine_cutoff_index(self, messages: list[ModelMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        kind, value = self.keep

        if kind == "messages":
            return self._find_safe_cutoff(messages, int(value))
        elif kind == "tokens":
            return self._find_token_based_cutoff(messages, int(value))
        elif kind == "fraction" and self.max_input_tokens:
            target_tokens = int(self.max_input_tokens * value)
            return self._find_token_based_cutoff(messages, target_tokens)

        return self._find_safe_cutoff(messages, _DEFAULT_WINDOW_SIZE)  # pragma: no cover

    def _find_token_based_cutoff(
        self, messages: list[ModelMessage], target_token_count: int
    ) -> int:
        """Find cutoff index based on target token retention."""
        if not messages or self.token_counter(messages) <= target_token_count:
            return 0

        # Binary search for the cutoff point
        left, right = 0, len(messages)
        cutoff_candidate = len(messages)

        for _ in range(len(messages).bit_length() + 1):
            if left >= right:
                break

            mid = (left + right) // 2
            if self.token_counter(messages[mid:]) <= target_token_count:
                cutoff_candidate = mid
                right = mid
            else:
                left = mid + 1

        if cutoff_candidate >= len(messages):  # pragma: no cover
            cutoff_candidate = max(0, len(messages) - 1)

        # Find a safe cutoff point (not splitting tool call pairs)
        for i in range(cutoff_candidate, -1, -1):  # pragma: no branch
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0  # pragma: no cover

    def _find_safe_cutoff(self, messages: list[ModelMessage], messages_to_keep: int) -> int:
        """Find safe cutoff point that preserves tool call/response pairs."""
        if len(messages) <= messages_to_keep:
            return 0

        target_cutoff = len(messages) - messages_to_keep

        for i in range(target_cutoff, -1, -1):
            if self._is_safe_cutoff_point(messages, i):
                return i

        return 0  # pragma: no cover

    def _is_safe_cutoff_point(self, messages: list[ModelMessage], cutoff_index: int) -> bool:
        """Check if cutting at index would separate tool call/response pairs.

        This ensures we never discard a tool call without its response or vice versa,
        which would confuse the model.
        """
        if cutoff_index >= len(messages):
            return True

        search_start = max(0, cutoff_index - _SEARCH_RANGE_FOR_TOOL_PAIRS)
        search_end = min(len(messages), cutoff_index + _SEARCH_RANGE_FOR_TOOL_PAIRS)

        for i in range(search_start, search_end):
            msg = messages[i]
            if not isinstance(msg, ModelResponse):
                continue

            tool_call_ids: set[str] = set()
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_call_id:
                    tool_call_ids.add(part.tool_call_id)

            if not tool_call_ids:
                continue

            # Check if cutoff separates this tool call from its response
            for j in range(i + 1, len(messages)):
                check_msg = messages[j]
                if isinstance(check_msg, ModelRequest):
                    for request_part in check_msg.parts:
                        if (
                            isinstance(request_part, ToolReturnPart)
                            and request_part.tool_call_id in tool_call_ids
                        ):
                            tool_before_cutoff = i < cutoff_index
                            response_before_cutoff = j < cutoff_index
                            if tool_before_cutoff != response_before_cutoff:
                                return False

        return True

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
