"""Shared cutoff and safety algorithms for history processors.

This internal module contains standalone functions for:
- Context size validation
- Trigger condition evaluation
- Safe cutoff point detection (preserving tool call/response pairs)
- Binary search-based token cutoff
- Trigger and keep configuration normalization
"""

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
)

from pydantic_ai_summarization.types import ContextSize, TokenCounter


async def async_count_tokens(token_counter: TokenCounter, messages: Sequence[ModelMessage]) -> int:
    """Call a token counter, awaiting if it returns an awaitable.

    Args:
        token_counter: Sync or async token counting function.
        messages: Messages to count tokens for.

    Returns:
        Token count.
    """
    result = token_counter(messages)
    if inspect.isawaitable(result):
        return await result
    return result


SEARCH_RANGE_FOR_TOOL_PAIRS: int = 5
"""Number of messages to search around cutoff point for tool call/response pairs."""


def validate_context_size(context: ContextSize, parameter_name: str) -> ContextSize:
    """Validate a context size configuration tuple.

    Args:
        context: A (kind, value) tuple specifying the context size.
        parameter_name: Name of the parameter (for error messages).

    Returns:
        The validated context tuple (unchanged).

    Raises:
        ValueError: If the context size is invalid.
    """
    kind, value = context
    if kind == "fraction":
        if not 0 < value <= 1:
            raise ValueError(
                f"Fractional {parameter_name} values must be between 0 and 1, got {value}."
            )
    elif kind in {"tokens", "messages"}:
        if value < 0:
            raise ValueError(f"{parameter_name} thresholds must be non-negative, got {value}.")
    else:
        raise ValueError(f"Unsupported context size type {kind} for {parameter_name}.")
    return context


def should_trigger(
    trigger_conditions: list[ContextSize],
    messages: Sequence[ModelMessage],
    total_tokens: int,
    max_input_tokens: int | None = None,
) -> bool:
    """Determine whether a context management action should run.

    Uses OR logic: returns True if ANY trigger condition is met.

    Args:
        trigger_conditions: List of trigger conditions to check.
        messages: Current message history.
        total_tokens: Current total token count.
        max_input_tokens: Max input tokens (for fraction-based triggers).

    Returns:
        True if at least one trigger condition is met.
    """
    if not trigger_conditions:
        return False

    for kind, value in trigger_conditions:
        if kind == "messages" and len(messages) >= value:
            return True
        if kind == "tokens" and total_tokens >= value:
            return True
        if kind == "fraction" and max_input_tokens:
            threshold = int(max_input_tokens * value)
            if total_tokens >= threshold:
                return True
    return False


def determine_cutoff_index(
    messages: list[ModelMessage],
    keep: ContextSize,
    token_counter: TokenCounter,
    max_input_tokens: int | None = None,
    default_keep: int = 20,
) -> int:
    """Choose cutoff index respecting retention configuration.

    Args:
        messages: Current message history.
        keep: How much context to retain after cutting.
        token_counter: Function to count tokens.
        max_input_tokens: Max input tokens (for fraction-based keep).
        default_keep: Fallback number of messages to keep.

    Returns:
        Index at which to cut the message history.
    """
    kind, value = keep

    if kind == "messages":
        return find_safe_cutoff(messages, int(value))
    elif kind == "tokens":
        return find_token_based_cutoff(messages, int(value), token_counter)
    elif kind == "fraction" and max_input_tokens:
        target_tokens = int(max_input_tokens * value)
        return find_token_based_cutoff(messages, target_tokens, token_counter)

    return find_safe_cutoff(messages, default_keep)  # pragma: no cover


def find_token_based_cutoff(
    messages: list[ModelMessage],
    target_token_count: int,
    token_counter: TokenCounter,
) -> int:
    """Find cutoff index based on target token retention using binary search.

    Args:
        messages: Current message history.
        target_token_count: Target number of tokens to retain.
        token_counter: Function to count tokens.

    Returns:
        Index at which to cut, preserving tool call/response pairs.
    """
    if not messages or cast(int, token_counter(messages)) <= target_token_count:
        return 0

    # Binary search for the cutoff point
    left, right = 0, len(messages)
    cutoff_candidate = len(messages)

    for _ in range(len(messages).bit_length() + 1):
        if left >= right:
            break

        mid = (left + right) // 2
        if cast(int, token_counter(messages[mid:])) <= target_token_count:
            cutoff_candidate = mid
            right = mid
        else:
            left = mid + 1

    if cutoff_candidate >= len(messages):  # pragma: no cover
        cutoff_candidate = max(0, len(messages) - 1)

    # Find a safe cutoff point (not splitting tool call pairs)
    for i in range(cutoff_candidate, -1, -1):  # pragma: no branch
        if is_safe_cutoff_point(messages, i):
            return i

    return 0  # pragma: no cover


def find_safe_cutoff(messages: list[ModelMessage], messages_to_keep: int) -> int:
    """Find safe cutoff point that preserves tool call/response pairs.

    Args:
        messages: Current message history.
        messages_to_keep: Number of messages to keep from the end.
            Use 0 to summarize everything (only summary survives).

    Returns:
        Index at which to cut, preserving tool call/response pairs.
    """
    if messages_to_keep == 0:
        return len(messages)

    if len(messages) <= messages_to_keep:
        return 0

    target_cutoff = len(messages) - messages_to_keep

    for i in range(target_cutoff, -1, -1):
        if is_safe_cutoff_point(messages, i):
            return i

    return 0  # pragma: no cover


def is_safe_cutoff_point(
    messages: list[ModelMessage],
    cutoff_index: int,
    search_range: int = SEARCH_RANGE_FOR_TOOL_PAIRS,
) -> bool:
    """Check if cutting at index would separate tool call/response pairs.

    Searches within +-search_range messages around the cutoff point to find
    tool call/response pairs. Returns False if the cutoff would split any pair.

    Args:
        messages: Current message history.
        cutoff_index: Proposed cutoff index.
        search_range: Number of messages to search around cutoff.

    Returns:
        True if the cutoff point is safe (doesn't split tool pairs).
    """
    if cutoff_index >= len(messages):
        return True

    search_start = max(0, cutoff_index - search_range)
    search_end = min(len(messages), cutoff_index + search_range)

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


def validate_triggers_and_keep(
    trigger: ContextSize | list[ContextSize] | None,
    keep: ContextSize,
    max_input_tokens: int | None,
) -> tuple[list[ContextSize], ContextSize]:
    """Validate and normalize trigger conditions and keep configuration.

    Args:
        trigger: Trigger condition(s) or None.
        keep: Retention configuration.
        max_input_tokens: Max input tokens (required for fraction-based configs).

    Returns:
        Tuple of (normalized trigger conditions list, validated keep).

    Raises:
        ValueError: If configuration is invalid.
    """
    # Normalize trigger to a list
    if trigger is None:
        trigger_conditions: list[ContextSize] = []
    elif isinstance(trigger, list):
        trigger_conditions = trigger
    else:
        trigger_conditions = [trigger]

    # Validate each trigger
    for t in trigger_conditions:
        validate_context_size(t, "trigger")

    # Validate keep
    validated_keep = validate_context_size(keep, "keep")

    # Check fraction requirements
    uses_fraction = any(t[0] == "fraction" for t in trigger_conditions) or keep[0] == "fraction"
    if uses_fraction and max_input_tokens is None:
        raise ValueError(
            "max_input_tokens is required when using fraction-based trigger or keep values."
        )

    return trigger_conditions, validated_keep


# -- Async variants for ContextManagerMiddleware --


async def async_determine_cutoff_index(
    messages: list[ModelMessage],
    keep: ContextSize,
    token_counter: TokenCounter,
    max_input_tokens: int | None = None,
    default_keep: int = 0,
) -> int:
    """Async variant of :func:`determine_cutoff_index`.

    Supports both sync and async token counters.
    """
    kind, value = keep

    if kind == "messages":
        return find_safe_cutoff(messages, int(value))
    elif kind == "tokens":
        return await async_find_token_based_cutoff(messages, int(value), token_counter)
    elif kind == "fraction" and max_input_tokens:
        target_tokens = int(max_input_tokens * value)
        return await async_find_token_based_cutoff(messages, target_tokens, token_counter)

    return find_safe_cutoff(messages, default_keep)  # pragma: no cover


async def async_find_token_based_cutoff(
    messages: list[ModelMessage],
    target_token_count: int,
    token_counter: TokenCounter,
) -> int:
    """Async variant of :func:`find_token_based_cutoff`.

    Supports both sync and async token counters.
    """
    if not messages or await async_count_tokens(token_counter, messages) <= target_token_count:
        return 0

    left, right = 0, len(messages)
    cutoff_candidate = len(messages)

    for _ in range(len(messages).bit_length() + 1):
        if left >= right:
            break

        mid = (left + right) // 2
        if await async_count_tokens(token_counter, messages[mid:]) <= target_token_count:
            cutoff_candidate = mid
            right = mid
        else:
            left = mid + 1

    if cutoff_candidate >= len(messages):  # pragma: no cover
        cutoff_candidate = max(0, len(messages) - 1)

    for i in range(cutoff_candidate, -1, -1):  # pragma: no branch
        if is_safe_cutoff_point(messages, i):
            return i

    return 0  # pragma: no cover
