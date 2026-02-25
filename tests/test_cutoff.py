"""Tests for the shared cutoff algorithms."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from pydantic_ai_summarization._cutoff import (
    SEARCH_RANGE_FOR_TOOL_PAIRS,
    async_count_tokens,
    async_determine_cutoff_index,
    async_find_token_based_cutoff,
    determine_cutoff_index,
    find_safe_cutoff,
    find_token_based_cutoff,
    is_safe_cutoff_point,
    should_trigger,
    validate_context_size,
    validate_triggers_and_keep,
)
from pydantic_ai_summarization.processor import count_tokens_approximately


def _make_messages(n: int) -> list[ModelMessage]:
    """Create n alternating user/assistant messages."""
    msgs: list[ModelMessage] = []
    for i in range(n):
        if i % 2 == 0:
            msgs.append(ModelRequest(parts=[UserPromptPart(content=f"msg-{i}")]))
        else:
            msgs.append(ModelResponse(parts=[TextPart(content=f"reply-{i}")]))
    return msgs


def _make_tool_pair(call_id: str) -> list[ModelMessage]:
    """Create a tool call + tool return pair."""
    return [
        ModelResponse(parts=[ToolCallPart(tool_name="test", args="{}", tool_call_id=call_id)]),
        ModelRequest(
            parts=[ToolReturnPart(tool_name="test", content="result", tool_call_id=call_id)]
        ),
    ]


class TestValidateContextSize:
    """Tests for validate_context_size."""

    def test_valid_fraction(self):
        result = validate_context_size(("fraction", 0.5), "trigger")
        assert result == ("fraction", 0.5)

    def test_valid_fraction_one(self):
        result = validate_context_size(("fraction", 1.0), "trigger")
        assert result == ("fraction", 1.0)

    def test_invalid_fraction_zero(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_context_size(("fraction", 0.0), "trigger")

    def test_invalid_fraction_above_one(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            validate_context_size(("fraction", 1.5), "trigger")

    def test_valid_tokens(self):
        result = validate_context_size(("tokens", 100), "keep")
        assert result == ("tokens", 100)

    def test_valid_tokens_zero(self):
        result = validate_context_size(("tokens", 0), "keep")
        assert result == ("tokens", 0)

    def test_invalid_tokens_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_context_size(("tokens", -1), "keep")

    def test_valid_messages(self):
        result = validate_context_size(("messages", 50), "keep")
        assert result == ("messages", 50)

    def test_valid_messages_zero(self):
        result = validate_context_size(("messages", 0), "keep")
        assert result == ("messages", 0)

    def test_invalid_messages_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_context_size(("messages", -1), "trigger")

    def test_unsupported_type(self):
        with pytest.raises(ValueError, match="Unsupported context size type"):
            validate_context_size(("invalid", 10), "trigger")  # type: ignore[arg-type]


class TestShouldTrigger:
    """Tests for should_trigger."""

    def test_empty_triggers(self):
        assert should_trigger([], _make_messages(10), 1000) is False

    def test_message_trigger_met(self):
        msgs = _make_messages(10)
        assert should_trigger([("messages", 10)], msgs, 0) is True

    def test_message_trigger_not_met(self):
        msgs = _make_messages(5)
        assert should_trigger([("messages", 10)], msgs, 0) is False

    def test_token_trigger_met(self):
        assert should_trigger([("tokens", 100)], _make_messages(2), 200) is True

    def test_token_trigger_not_met(self):
        assert should_trigger([("tokens", 100)], _make_messages(2), 50) is False

    def test_fraction_trigger_met(self):
        assert (
            should_trigger([("fraction", 0.5)], _make_messages(2), 60000, max_input_tokens=100000)
            is True
        )

    def test_fraction_trigger_not_met(self):
        assert (
            should_trigger([("fraction", 0.8)], _make_messages(2), 50000, max_input_tokens=100000)
            is False
        )

    def test_fraction_trigger_no_max_tokens(self):
        # fraction trigger without max_input_tokens is never met
        assert (
            should_trigger([("fraction", 0.5)], _make_messages(2), 60000, max_input_tokens=None)
            is False
        )

    def test_multiple_triggers_first_met(self):
        msgs = _make_messages(10)
        assert should_trigger([("messages", 10), ("tokens", 999999)], msgs, 0) is True

    def test_multiple_triggers_second_met(self):
        msgs = _make_messages(5)
        assert should_trigger([("messages", 999), ("tokens", 50)], msgs, 100) is True


class TestIsSafeCutoffPoint:
    """Tests for is_safe_cutoff_point."""

    def test_beyond_messages(self):
        msgs = _make_messages(4)
        assert is_safe_cutoff_point(msgs, 10) is True

    def test_no_tool_calls(self):
        msgs = _make_messages(6)
        assert is_safe_cutoff_point(msgs, 3) is True

    def test_safe_cutoff_after_tool_pair(self):
        msgs = [
            ModelRequest(parts=[UserPromptPart(content="q")]),
            *_make_tool_pair("call-1"),
            ModelResponse(parts=[TextPart(content="answer")]),
        ]
        # Cutoff at 3 is after the tool pair (both at indices 1,2)
        assert is_safe_cutoff_point(msgs, 3) is True

    def test_unsafe_cutoff_splitting_tool_pair(self):
        msgs = [
            ModelRequest(parts=[UserPromptPart(content="q")]),
            *_make_tool_pair("call-1"),
            ModelResponse(parts=[TextPart(content="answer")]),
        ]
        # Cutoff at 2 splits the tool call (1) from its return (2)
        assert is_safe_cutoff_point(msgs, 2) is False

    def test_non_response_messages_ignored(self):
        msgs = _make_messages(6)
        assert is_safe_cutoff_point(msgs, 2) is True

    def test_tool_call_without_id(self):
        msgs = [
            ModelRequest(parts=[UserPromptPart(content="q")]),
            ModelResponse(parts=[ToolCallPart(tool_name="t", args="{}")]),
            ModelRequest(parts=[UserPromptPart(content="q2")]),
        ]
        # No tool_call_id, so no pair to split
        assert is_safe_cutoff_point(msgs, 1) is True

    def test_search_range_constant(self):
        assert SEARCH_RANGE_FOR_TOOL_PAIRS == 5


class TestFindSafeCutoff:
    """Tests for find_safe_cutoff."""

    def test_few_messages(self):
        msgs = _make_messages(3)
        assert find_safe_cutoff(msgs, 5) == 0

    def test_normal_cutoff(self):
        msgs = _make_messages(10)
        assert find_safe_cutoff(msgs, 4) > 0

    def test_with_tool_pair(self):
        msgs = [
            *_make_messages(4),
            *_make_tool_pair("call-1"),
            *_make_messages(4),
        ]
        cutoff = find_safe_cutoff(msgs, 4)
        assert cutoff > 0
        assert is_safe_cutoff_point(msgs, cutoff)


class TestFindTokenBasedCutoff:
    """Tests for find_token_based_cutoff."""

    def test_empty_messages(self):
        assert find_token_based_cutoff([], 100, count_tokens_approximately) == 0

    def test_below_limit(self):
        msgs = _make_messages(4)
        assert find_token_based_cutoff(msgs, 999999, count_tokens_approximately) == 0

    def test_needs_cut(self):
        msgs = _make_messages(20)
        cutoff = find_token_based_cutoff(msgs, 10, count_tokens_approximately)
        assert cutoff > 0

    def test_with_tool_pairs(self):
        msgs = [
            *_make_messages(10),
            *_make_tool_pair("call-1"),
            *_make_messages(4),
        ]
        cutoff = find_token_based_cutoff(msgs, 20, count_tokens_approximately)
        assert cutoff >= 0
        if cutoff > 0:
            assert is_safe_cutoff_point(msgs, cutoff)


class TestDetermineCutoffIndex:
    """Tests for determine_cutoff_index."""

    def test_message_based(self):
        msgs = _make_messages(10)
        cutoff = determine_cutoff_index(msgs, ("messages", 4), count_tokens_approximately)
        assert cutoff > 0

    def test_token_based(self):
        msgs = _make_messages(20)
        cutoff = determine_cutoff_index(msgs, ("tokens", 10), count_tokens_approximately)
        assert cutoff > 0

    def test_fraction_based(self):
        msgs = _make_messages(20)
        cutoff = determine_cutoff_index(
            msgs,
            ("fraction", 0.1),
            count_tokens_approximately,
            max_input_tokens=100,
        )
        assert cutoff > 0

    def test_few_messages(self):
        msgs = _make_messages(3)
        cutoff = determine_cutoff_index(msgs, ("messages", 10), count_tokens_approximately)
        assert cutoff == 0


class TestValidateTriggersAndKeep:
    """Tests for validate_triggers_and_keep."""

    def test_none_trigger(self):
        triggers, keep = validate_triggers_and_keep(None, ("messages", 20), None)
        assert triggers == []
        assert keep == ("messages", 20)

    def test_single_trigger(self):
        triggers, keep = validate_triggers_and_keep(("tokens", 100), ("messages", 10), None)
        assert triggers == [("tokens", 100)]

    def test_list_triggers(self):
        triggers, keep = validate_triggers_and_keep(
            [("tokens", 100), ("messages", 50)], ("messages", 10), None
        )
        assert len(triggers) == 2

    def test_fraction_trigger_requires_max_input_tokens(self):
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            validate_triggers_and_keep(("fraction", 0.8), ("messages", 10), None)

    def test_fraction_keep_requires_max_input_tokens(self):
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            validate_triggers_and_keep(("tokens", 100), ("fraction", 0.5), None)

    def test_fraction_with_max_input_tokens(self):
        triggers, keep = validate_triggers_and_keep(
            ("fraction", 0.8), ("messages", 10), max_input_tokens=200000
        )
        assert triggers == [("fraction", 0.8)]

    def test_valid_trigger_zero(self):
        triggers, keep = validate_triggers_and_keep(("tokens", 0), ("messages", 10), None)
        assert triggers == [("tokens", 0)]

    def test_invalid_trigger_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_triggers_and_keep(("tokens", -1), ("messages", 10), None)

    def test_valid_keep_zero(self):
        triggers, keep = validate_triggers_and_keep(None, ("messages", 0), None)
        assert keep == ("messages", 0)

    def test_invalid_keep_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            validate_triggers_and_keep(None, ("messages", -1), None)


class TestAsyncCountTokens:
    """Tests for async_count_tokens with sync and async counters."""

    async def test_sync_counter(self):
        """Sync counter result is returned directly."""
        msgs = _make_messages(4)
        result = await async_count_tokens(count_tokens_approximately, msgs)
        assert result == count_tokens_approximately(msgs)

    async def test_async_counter(self):
        """Async counter result is awaited."""

        async def async_counter(messages: Sequence[ModelMessage]) -> int:
            return len(messages) * 10

        msgs = _make_messages(4)
        result = await async_count_tokens(async_counter, msgs)
        assert result == 40


class TestFindSafeCutoffZeroKeep:
    """Tests for find_safe_cutoff with messages_to_keep == 0."""

    def test_zero_keep_returns_length(self):
        """messages_to_keep=0 means summarize everything."""
        msgs = _make_messages(5)
        assert find_safe_cutoff(msgs, 0) == 5

    def test_zero_keep_empty_messages(self):
        """messages_to_keep=0 with empty list returns 0."""
        assert find_safe_cutoff([], 0) == 0


class TestAsyncDetermineCutoffIndex:
    """Tests for async_determine_cutoff_index."""

    async def test_message_based_keep(self):
        """Message-based keep delegates to find_safe_cutoff."""
        msgs = _make_messages(10)
        cutoff = await async_determine_cutoff_index(
            msgs, ("messages", 4), count_tokens_approximately
        )
        assert cutoff > 0

    async def test_token_based_keep(self):
        """Token-based keep delegates to async_find_token_based_cutoff."""
        msgs = _make_messages(20)
        cutoff = await async_determine_cutoff_index(
            msgs, ("tokens", 10), count_tokens_approximately
        )
        assert cutoff > 0

    async def test_fraction_based_keep(self):
        """Fraction-based keep computes target tokens and delegates."""
        msgs = _make_messages(20)
        cutoff = await async_determine_cutoff_index(
            msgs,
            ("fraction", 0.1),
            count_tokens_approximately,
            max_input_tokens=100,
        )
        assert cutoff > 0

    async def test_few_messages_message_keep(self):
        """When messages < keep, cutoff is 0."""
        msgs = _make_messages(3)
        cutoff = await async_determine_cutoff_index(
            msgs, ("messages", 10), count_tokens_approximately
        )
        assert cutoff == 0

    async def test_with_async_token_counter(self):
        """Works with async token counter."""

        async def async_counter(messages: Sequence[ModelMessage]) -> int:
            return len(messages) * 5

        msgs = _make_messages(20)
        cutoff = await async_determine_cutoff_index(
            msgs, ("tokens", 10), async_counter
        )
        assert cutoff > 0


class TestAsyncFindTokenBasedCutoff:
    """Tests for async_find_token_based_cutoff."""

    async def test_empty_messages(self):
        """Empty messages returns 0."""
        result = await async_find_token_based_cutoff([], 100, count_tokens_approximately)
        assert result == 0

    async def test_below_limit(self):
        """Messages below token limit returns 0."""
        msgs = _make_messages(4)
        result = await async_find_token_based_cutoff(msgs, 999999, count_tokens_approximately)
        assert result == 0

    async def test_needs_cut(self):
        """Messages above token limit get cut."""
        msgs = _make_messages(20)
        cutoff = await async_find_token_based_cutoff(msgs, 10, count_tokens_approximately)
        assert cutoff > 0

    async def test_with_tool_pairs(self):
        """Preserves tool call/response pairs."""
        msgs = [
            *_make_messages(10),
            *_make_tool_pair("call-1"),
            *_make_messages(4),
        ]
        cutoff = await async_find_token_based_cutoff(msgs, 20, count_tokens_approximately)
        assert cutoff >= 0
        if cutoff > 0:
            assert is_safe_cutoff_point(msgs, cutoff)

    async def test_with_async_counter(self):
        """Works with an async token counter."""

        async def async_counter(messages: Sequence[ModelMessage]) -> int:
            return len(messages) * 5

        msgs = _make_messages(20)
        cutoff = await async_find_token_based_cutoff(msgs, 10, async_counter)
        assert cutoff > 0

    async def test_with_async_counter_below_limit(self):
        """Async counter with messages below limit returns 0."""

        async def async_counter(messages: Sequence[ModelMessage]) -> int:
            return len(messages) * 1

        msgs = _make_messages(4)
        result = await async_find_token_based_cutoff(msgs, 999999, async_counter)
        assert result == 0
