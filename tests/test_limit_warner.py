"""Tests for LimitWarnerProcessor."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.usage import RunUsage

from pydantic_ai_summarization import (
    LimitWarnerProcessor,
    create_limit_warner_processor,
)

_WARNING_MARKER = "[LimitWarnerProcessor]"


def _make_ctx(
    *,
    requests: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> RunContext[Any]:
    """Create a RunContext for testing."""
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(
            requests=requests,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
        prompt="Current prompt",
    )


def _make_messages(
    *,
    trailing_parts: list[SystemPromptPart | UserPromptPart] | None = None,
) -> list[ModelMessage]:
    """Create a simple valid history ending with a ModelRequest."""
    last_parts = trailing_parts or [UserPromptPart(content="Current question")]
    return [
        ModelRequest(parts=[UserPromptPart(content="Earlier question")]),
        ModelResponse(parts=[TextPart(content="Earlier answer")]),
        ModelRequest(parts=last_parts),
    ]


def _last_request(messages: list[ModelMessage]) -> ModelRequest:
    """Get the trailing ModelRequest."""
    last_message = messages[-1]
    assert isinstance(last_message, ModelRequest)
    return last_message


def _generated_warning_parts(messages: list[ModelMessage]) -> list[SystemPromptPart]:
    """Return generated warning parts from the trailing request."""
    return [
        part
        for part in _last_request(messages).parts
        if isinstance(part, SystemPromptPart) and _WARNING_MARKER in part.content
    ]


class TestLimitWarnerProcessor:
    """Tests for LimitWarnerProcessor."""

    def test_create_with_defaults(self):
        """Test creating the processor with one configured limit."""
        processor = create_limit_warner_processor(max_iterations=20)
        assert processor.max_iterations == 20
        assert processor.max_context_tokens is None
        assert processor.max_total_tokens is None
        assert processor.warning_threshold == 0.7
        assert processor.critical_remaining_iterations == 3
        assert processor._warn_on == ("iterations",)

    def test_create_with_multiple_limits(self):
        """Test creating the processor with multiple configured limits."""
        processor = create_limit_warner_processor(
            max_iterations=10,
            max_context_tokens=500,
            max_total_tokens=1000,
        )
        assert processor._warn_on == ("iterations", "context_window", "total_tokens")

    def test_create_with_custom_warn_on(self):
        """Test explicit warn_on selection is preserved."""
        processor = LimitWarnerProcessor(
            max_iterations=20,
            max_total_tokens=500,
            warn_on=["total_tokens", "iterations"],
        )
        assert processor._warn_on == ("total_tokens", "iterations")

    def test_requires_at_least_one_limit(self):
        """Test that at least one limit must be configured."""
        with pytest.raises(ValueError, match="At least one max_\\* limit must be configured"):
            LimitWarnerProcessor()

    def test_rejects_empty_warn_on(self):
        """Test that an empty warn_on list is invalid."""
        with pytest.raises(ValueError, match="warn_on must not be empty"):
            LimitWarnerProcessor(max_iterations=10, warn_on=[])

    def test_rejects_unsupported_warn_on_value(self):
        """Test invalid warning names are rejected."""
        with pytest.raises(ValueError, match="Unsupported warn_on value"):
            LimitWarnerProcessor(
                max_iterations=10,
                warn_on=["iterations", "invalid"],  # type: ignore[list-item]
            )

    def test_warn_on_requires_matching_limit(self):
        """Test warn_on cannot enable an unconfigured limit."""
        with pytest.raises(ValueError, match="requires its corresponding max_\\* limit"):
            LimitWarnerProcessor(
                max_iterations=10,
                warn_on=["iterations", "context_window"],
            )

    def test_invalid_warning_threshold(self):
        """Test that invalid thresholds are rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            LimitWarnerProcessor(max_iterations=10, warning_threshold=1.2)

    def test_invalid_critical_remaining_iterations(self):
        """Test that negative critical_remaining_iterations is rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            LimitWarnerProcessor(max_iterations=10, critical_remaining_iterations=-1)

    def test_invalid_limit_values(self):
        """Test that max limits must be positive."""
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            LimitWarnerProcessor(max_iterations=0)

        with pytest.raises(ValueError, match="max_context_tokens must be positive"):
            LimitWarnerProcessor(max_context_tokens=-1)

        with pytest.raises(ValueError, match="max_total_tokens must be positive"):
            LimitWarnerProcessor(max_total_tokens=0)

    @pytest.mark.anyio
    async def test_call_below_threshold_returns_messages_unchanged(self):
        """Test no warning is injected below all thresholds."""
        processor = LimitWarnerProcessor(
            max_iterations=10,
            max_context_tokens=500,
            max_total_tokens=200,
            token_counter=lambda _messages: 100,
        )
        messages = _make_messages()
        ctx = _make_ctx(requests=5, input_tokens=50, output_tokens=20)

        result = await processor(ctx, messages)
        assert result == messages

    @pytest.mark.anyio
    async def test_call_injects_combined_critical_warning(self):
        """Test a combined warning is added when multiple limits trigger."""
        processor = LimitWarnerProcessor(
            max_iterations=10,
            max_context_tokens=100,
            max_total_tokens=100,
            token_counter=lambda _messages: 110,
        )
        messages = _make_messages()
        ctx = _make_ctx(requests=8, input_tokens=70, output_tokens=40)

        result = await processor(ctx, messages)
        warning_parts = _generated_warning_parts(result)
        assert len(warning_parts) == 1

        warning = warning_parts[0].content
        assert "CRITICAL: Configured run limits are approaching." in warning
        assert "Iterations: 8/10 requests used (80%); 2 remaining." in warning
        assert "Context window: 110/100 tokens used (110%); 0 remaining." in warning
        assert "Total tokens: 110/100 used (110%); 0 remaining." in warning

    @pytest.mark.anyio
    async def test_call_replaces_old_warning_and_preserves_other_system_parts(self):
        """Test repeated runs replace prior warnings instead of accumulating them."""
        processor = LimitWarnerProcessor(
            max_iterations=10,
            max_total_tokens=100,
        )
        messages = _make_messages(
            trailing_parts=[
                SystemPromptPart(content="Keep replies short."),
                UserPromptPart(content="Current question"),
            ]
        )

        first_result = await processor(
            _make_ctx(requests=8, input_tokens=45, output_tokens=35), messages
        )
        second_result = await processor(
            _make_ctx(requests=9, input_tokens=50, output_tokens=40),
            first_result,
        )

        last_request = _last_request(second_result)
        warning_parts = _generated_warning_parts(second_result)
        assert len(warning_parts) == 1
        assert len(last_request.parts) == 3
        assert any(
            isinstance(part, SystemPromptPart) and part.content == "Keep replies short."
            for part in last_request.parts
        )
        assert "Iterations: 9/10 requests used (90%); 1 remaining." in warning_parts[0].content
        assert "Iterations: 8/10 requests used (80%); 2 remaining." not in warning_parts[0].content

    @pytest.mark.anyio
    async def test_context_warning_clears_after_history_shrinks(self):
        """Test context warnings disappear once the history fits again."""
        processor = LimitWarnerProcessor(
            max_context_tokens=150,
            token_counter=lambda messages: len(messages) * 100,
        )
        messages = _make_messages()
        ctx = _make_ctx()

        warned_messages = await processor(ctx, messages)
        assert len(_generated_warning_parts(warned_messages)) == 1

        trimmed_messages = [warned_messages[-1]]
        cleared_messages = await processor(ctx, trimmed_messages)
        assert _generated_warning_parts(cleared_messages) == []

    @pytest.mark.anyio
    async def test_iteration_and_total_token_warnings_persist_after_context_shrinks(self):
        """Test non-context warnings still apply after compaction reduces context usage."""
        processor = LimitWarnerProcessor(
            max_iterations=10,
            max_context_tokens=250,
            max_total_tokens=100,
            token_counter=lambda messages: len(messages) * 100,
        )
        messages = _make_messages()
        ctx = _make_ctx(requests=8, input_tokens=55, output_tokens=25)

        warned_messages = await processor(ctx, messages)
        trimmed_messages = [warned_messages[-1]]
        rewarned_messages = await processor(ctx, trimmed_messages)
        warning_parts = _generated_warning_parts(rewarned_messages)

        assert len(warning_parts) == 1
        warning = warning_parts[0].content
        assert "Iterations: 8/10 requests used (80%); 2 remaining." in warning
        assert "Total tokens: 80/100 used (80%); 20 remaining." in warning
        assert "Context window:" not in warning

    @pytest.mark.anyio
    async def test_async_token_counter_is_awaited(self):
        """Test async token counters are supported for context warnings."""
        calls: list[int] = []

        async def token_counter(messages: list[ModelMessage]) -> int:
            calls.append(len(messages))
            return 250

        processor = LimitWarnerProcessor(
            max_context_tokens=300,
            token_counter=token_counter,
        )
        messages = _make_messages()

        result = await processor(_make_ctx(), messages)
        assert calls == [3]
        assert len(_generated_warning_parts(result)) == 1
