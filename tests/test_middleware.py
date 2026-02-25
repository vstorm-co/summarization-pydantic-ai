"""Tests for ContextManagerMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_summarization.middleware import (
    ContextManagerMiddleware,
    _truncate_tool_output,
    create_context_manager_middleware,
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


def _make_large_messages(n: int, content_size: int = 1000) -> list[ModelMessage]:
    """Create messages with large content for token threshold testing."""
    msgs: list[ModelMessage] = []
    for i in range(n):
        content = "x" * content_size
        if i % 2 == 0:
            msgs.append(ModelRequest(parts=[UserPromptPart(content=content)]))
        else:
            msgs.append(ModelResponse(parts=[TextPart(content=content)]))
    return msgs


class TestContextManagerInit:
    """Tests for ContextManagerMiddleware initialization."""

    def test_defaults(self):
        m = ContextManagerMiddleware()
        assert m.max_tokens == 200_000
        assert m.compress_threshold == 0.9
        assert m.keep == ("messages", 0)
        assert m.summarization_model == "openai:gpt-4.1-mini"
        assert m.max_tool_output_tokens is None
        assert m.tool_output_head_lines == 5
        assert m.tool_output_tail_lines == 5
        assert m.on_usage_update is None
        assert m.compression_count == 0

    def test_custom_params(self):
        m = ContextManagerMiddleware(
            max_tokens=100_000,
            compress_threshold=0.8,
            keep=("tokens", 5000),
            summarization_model="openai:gpt-4.1",
            max_tool_output_tokens=500,
            tool_output_head_lines=10,
            tool_output_tail_lines=3,
        )
        assert m.max_tokens == 100_000
        assert m.compress_threshold == 0.8
        assert m.keep == ("tokens", 5000)
        assert m.max_tool_output_tokens == 500

    def test_invalid_threshold_zero(self):
        with pytest.raises(ValueError, match="compress_threshold must be between 0 and 1"):
            ContextManagerMiddleware(compress_threshold=0.0)

    def test_invalid_threshold_above_one(self):
        with pytest.raises(ValueError, match="compress_threshold must be between 0 and 1"):
            ContextManagerMiddleware(compress_threshold=1.5)

    def test_invalid_threshold_negative(self):
        with pytest.raises(ValueError, match="compress_threshold must be between 0 and 1"):
            ContextManagerMiddleware(compress_threshold=-0.1)

    def test_threshold_one_valid(self):
        m = ContextManagerMiddleware(compress_threshold=1.0)
        assert m.compress_threshold == 1.0

    def test_valid_keep_zero(self):
        m = ContextManagerMiddleware(keep=("messages", 0))
        assert m.keep == ("messages", 0)

    def test_invalid_keep_negative(self):
        with pytest.raises(ValueError, match="non-negative"):
            ContextManagerMiddleware(keep=("messages", -1))

    def test_fraction_keep_requires_max_input_tokens(self):
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            ContextManagerMiddleware(keep=("fraction", 0.3))

    def test_fraction_keep_with_max_input_tokens(self):
        m = ContextManagerMiddleware(keep=("fraction", 0.3), max_input_tokens=200_000)
        assert m.keep == ("fraction", 0.3)
        assert m.max_input_tokens == 200_000


class TestUsageTracking:
    """Tests for __call__ usage tracking."""

    async def test_below_threshold_no_compress(self):
        """Messages below threshold should pass through unchanged."""
        m = ContextManagerMiddleware(max_tokens=1_000_000, compress_threshold=0.9)
        msgs = _make_messages(4)
        result = await m(msgs)
        assert result == msgs
        assert m.compression_count == 0

    async def test_usage_callback_called(self):
        """Usage callback should be called with correct values."""
        calls: list[tuple[float, int, int]] = []

        def on_usage(pct: float, current: int, maximum: int) -> None:
            calls.append((pct, current, maximum))

        m = ContextManagerMiddleware(
            max_tokens=1000,
            compress_threshold=0.99,
            on_usage_update=on_usage,
        )
        msgs = _make_messages(4)
        await m(msgs)

        assert len(calls) == 1
        pct, current, maximum = calls[0]
        assert maximum == 1000
        assert current == count_tokens_approximately(msgs)
        assert pct == current / 1000

    async def test_usage_callback_async(self):
        """Async usage callback should be awaited."""
        calls: list[tuple[float, int, int]] = []

        async def on_usage(pct: float, current: int, maximum: int) -> None:
            calls.append((pct, current, maximum))

        m = ContextManagerMiddleware(
            max_tokens=1000,
            compress_threshold=0.99,
            on_usage_update=on_usage,
        )
        msgs = _make_messages(4)
        await m(msgs)
        assert len(calls) == 1

    async def test_no_callback(self):
        """No callback should not raise."""
        m = ContextManagerMiddleware(max_tokens=1000, on_usage_update=None)
        msgs = _make_messages(4)
        result = await m(msgs)
        assert result == msgs

    async def test_zero_max_tokens(self):
        """Zero max_tokens should produce 0 percentage."""
        calls: list[tuple[float, int, int]] = []

        def on_usage(pct: float, current: int, maximum: int) -> None:
            calls.append((pct, current, maximum))

        m = ContextManagerMiddleware(
            max_tokens=0,
            compress_threshold=1.0,
            on_usage_update=on_usage,
        )
        msgs = _make_messages(4)
        await m(msgs)
        assert calls[0][0] == 0.0


class TestAutoCompression:
    """Tests for auto-compression behavior."""

    async def test_compress_triggers_above_threshold(self):
        """Compression should trigger when above threshold."""
        m = ContextManagerMiddleware(
            max_tokens=10,  # Very low to trigger compression
            compress_threshold=0.1,
            keep=("messages", 2),
        )
        msgs = _make_large_messages(10, content_size=100)

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = msgs[-2:]
            await m(msgs)
            mock_compress.assert_called_once_with(msgs, focus=None)
            assert m.compression_count == 1

    async def test_compress_not_triggered_below_threshold(self):
        """Compression should not trigger when below threshold."""
        m = ContextManagerMiddleware(
            max_tokens=1_000_000,
            compress_threshold=0.9,
        )
        msgs = _make_messages(4)

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            await m(msgs)
            mock_compress.assert_not_called()
            assert m.compression_count == 0

    async def test_compression_count_increments(self):
        """Compression count should increment on each compression."""
        m = ContextManagerMiddleware(
            max_tokens=10,
            compress_threshold=0.1,
            keep=("messages", 2),
        )
        msgs = _make_large_messages(10, content_size=100)

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = msgs[-2:]
            await m(msgs)
            await m(msgs)
            assert m.compression_count == 2

    async def test_compress_cutoff_zero_returns_unchanged(self):
        """When cutoff is 0 (too few messages), _compress returns unchanged."""
        m = ContextManagerMiddleware(
            max_tokens=100,
            compress_threshold=0.1,
            keep=("messages", 50),
        )
        msgs = _make_messages(4)

        # Manually call _compress — cutoff should be 0 (too few messages to cut)
        result = await m._compress(msgs)
        assert result == msgs

    async def test_usage_callback_after_compression(self):
        """Usage callback should be called again after compression."""
        calls: list[tuple[float, int, int]] = []

        def on_usage(pct: float, current: int, maximum: int) -> None:
            calls.append((pct, current, maximum))

        m = ContextManagerMiddleware(
            max_tokens=10,
            compress_threshold=0.1,
            on_usage_update=on_usage,
        )
        msgs = _make_large_messages(10, content_size=100)

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = _make_messages(2)
            await m(msgs)
            # Should be called twice: before and after compression
            assert len(calls) == 2


class TestToolOutputTruncation:
    """Tests for after_tool_call tool output truncation."""

    async def test_disabled_when_no_limit(self):
        """No truncation when max_tool_output_tokens is None."""
        m = ContextManagerMiddleware(max_tool_output_tokens=None)
        result = await m.after_tool_call("tool", {}, "long " * 10000, None)
        assert result == "long " * 10000

    async def test_short_output_passes_through(self):
        """Short output should pass through unchanged."""
        m = ContextManagerMiddleware(max_tool_output_tokens=1000)
        result = await m.after_tool_call("tool", {}, "short", None)
        assert result == "short"

    async def test_long_output_truncated(self):
        """Long output should be truncated."""
        m = ContextManagerMiddleware(
            max_tool_output_tokens=10,
            tool_output_head_lines=2,
            tool_output_tail_lines=2,
        )
        long_output = "\n".join(f"line-{i}" for i in range(100))
        result = await m.after_tool_call("tool", {}, long_output, None)
        assert "line-0" in result
        assert "line-1" in result
        assert "line-99" in result
        assert "line-98" in result
        assert "omitted" in result

    async def test_non_string_result_converted(self):
        """Non-string results should be converted."""
        m = ContextManagerMiddleware(max_tool_output_tokens=1000)
        result = await m.after_tool_call("tool", {}, 42, None)
        assert result == 42  # under limit, passed through

    async def test_non_string_long_truncated(self):
        """Non-string results exceeding limit should be truncated."""
        m = ContextManagerMiddleware(
            max_tool_output_tokens=5,
            tool_output_head_lines=1,
            tool_output_tail_lines=1,
        )
        # Use an object whose str() produces multiple lines
        long_text = "\n".join(f"line-{i}: {'x' * 50}" for i in range(100))

        class BigResult:
            def __str__(self) -> str:
                return long_text

        result = await m.after_tool_call("tool", {}, BigResult(), None)
        assert isinstance(result, str)
        assert "omitted" in result

    async def test_with_ctx_parameter(self):
        """Ctx parameter should be accepted."""
        m = ContextManagerMiddleware(max_tool_output_tokens=None)
        result = await m.after_tool_call("tool", {}, "output", None, ctx="context")
        assert result == "output"


class TestTruncateToolOutput:
    """Tests for _truncate_tool_output helper."""

    def test_short_text_unchanged(self):
        text = "line1\nline2\nline3"
        assert _truncate_tool_output(text, 5, 5) == text

    def test_exact_boundary(self):
        text = "\n".join(f"line-{i}" for i in range(10))
        result = _truncate_tool_output(text, 5, 5)
        assert result == text

    def test_truncation(self):
        text = "\n".join(f"line-{i}" for i in range(20))
        result = _truncate_tool_output(text, 2, 2)
        assert "line-0" in result
        assert "line-1" in result
        assert "line-18" in result
        assert "line-19" in result
        assert "16 lines omitted" in result

    def test_single_head_tail(self):
        text = "\n".join(f"line-{i}" for i in range(10))
        result = _truncate_tool_output(text, 1, 1)
        assert "line-0" in result
        assert "line-9" in result
        assert "8 lines omitted" in result


class TestFactory:
    """Tests for create_context_manager_middleware factory."""

    def test_defaults(self):
        m = create_context_manager_middleware()
        assert m.max_tokens == 200_000
        assert m.compress_threshold == 0.9
        assert m.keep == ("messages", 0)
        assert m.summarization_model == "openai:gpt-4.1-mini"
        assert m.on_usage_update is None

    def test_custom_params(self):
        counter = count_tokens_approximately
        prompt = "Custom prompt: {messages}"

        def callback(pct: float, cur: int, mx: int) -> None:
            pass

        m = create_context_manager_middleware(
            max_tokens=100_000,
            compress_threshold=0.8,
            keep=("tokens", 5000),
            summarization_model="openai:gpt-4.1",
            token_counter=counter,
            summary_prompt=prompt,
            max_tool_output_tokens=500,
            tool_output_head_lines=10,
            tool_output_tail_lines=3,
            on_usage_update=callback,
            max_input_tokens=200_000,
        )
        assert m.max_tokens == 100_000
        assert m.compress_threshold == 0.8
        assert m.keep == ("tokens", 5000)
        assert m.summarization_model == "openai:gpt-4.1"
        assert m.token_counter is counter
        assert m.summary_prompt == prompt
        assert m.max_tool_output_tokens == 500
        assert m.tool_output_head_lines == 10
        assert m.tool_output_tail_lines == 3
        assert m.on_usage_update is callback
        assert m.max_input_tokens == 200_000

    def test_none_optionals_excluded(self):
        m = create_context_manager_middleware()
        assert m.token_counter is count_tokens_approximately
        assert m.summary_prompt is not None
