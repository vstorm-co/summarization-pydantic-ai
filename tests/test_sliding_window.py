"""Tests for SlidingWindowProcessor."""

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

from pydantic_ai_summarization import (
    SlidingWindowProcessor,
    create_sliding_window_processor,
)


class TestSlidingWindowProcessor:
    """Tests for SlidingWindowProcessor."""

    def test_create_with_defaults(self):
        """Test creating processor with default settings."""
        processor = create_sliding_window_processor()
        assert processor.trigger == ("messages", 100)
        assert processor.keep == ("messages", 50)

    def test_create_with_custom_settings(self):
        """Test creating processor with custom settings."""
        processor = create_sliding_window_processor(
            trigger=("messages", 50),
            keep=("messages", 10),
        )
        assert processor.trigger == ("messages", 50)
        assert processor.keep == ("messages", 10)

    def test_create_with_multiple_triggers(self):
        """Test creating processor with multiple triggers."""
        processor = SlidingWindowProcessor(
            trigger=[("messages", 50), ("tokens", 100000)],
            keep=("messages", 10),
        )
        assert len(processor._trigger_conditions) == 2

    def test_fraction_trigger_requires_max_tokens(self):
        """Test that fraction trigger requires max_input_tokens."""
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            SlidingWindowProcessor(
                trigger=("fraction", 0.8),
            )

    def test_fraction_trigger_with_max_tokens(self):
        """Test fraction trigger with max_input_tokens provided."""
        processor = SlidingWindowProcessor(
            trigger=("fraction", 0.8),
            max_input_tokens=200000,
        )
        assert processor._trigger_conditions == [("fraction", 0.8)]

    def test_invalid_fraction_value(self):
        """Test that invalid fraction values are rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            SlidingWindowProcessor(
                trigger=("fraction", 1.5),
                max_input_tokens=200000,
            )

    def test_invalid_message_threshold(self):
        """Test that invalid message thresholds are rejected."""
        with pytest.raises(ValueError, match="greater than 0"):
            SlidingWindowProcessor(
                trigger=("messages", -1),
            )

    def test_invalid_token_threshold(self):
        """Test that invalid token thresholds are rejected."""
        with pytest.raises(ValueError, match="greater than 0"):
            SlidingWindowProcessor(
                trigger=("tokens", -100),
            )

    def test_invalid_keep_threshold(self):
        """Test that invalid keep thresholds are rejected."""
        with pytest.raises(ValueError, match="greater than 0"):
            SlidingWindowProcessor(
                keep=("messages", 0),
            )

    def test_invalid_context_type(self):
        """Test that invalid context type raises error."""
        with pytest.raises(ValueError, match="Unsupported context size type"):
            SlidingWindowProcessor(
                trigger=("invalid", 10),  # type: ignore[arg-type]
            )

    def test_zero_fraction(self):
        """Test that zero fraction value is rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            SlidingWindowProcessor(
                trigger=("fraction", 0.0),
                max_input_tokens=100000,
            )

    def test_fraction_keep_requires_max_tokens(self):
        """Test that fraction-based keep requires max_input_tokens."""
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            SlidingWindowProcessor(
                trigger=("messages", 10),
                keep=("fraction", 0.5),
            )

    def test_should_trim_no_trigger(self):
        """Test that no trimming happens without trigger."""
        processor = SlidingWindowProcessor(trigger=None)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ] * 100
        assert not processor._should_trim(messages, 100000)

    def test_should_trim_message_trigger(self):
        """Test trimming triggers on message count."""
        processor = SlidingWindowProcessor(trigger=("messages", 10))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ] * 15
        assert processor._should_trim(messages, 100)

    def test_should_trim_token_trigger(self):
        """Test trimming triggers on token count."""
        processor = SlidingWindowProcessor(trigger=("tokens", 100))
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        assert processor._should_trim(messages, 150)

    def test_should_trim_fraction_trigger(self):
        """Test trimming triggers on fraction of max tokens."""
        processor = SlidingWindowProcessor(
            trigger=("fraction", 0.5),
            max_input_tokens=1000,
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        # 600 tokens > 0.5 * 1000 = 500 threshold
        assert processor._should_trim(messages, 600)
        # 400 tokens < 0.5 * 1000 = 500 threshold
        assert not processor._should_trim(messages, 400)

    def test_find_safe_cutoff(self):
        """Test finding safe cutoff point."""
        processor = SlidingWindowProcessor(keep=("messages", 5))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(20)
        ]
        cutoff = processor._find_safe_cutoff(messages, 5)
        # Should keep last 5, so cutoff at 15
        assert cutoff == 15

    def test_find_safe_cutoff_few_messages(self):
        """Test safe cutoff with fewer messages than keep threshold."""
        processor = SlidingWindowProcessor(keep=("messages", 10))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(5)
        ]
        cutoff = processor._find_safe_cutoff(messages, 10)
        # Should return 0 since we have fewer messages than keep
        assert cutoff == 0

    def test_find_safe_cutoff_with_tool_pairs(self):
        """Test that safe cutoff preserves tool call pairs."""
        processor = SlidingWindowProcessor(keep=("messages", 3))
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Question")]),
            ModelResponse(parts=[ToolCallPart(tool_name="search", args={}, tool_call_id="call_1")]),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="search", content="Result", tool_call_id="call_1")]
            ),
            ModelResponse(parts=[TextPart(content="Answer")]),
            ModelRequest(parts=[UserPromptPart(content="Follow up")]),
        ]
        cutoff = processor._find_safe_cutoff(messages, 3)
        # Should find a safe cutoff point that doesn't break tool pairs
        assert cutoff >= 0
        assert cutoff <= 2

    def test_is_safe_cutoff_point(self):
        """Test checking if cutoff point is safe."""
        processor = SlidingWindowProcessor()
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name="test", args={}, tool_call_id="call_1")]),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="test", content="Result", tool_call_id="call_1")]
            ),
        ]
        # Cutting between tool call and return is not safe
        assert not processor._is_safe_cutoff_point(messages, 1)
        # Cutting after both is safe
        assert processor._is_safe_cutoff_point(messages, 2)

    def test_is_safe_cutoff_point_beyond_messages(self):
        """Test safe cutoff when index is beyond message length."""
        processor = SlidingWindowProcessor()
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ]
        assert processor._is_safe_cutoff_point(messages, 5)

    def test_is_safe_cutoff_with_no_tool_id(self):
        """Test safe cutoff with tool call that has no ID."""
        processor = SlidingWindowProcessor()
        messages: list[ModelMessage] = [
            ModelResponse(parts=[ToolCallPart(tool_name="test", args={})]),  # No tool_call_id
            ModelRequest(parts=[UserPromptPart(content="Next message")]),
        ]
        # Should be safe since there's no tool_call_id to track
        assert processor._is_safe_cutoff_point(messages, 1)

    @pytest.mark.anyio
    async def test_call_no_trimming_needed(self):
        """Test processor returns messages unchanged when no trimming needed."""
        processor = SlidingWindowProcessor(
            trigger=("messages", 100),  # High threshold
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi")]),
        ]
        result = await processor(messages)
        assert result == messages

    @pytest.mark.anyio
    async def test_call_triggers_trimming(self):
        """Test processor trims messages when threshold reached."""
        processor = SlidingWindowProcessor(
            trigger=("messages", 5),
            keep=("messages", 3),
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(10)
        ]
        result = await processor(messages)
        # Should keep last 3 messages
        assert len(result) == 3
        # Verify we kept the last ones
        assert result == messages[-3:]

    @pytest.mark.anyio
    async def test_call_below_cutoff_threshold(self):
        """Test processor returns messages when below cutoff."""
        processor = SlidingWindowProcessor(
            trigger=("messages", 5),
            keep=("messages", 10),  # Keep more than we have
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ] * 6
        result = await processor(messages)
        # Should return as-is since we can't cut anything meaningful
        assert len(result) == 6

    @pytest.mark.anyio
    async def test_call_preserves_tool_pairs(self):
        """Test that call preserves tool call/response pairs when trimming."""
        processor = SlidingWindowProcessor(
            trigger=("messages", 4),
            keep=("messages", 2),
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Initial")]),
            ModelResponse(parts=[ToolCallPart(tool_name="test", args={}, tool_call_id="call_1")]),
            ModelRequest(
                parts=[ToolReturnPart(tool_name="test", content="Result", tool_call_id="call_1")]
            ),
            ModelResponse(parts=[TextPart(content="Final answer")]),
        ]
        result = await processor(messages)
        # Should keep messages without breaking tool pairs
        assert len(result) >= 2

    def test_determine_cutoff_with_token_keep(self):
        """Test cutoff determination with token-based keep."""
        processor = SlidingWindowProcessor(
            trigger=("messages", 5),
            keep=("tokens", 100),
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="x" * 20)]) for _ in range(10)
        ]
        cutoff = processor._determine_cutoff_index(messages)
        assert cutoff >= 0

    def test_determine_cutoff_with_fraction_keep(self):
        """Test cutoff determination with fraction-based keep."""
        processor = SlidingWindowProcessor(
            trigger=("messages", 5),
            keep=("fraction", 0.5),
            max_input_tokens=1000,
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="x" * 20)]) for _ in range(10)
        ]
        cutoff = processor._determine_cutoff_index(messages)
        assert cutoff >= 0

    def test_find_token_based_cutoff_empty(self):
        """Test token-based cutoff with empty messages."""
        processor = SlidingWindowProcessor()
        assert processor._find_token_based_cutoff([], 100) == 0

    def test_find_token_based_cutoff_below_limit(self):
        """Test token-based cutoff when already below limit."""
        processor = SlidingWindowProcessor()
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
        assert processor._find_token_based_cutoff(messages, 10000) == 0

    def test_find_token_based_cutoff_needs_cut(self):
        """Test token-based cutoff when cut is needed."""
        processor = SlidingWindowProcessor()
        # Create messages with substantial content
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="x" * 100)]) for _ in range(20)
        ]
        # Target keeping only ~100 tokens worth
        cutoff = processor._find_token_based_cutoff(messages, 100)
        # Should cut some messages
        assert cutoff > 0

    def test_create_with_custom_token_counter(self):
        """Test creating processor with custom token counter."""

        def custom_counter(messages: list[ModelMessage]) -> int:
            return len(messages) * 100

        processor = create_sliding_window_processor(token_counter=custom_counter)
        assert processor.token_counter is custom_counter

    def test_create_with_max_input_tokens(self):
        """Test creating processor with max_input_tokens."""
        processor = create_sliding_window_processor(max_input_tokens=100000)
        assert processor.max_input_tokens == 100000

    def test_create_with_token_trigger(self):
        """Test creating processor with token-based trigger."""
        processor = create_sliding_window_processor(
            trigger=("tokens", 50000),
            keep=("tokens", 25000),
        )
        assert processor.trigger == ("tokens", 50000)
        assert processor.keep == ("tokens", 25000)

    def test_create_with_fraction_config(self):
        """Test creating processor with fraction-based config."""
        processor = create_sliding_window_processor(
            trigger=("fraction", 0.8),
            keep=("fraction", 0.3),
            max_input_tokens=100000,
        )
        assert processor.trigger == ("fraction", 0.8)
        assert processor.keep == ("fraction", 0.3)
        assert processor.max_input_tokens == 100000

    @pytest.mark.anyio
    async def test_call_with_none_trigger(self):
        """Test processor with None trigger returns messages unchanged."""
        processor = SlidingWindowProcessor(trigger=None)
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(100)
        ]
        result = await processor(messages)
        assert result == messages

    def test_trigger_with_multiple_conditions(self):
        """Test processor triggers on first matching condition."""
        processor = SlidingWindowProcessor(
            trigger=[("messages", 100), ("tokens", 50)],  # tokens will trigger first
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")])
        ] * 10
        # Only 10 messages but above token threshold
        assert processor._should_trim(messages, 100)

    def test_non_response_messages_in_cutoff_check(self):
        """Test cutoff check handles non-response messages correctly."""
        processor = SlidingWindowProcessor()
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelRequest(parts=[UserPromptPart(content="World")]),
            ModelRequest(parts=[UserPromptPart(content="Test")]),
        ]
        # All are requests, no tool pairs to break
        assert processor._is_safe_cutoff_point(messages, 1)
        assert processor._is_safe_cutoff_point(messages, 2)
