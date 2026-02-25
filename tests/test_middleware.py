"""Tests for ContextManagerMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_summarization.middleware import (
    ContextManagerMiddleware,
    _truncate_tool_output,
    create_context_manager_middleware,
    resolve_max_tokens,
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

    def test_factory_with_model_name(self):
        """Factory passes model_name to middleware."""
        m = create_context_manager_middleware(
            max_tokens=100_000,
            model_name="openai:gpt-4.1",
        )
        assert m.model_name == "openai:gpt-4.1"
        assert m.max_tokens == 100_000

    def test_factory_with_on_before_compress(self):
        """Factory passes on_before_compress callback."""

        def before_cb(msgs: list[ModelMessage], idx: int) -> None:
            pass

        m = create_context_manager_middleware(on_before_compress=before_cb)
        assert m.on_before_compress is before_cb

    def test_factory_with_on_after_compress(self):
        """Factory passes on_after_compress callback."""

        def after_cb(msgs: list[ModelMessage]) -> str:
            return "injected"

        m = create_context_manager_middleware(on_after_compress=after_cb)
        assert m.on_after_compress is after_cb

    def test_factory_with_messages_path(self, tmp_path):
        """Factory passes messages_path to middleware."""
        path = str(tmp_path / "messages.json")
        m = create_context_manager_middleware(messages_path=path)
        assert m.messages_path == path


class TestResolveMaxTokens:
    """Tests for resolve_max_tokens function."""

    def test_known_model_with_provider(self):
        """Resolve a known model with provider:model format."""
        mock_model = MagicMock()
        mock_model.context_window = 128000

        mock_snapshot = MagicMock()
        mock_snapshot.find_provider_model.return_value = (MagicMock(), mock_model)

        with (
            patch(
                "pydantic_ai_summarization.middleware.resolve_max_tokens",
                wraps=resolve_max_tokens,
            ),
            patch.dict(
                "sys.modules",
                {"genai_prices": MagicMock(), "genai_prices.data_snapshot": MagicMock()},
            ),
            patch(
                "genai_prices.data_snapshot.get_snapshot",
                return_value=mock_snapshot,
            ),
        ):
            result = resolve_max_tokens("openai:gpt-4.1")

        assert result == 128000

    def test_known_model_without_provider(self):
        """Resolve a model without provider prefix."""
        mock_model = MagicMock()
        mock_model.context_window = 200000

        mock_snapshot = MagicMock()
        mock_snapshot.find_provider_model.return_value = (MagicMock(), mock_model)

        with patch(
            "genai_prices.data_snapshot.get_snapshot",
            return_value=mock_snapshot,
        ):
            result = resolve_max_tokens("gpt-4.1")

        assert result == 200000

    def test_model_not_found_returns_none(self):
        """Return None when model lookup raises an exception."""
        mock_snapshot = MagicMock()
        mock_snapshot.find_provider_model.side_effect = Exception("not found")

        with patch(
            "genai_prices.data_snapshot.get_snapshot",
            return_value=mock_snapshot,
        ):
            result = resolve_max_tokens("openai:nonexistent-model")

        assert result is None

    def test_zero_context_window_returns_none(self):
        """Return None when context_window is 0."""
        mock_model = MagicMock()
        mock_model.context_window = 0

        mock_snapshot = MagicMock()
        mock_snapshot.find_provider_model.return_value = (MagicMock(), mock_model)

        with patch(
            "genai_prices.data_snapshot.get_snapshot",
            return_value=mock_snapshot,
        ):
            result = resolve_max_tokens("openai:gpt-4.1")

        assert result is None

    def test_none_context_window_returns_none(self):
        """Return None when context_window is None."""
        mock_model = MagicMock()
        mock_model.context_window = None

        mock_snapshot = MagicMock()
        mock_snapshot.find_provider_model.return_value = (MagicMock(), mock_model)

        with patch(
            "genai_prices.data_snapshot.get_snapshot",
            return_value=mock_snapshot,
        ):
            result = resolve_max_tokens("openai:gpt-4.1")

        assert result is None


class TestPostInitAutoDetect:
    """Tests for __post_init__ auto-detection of max_tokens."""

    def test_auto_detect_with_model_name(self):
        """Auto-detect max_tokens from model_name via resolve_max_tokens."""
        with patch(
            "pydantic_ai_summarization.middleware.resolve_max_tokens",
            return_value=128000,
        ):
            m = ContextManagerMiddleware(model_name="openai:gpt-4.1")
        assert m.max_tokens == 128000

    def test_auto_detect_fallback_when_model_not_found(self):
        """Fall back to 200_000 when resolve_max_tokens returns None."""
        with patch(
            "pydantic_ai_summarization.middleware.resolve_max_tokens",
            return_value=None,
        ):
            m = ContextManagerMiddleware(model_name="openai:nonexistent")
        assert m.max_tokens == 200_000

    def test_auto_detect_without_model_name(self):
        """Fall back to 200_000 when no model_name and no max_tokens."""
        m = ContextManagerMiddleware()
        assert m.max_tokens == 200_000

    def test_explicit_max_tokens_skips_auto_detect(self):
        """Explicit max_tokens bypasses auto-detection."""
        m = ContextManagerMiddleware(max_tokens=50_000, model_name="openai:gpt-4.1")
        assert m.max_tokens == 50_000


class TestHistoryPersistence:
    """Tests for _load_history, _save_new_messages, and _persist_history."""

    def test_load_history_empty_file(self, tmp_path):
        """Loading from an empty file keeps empty history."""
        path = tmp_path / "messages.json"
        path.write_text("")
        m = ContextManagerMiddleware(messages_path=str(path))
        assert m._full_history == []

    def test_load_history_nonexistent_file(self, tmp_path):
        """Loading from a nonexistent file keeps empty history."""
        path = tmp_path / "nonexistent.json"
        m = ContextManagerMiddleware(messages_path=str(path))
        assert m._full_history == []

    def test_load_history_with_existing_messages(self, tmp_path):
        """Loading from file with existing messages populates history."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        path = tmp_path / "messages.json"
        msgs = _make_messages(4)
        data = ModelMessagesTypeAdapter.dump_json(msgs)
        path.write_bytes(data)

        m = ContextManagerMiddleware(messages_path=str(path))
        assert len(m._full_history) == 4

    def test_persist_history_creates_file(self, tmp_path):
        """_persist_history writes history to disk."""
        path = tmp_path / "subdir" / "messages.json"
        m = ContextManagerMiddleware(messages_path=str(path))
        m._full_history = _make_messages(3)
        m._persist_history()

        assert path.exists()
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        loaded = ModelMessagesTypeAdapter.validate_json(path.read_bytes())
        assert len(list(loaded)) == 3

    async def test_save_new_messages_first_call_fresh(self, tmp_path):
        """First call to _save_new_messages with no prior history saves messages."""
        path = tmp_path / "messages.json"
        m = ContextManagerMiddleware(messages_path=str(path))

        msgs = _make_messages(4)
        m._save_new_messages(msgs)

        assert m._history_initialized is True
        assert len(m._full_history) == 4
        assert m._last_context_count == 4

    async def test_save_new_messages_first_call_resumed(self, tmp_path):
        """First call on resumed session (pre-loaded history) skips saving."""
        from pydantic_ai.messages import ModelMessagesTypeAdapter

        path = tmp_path / "messages.json"
        msgs = _make_messages(4)
        path.write_bytes(ModelMessagesTypeAdapter.dump_json(msgs))

        m = ContextManagerMiddleware(messages_path=str(path))
        # _full_history was loaded, simulating a resume
        assert len(m._full_history) == 4

        # First _save_new_messages call should just set _last_context_count
        m._save_new_messages(msgs)
        assert m._history_initialized is True
        assert m._last_context_count == 4
        # No new messages appended
        assert len(m._full_history) == 4

    async def test_save_new_messages_incremental(self, tmp_path):
        """Subsequent calls append only new messages."""
        path = tmp_path / "messages.json"
        m = ContextManagerMiddleware(messages_path=str(path))

        msgs1 = _make_messages(2)
        m._save_new_messages(msgs1)
        assert len(m._full_history) == 2

        # Add more messages (simulating agent adding to the list)
        msgs2 = msgs1 + _make_messages(3)
        m._save_new_messages(msgs2)
        assert len(m._full_history) == 5
        assert m._last_context_count == 5

    async def test_save_new_messages_no_new(self, tmp_path):
        """No new messages means no append."""
        path = tmp_path / "messages.json"
        m = ContextManagerMiddleware(messages_path=str(path))

        msgs = _make_messages(3)
        m._save_new_messages(msgs)
        initial_len = len(m._full_history)

        # Same messages, no change
        m._save_new_messages(msgs)
        assert len(m._full_history) == initial_len

    async def test_call_with_messages_path_saves(self, tmp_path):
        """__call__ saves messages when messages_path is set."""
        path = tmp_path / "messages.json"
        m = ContextManagerMiddleware(
            max_tokens=1_000_000,
            compress_threshold=0.99,
            messages_path=str(path),
        )
        msgs = _make_messages(4)
        result = await m(msgs)
        assert result == msgs
        assert path.exists()
        assert len(m._full_history) == 4


class TestRequestCompact:
    """Tests for request_compact method."""

    async def test_request_compact_triggers_compression(self):
        """request_compact forces compression on next __call__."""
        m = ContextManagerMiddleware(
            max_tokens=1_000_000,  # High enough that threshold won't trigger
            compress_threshold=0.99,
            keep=("messages", 2),
        )
        msgs = _make_messages(4)

        m.request_compact()
        assert m._compact_requested is True
        assert m._compact_focus is None

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = msgs[-2:]
            await m(msgs)
            mock_compress.assert_called_once_with(msgs, focus=None)
            assert m.compression_count == 1

        # Flag should be reset after compression
        assert m._compact_requested is False

    async def test_request_compact_with_focus(self):
        """request_compact with focus passes focus to _compress."""
        m = ContextManagerMiddleware(
            max_tokens=1_000_000,
            compress_threshold=0.99,
            keep=("messages", 2),
        )
        msgs = _make_messages(4)

        m.request_compact(focus="API changes")
        assert m._compact_requested is True
        assert m._compact_focus == "API changes"

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = msgs[-2:]
            await m(msgs)
            mock_compress.assert_called_once_with(msgs, focus="API changes")

        assert m._compact_focus is None


class TestCompactMethod:
    """Tests for the compact() direct method."""

    async def test_compact_compresses_immediately(self):
        """compact() compresses messages and returns result."""
        m = ContextManagerMiddleware(
            max_tokens=1_000_000,
            keep=("messages", 2),
        )
        msgs = _make_messages(6)
        compressed = msgs[-2:]

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = compressed
            result = await m.compact(msgs)
            mock_compress.assert_called_once_with(msgs, focus=None)

        assert result == compressed
        assert m.compression_count == 1

    async def test_compact_with_focus(self):
        """compact() passes focus to _compress."""
        m = ContextManagerMiddleware(
            max_tokens=1_000_000,
            keep=("messages", 2),
        )
        msgs = _make_messages(6)
        compressed = msgs[-2:]

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = compressed
            result = await m.compact(msgs, focus="Focus on tests")
            mock_compress.assert_called_once_with(msgs, focus="Focus on tests")

        assert result == compressed

    async def test_compact_runs_after_compress(self):
        """compact() runs on_after_compress callback."""

        def after_cb(msgs: list[ModelMessage]) -> str:
            return "Re-injected context"

        m = ContextManagerMiddleware(
            max_tokens=1_000_000,
            keep=("messages", 2),
            on_after_compress=after_cb,
        )
        msgs = _make_messages(6)
        summary_msg = ModelRequest(
            parts=[SystemPromptPart(content="Summary of previous conversation:\n\nTest")]
        )
        compressed = [summary_msg, msgs[-1]]

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = compressed
            result = await m.compact(msgs)

        # Should have injected the re-injection message after the summary
        assert len(result) == 3
        assert m.compression_count == 1


class TestOnAfterCompress:
    """Tests for _run_after_compress with on_after_compress callback."""

    async def test_no_callback_returns_unchanged(self):
        """No on_after_compress returns messages unchanged."""
        m = ContextManagerMiddleware(on_after_compress=None)
        msgs = _make_messages(4)
        result = await m._run_after_compress(msgs)
        assert result == msgs

    async def test_callback_returns_string_injects_system_prompt(self):
        """String return from callback is injected as SystemPromptPart."""

        def after_cb(msgs: list[ModelMessage]) -> str:
            return "Critical instructions to preserve"

        m = ContextManagerMiddleware(on_after_compress=after_cb)
        summary = ModelRequest(parts=[SystemPromptPart(content="Summary")])
        preserved = _make_messages(2)
        msgs = [summary, *preserved]

        result = await m._run_after_compress(msgs)

        # Should have injected a message after the summary
        assert len(result) == 4  # summary + injected + 2 preserved
        injected = result[1]
        assert isinstance(injected, ModelRequest)
        assert len(injected.parts) == 1
        assert isinstance(injected.parts[0], SystemPromptPart)
        assert injected.parts[0].content == "Critical instructions to preserve"

    async def test_async_callback_returns_string(self):
        """Async callback returning string is awaited and injected."""

        async def after_cb(msgs: list[ModelMessage]) -> str:
            return "Async injected context"

        m = ContextManagerMiddleware(on_after_compress=after_cb)
        summary = ModelRequest(parts=[SystemPromptPart(content="Summary")])
        msgs = [summary, *_make_messages(2)]

        result = await m._run_after_compress(msgs)

        assert len(result) == 4
        injected = result[1]
        assert isinstance(injected, ModelRequest)
        assert isinstance(injected.parts[0], SystemPromptPart)
        assert injected.parts[0].content == "Async injected context"

    async def test_callback_returns_none_no_injection(self):
        """None return from callback means no injection."""

        def after_cb(msgs: list[ModelMessage]) -> None:
            return None

        m = ContextManagerMiddleware(on_after_compress=after_cb)
        msgs = _make_messages(4)

        result = await m._run_after_compress(msgs)
        assert result == msgs

    async def test_callback_returns_empty_string_no_injection(self):
        """Empty string return from callback means no injection."""

        def after_cb(msgs: list[ModelMessage]) -> str:
            return ""

        m = ContextManagerMiddleware(on_after_compress=after_cb)
        msgs = _make_messages(4)

        result = await m._run_after_compress(msgs)
        assert result == msgs

    async def test_after_compress_called_during_auto_compression(self):
        """on_after_compress is called during auto-compression via __call__."""

        after_calls: list[list[ModelMessage]] = []

        def after_cb(msgs: list[ModelMessage]) -> str:
            after_calls.append(msgs)
            return "Re-injected"

        m = ContextManagerMiddleware(
            max_tokens=10,
            compress_threshold=0.1,
            on_after_compress=after_cb,
        )
        msgs = _make_large_messages(10, content_size=100)
        summary = ModelRequest(parts=[SystemPromptPart(content="Summary")])

        with patch.object(m, "_compress", new_callable=AsyncMock) as mock_compress:
            mock_compress.return_value = [summary, msgs[-1]]
            result = await m(msgs)

        assert len(after_calls) == 1
        # Result should have summary + injected + preserved
        assert len(result) == 3
