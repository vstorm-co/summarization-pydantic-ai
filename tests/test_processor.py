"""Tests for SummarizationProcessor."""

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import ModelResponse as _MR
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_ai_summarization import (
    SummarizationProcessor,
    count_tokens_approximately,
    create_summarization_processor,
    format_messages_for_summary,
)


class TestTokenCounting:
    """Tests for token counting utilities."""

    def test_count_tokens_empty(self):
        """Test counting tokens in empty message list."""
        assert count_tokens_approximately([]) == 0

    def test_count_tokens_user_prompt(self):
        """Test counting tokens in user prompt."""
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello world")])]
        tokens = count_tokens_approximately(messages)
        # "Hello world" = 11 chars / 4 = ~2-3 tokens
        assert tokens > 0

    def test_count_tokens_user_prompt_multipart(self):
        """Test counting tokens in multipart user prompt."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ]
                    )
                ]
            )
        ]
        tokens = count_tokens_approximately(messages)
        assert tokens > 0

    def test_count_tokens_system_prompt(self):
        """Test counting tokens in system prompt."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="You are helpful")])
        ]
        tokens = count_tokens_approximately(messages)
        assert tokens > 0

    def test_count_tokens_ai_response(self):
        """Test counting tokens in AI response."""
        messages: list[ModelMessage] = [
            ModelResponse(parts=[TextPart(content="This is a test response")])
        ]
        tokens = count_tokens_approximately(messages)
        assert tokens > 0

    def test_count_tokens_tool_messages(self):
        """Test counting tokens in tool messages."""
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="test_tool", args={"key": "value"}, tool_call_id="1")]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="test_tool", content="Tool result", tool_call_id="1")
                ]
            ),
        ]
        tokens = count_tokens_approximately(messages)
        assert tokens > 0


class TestFormatMessages:
    """Tests for message formatting."""

    def test_format_user_message(self):
        """Test formatting user messages."""
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        formatted = format_messages_for_summary(messages)
        assert "User: Hello" in formatted

    def test_format_user_message_multipart(self):
        """Test formatting multipart user messages."""
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    UserPromptPart(
                        content=[
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": "World"},
                        ]
                    )
                ]
            )
        ]
        formatted = format_messages_for_summary(messages)
        assert "User: Hello World" in formatted

    def test_format_user_message_multipart_empty(self):
        """Test formatting multipart user messages with no text."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=[{"type": "image", "data": "..."}])])
        ]
        formatted = format_messages_for_summary(messages)
        assert formatted == ""

    def test_format_ai_response(self):
        """Test formatting AI responses."""
        messages: list[ModelMessage] = [ModelResponse(parts=[TextPart(content="Hi there")])]
        formatted = format_messages_for_summary(messages)
        assert "Assistant: Hi there" in formatted

    def test_format_system_prompt(self):
        """Test formatting system prompts."""
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="You are helpful")])
        ]
        formatted = format_messages_for_summary(messages)
        assert "System: You are helpful" in formatted

    def test_format_tool_messages(self):
        """Test formatting tool call and return messages."""
        messages: list[ModelMessage] = [
            ModelResponse(
                parts=[ToolCallPart(tool_name="search", args={"q": "test"}, tool_call_id="1")]
            ),
            ModelRequest(
                parts=[
                    ToolReturnPart(tool_name="search", content="Found 5 results", tool_call_id="1")
                ]
            ),
        ]
        formatted = format_messages_for_summary(messages)
        assert "Tool Call [search]" in formatted
        assert "Tool [search]: Found 5 results" in formatted

    def test_format_tool_return_truncation(self):
        """Test that long tool returns are truncated."""
        long_content = "x" * 600
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[ToolReturnPart(tool_name="search", content=long_content, tool_call_id="1")]
            ),
        ]
        formatted = format_messages_for_summary(messages)
        assert "..." in formatted
        assert len(formatted) < len(long_content)


class TestSummarizationProcessor:
    """Tests for SummarizationProcessor."""

    def test_create_with_defaults(self):
        """Test creating processor with default settings."""
        processor = create_summarization_processor()
        assert processor.model == "openai:gpt-4.1"
        assert processor.trigger == ("tokens", 170000)
        assert processor.keep == ("messages", 20)

    def test_create_with_custom_settings(self):
        """Test creating processor with custom settings."""
        processor = create_summarization_processor(
            model="openai:gpt-4",
            trigger=("messages", 50),
            keep=("messages", 10),
        )
        assert processor.model == "openai:gpt-4"
        assert processor.trigger == ("messages", 50)
        assert processor.keep == ("messages", 10)

    def test_create_with_multiple_triggers(self):
        """Test creating processor with multiple triggers."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=[("messages", 50), ("tokens", 100000)],
            keep=("messages", 10),
        )
        assert len(processor._trigger_conditions) == 2

    def test_fraction_trigger_requires_max_tokens(self):
        """Test that fraction trigger requires max_input_tokens."""
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("fraction", 0.8),
            )

    def test_fraction_trigger_with_max_tokens(self):
        """Test fraction trigger with max_input_tokens provided."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("fraction", 0.8),
            max_input_tokens=200000,
        )
        assert processor._trigger_conditions == [("fraction", 0.8)]

    def test_invalid_fraction_value(self):
        """Test that invalid fraction values are rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("fraction", 1.5),
                max_input_tokens=200000,
            )

    def test_invalid_message_threshold(self):
        """Test that invalid message thresholds are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("messages", -1),
            )

    def test_should_summarize_no_trigger(self):
        """Test that no summarization happens without trigger."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=None,
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])] * 100
        assert not processor._should_summarize(messages, 100000)

    def test_should_summarize_message_trigger(self):
        """Test summarization triggers on message count."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 10),
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])] * 15
        assert processor._should_summarize(messages, 100)

    def test_should_summarize_token_trigger(self):
        """Test summarization triggers on token count."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("tokens", 100),
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        assert processor._should_summarize(messages, 150)

    def test_find_safe_cutoff(self):
        """Test finding safe cutoff point."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            keep=("messages", 5),
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content=f"Message {i}")]) for i in range(20)
        ]
        cutoff = processor._find_safe_cutoff(messages, 5)
        # Should keep last 5, so cutoff at 15
        assert cutoff == 15

    def test_find_safe_cutoff_with_tool_pairs(self):
        """Test that safe cutoff preserves tool call pairs."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            keep=("messages", 3),
        )
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
        # Should find a safe cutoff point - checking the algorithm finds one
        # that doesn't break the tool call/return sequence
        assert cutoff >= 0
        assert cutoff <= 2  # Should cut before or within the tool pair to keep last 3

    def test_is_safe_cutoff_point(self):
        """Test checking if cutoff point is safe."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
        )
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

    @pytest.mark.anyio
    async def test_call_no_summarization_needed(self):
        """Test processor returns messages unchanged when no summarization needed."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 100),  # High threshold
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
            ModelResponse(parts=[TextPart(content="Hi")]),
        ]
        result = await processor(messages)
        assert result == messages

    @pytest.mark.anyio
    async def test_call_below_cutoff_threshold(self):
        """Test processor returns messages when below cutoff."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 10),  # Keep more than we have
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])] * 6
        result = await processor(messages)
        # Should return as-is since we can't cut anything
        assert len(result) == 6

    def test_should_summarize_fraction_trigger(self):
        """Test summarization triggers on fraction of max tokens."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("fraction", 0.5),
            max_input_tokens=1000,
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        # 600 tokens > 0.5 * 1000 = 500 threshold
        assert processor._should_summarize(messages, 600)
        # 400 tokens < 0.5 * 1000 = 500 threshold
        assert not processor._should_summarize(messages, 400)

    def test_determine_cutoff_with_token_keep(self):
        """Test cutoff determination with token-based keep."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("tokens", 100),
        )
        # Create messages with small token footprint
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="x" * 20)]) for _ in range(10)
        ]
        cutoff = processor._determine_cutoff_index(messages)
        assert cutoff >= 0

    def test_determine_cutoff_with_fraction_keep(self):
        """Test cutoff determination with fraction-based keep."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
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
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
        )
        assert processor._find_token_based_cutoff([], 100) == 0

    def test_find_token_based_cutoff_below_limit(self):
        """Test token-based cutoff when already below limit."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
        )
        messages: list[ModelMessage] = [ModelRequest(parts=[UserPromptPart(content="Hi")])]
        assert processor._find_token_based_cutoff(messages, 10000) == 0

    def test_find_token_based_cutoff_needs_cut(self):
        """Test token-based cutoff when cut is needed."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
        )
        # Create messages with substantial content
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="x" * 100)]) for _ in range(20)
        ]
        # Target keeping only ~100 tokens worth
        cutoff = processor._find_token_based_cutoff(messages, 100)
        # Should cut some messages
        assert cutoff > 0

    def test_invalid_context_type(self):
        """Test that invalid context type raises error."""
        with pytest.raises(ValueError, match="Unsupported context size type"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("invalid", 10),  # type: ignore[arg-type]
            )

    def test_create_processor_with_custom_token_counter(self):
        """Test creating processor with custom token counter."""

        def custom_counter(messages: list[ModelMessage]) -> int:
            return len(messages) * 100

        processor = create_summarization_processor(token_counter=custom_counter)
        assert processor.token_counter is custom_counter

    def test_create_processor_with_custom_prompt(self):
        """Test creating processor with custom summary prompt."""
        custom_prompt = "Summarize: {messages}"
        processor = create_summarization_processor(summary_prompt=custom_prompt)
        assert processor.summary_prompt == custom_prompt

    def test_create_processor_with_max_input_tokens(self):
        """Test creating processor with max_input_tokens."""
        processor = create_summarization_processor(max_input_tokens=100000)
        assert processor.max_input_tokens == 100000

    def test_fraction_keep_requires_max_tokens(self):
        """Test that fraction-based keep requires max_input_tokens."""
        with pytest.raises(ValueError, match="max_input_tokens is required"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("messages", 10),
                keep=("fraction", 0.5),
            )

    def test_invalid_token_threshold(self):
        """Test that invalid token thresholds are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("tokens", -100),
            )

    def test_valid_keep_zero(self):
        """Test that zero keep is valid (only summary survives)."""
        proc = SummarizationProcessor(
            model="openai:gpt-4.1",
            keep=("messages", 0),
        )
        assert proc.keep == ("messages", 0)

    def test_invalid_keep_negative(self):
        """Test that negative keep thresholds are rejected."""
        with pytest.raises(ValueError, match="non-negative"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                keep=("messages", -1),
            )

    def test_zero_fraction(self):
        """Test that zero fraction value is rejected."""
        with pytest.raises(ValueError, match="between 0 and 1"):
            SummarizationProcessor(
                model="openai:gpt-4.1",
                trigger=("fraction", 0.0),
                max_input_tokens=100000,
            )

    def test_validate_context_size(self):
        """Test _validate_context_size wrapper delegates correctly."""
        processor = SummarizationProcessor(model="openai:gpt-4.1")
        result = processor._validate_context_size(("messages", 10), "test")
        assert result == ("messages", 10)


def _build_summarizable_messages(count: int = 12) -> list[ModelMessage]:
    """Build a long message list that triggers summarization."""
    messages: list[ModelMessage] = []
    for i in range(count):
        messages.append(ModelRequest(parts=[UserPromptPart(content=f"User message {i}")]))
        messages.append(ModelResponse(parts=[TextPart(content=f"Assistant reply {i}")]))
    return messages


class TestSummarizationCallPath:
    """Tests for the __call__ summarization path (focus + error handling)."""

    @pytest.mark.anyio
    async def test_call_summarizes_and_threads_focus(self):
        """A provided focus is forwarded into the summarization prompt."""

        captured_prompts: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> _MR:
            last = messages[-1]
            for part in last.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    captured_prompts.append(content)
            return _MR(parts=[TextPart(content="SUMMARY")])

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )
        processor._summarization_agent = Agent(FunctionModel(respond))

        messages = _build_summarizable_messages()
        result = await processor(messages, focus="billing logic")

        # History was replaced by a summary message plus the preserved tail.
        assert len(result) < len(messages)
        assert isinstance(result[0], ModelRequest)
        assert "billing logic" in captured_prompts[0]

    @pytest.mark.anyio
    async def test_call_summarizes_without_focus(self):
        """Without focus the prompt contains no focus block."""

        captured_prompts: list[str] = []

        def respond(messages: list[ModelMessage], info: AgentInfo) -> _MR:
            last = messages[-1]
            for part in last.parts:
                content = getattr(part, "content", None)
                if isinstance(content, str):
                    captured_prompts.append(content)
            return _MR(parts=[TextPart(content="SUMMARY")])

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )
        processor._summarization_agent = Agent(FunctionModel(respond))

        messages = _build_summarizable_messages()
        result = await processor(messages)

        assert len(result) < len(messages)
        assert "<focus>" not in captured_prompts[0]

    @pytest.mark.anyio
    async def test_call_keeps_history_on_summary_failure(self):
        """If summary generation raises, the original history is returned unchanged."""

        def boom(messages: list[ModelMessage], info: AgentInfo) -> _MR:
            raise RuntimeError("secret connection string leaked")

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )
        processor._summarization_agent = Agent(FunctionModel(boom))

        messages = _build_summarizable_messages()
        result = await processor(messages)

        # Context is preserved; no error text injected into the history.
        assert result == messages
        for msg in result:
            for part in msg.parts:
                content = getattr(part, "content", "")
                if isinstance(content, str):
                    assert "secret connection string" not in content
                    assert "Error generating summary" not in content


class TestAsyncTokenCounterCallPath:
    """Async ``token_counter`` must work end-to-end through the compress path.

    Regression tests for https://github.com/vstorm-co/summarization-pydantic-ai/issues/28
    where the gating check awaited the counter but the compression/cutoff path
    called it synchronously, crashing with a coroutine ``TypeError``.
    """

    @pytest.mark.anyio
    async def test_call_async_counter_triggers_summarization(self):
        """An async counter drives both the trigger check and summarization."""
        calls = 0

        async def async_counter(messages: list[ModelMessage]) -> int:
            nonlocal calls
            calls += 1
            return 999_999

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("tokens", 100),
            keep=("messages", 4),
            token_counter=async_counter,
        )
        processor._summarization_agent = Agent(
            FunctionModel(lambda m, i: _MR(parts=[TextPart(content="SUMMARY")]))
        )

        messages = _build_summarizable_messages()
        result = await processor(messages)

        assert calls >= 1
        assert len(result) < len(messages)
        assert isinstance(result[0], ModelRequest)

    @pytest.mark.anyio
    async def test_call_async_counter_token_keep(self):
        """An async counter is awaited inside the token-based binary-search cutoff."""

        async def async_counter(messages: list[ModelMessage]) -> int:
            return len(messages) * 10

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("tokens", 10),
            keep=("tokens", 30),
            token_counter=async_counter,
        )
        processor._summarization_agent = Agent(
            FunctionModel(lambda m, i: _MR(parts=[TextPart(content="SUMMARY")]))
        )

        messages = _build_summarizable_messages()
        result = await processor(messages)

        # Summary message plus a small preserved tail (~3 messages worth of tokens).
        assert len(result) < len(messages)
        assert isinstance(result[0], ModelRequest)

    @pytest.mark.anyio
    async def test_call_async_counter_below_trigger(self):
        """An async counter under the trigger returns history unchanged."""

        async def async_counter(messages: list[ModelMessage]) -> int:
            return 5

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("tokens", 100),
            keep=("messages", 4),
            token_counter=async_counter,
        )

        messages = _build_summarizable_messages()
        result = await processor(messages)

        assert result == messages


class TestTwoPhaseApi:
    """Tests for plan_compression / execute_plan / process (issue #30)."""

    @pytest.mark.anyio
    async def test_plan_returns_none_when_below_trigger(self):
        """plan_compression returns None when no trigger condition matches."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 100),
            keep=("messages", 4),
        )
        messages = _build_summarizable_messages(count=2)
        plan = await processor.plan_compression(messages)
        assert plan is None

    @pytest.mark.anyio
    async def test_plan_returns_none_when_cutoff_collapses_to_zero(self):
        """plan_compression returns None when cutoff computes to 0.

        Trigger fires (5+ messages) but keep=("messages", N) where N >= len(messages)
        collapses the cutoff to 0, so there's nothing to summarize.
        """
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 100),  # larger than message count
        )
        messages = _build_summarizable_messages(count=6)
        plan = await processor.plan_compression(messages)
        assert plan is None

    @pytest.mark.anyio
    async def test_plan_returns_plan_when_triggered(self):
        """plan_compression returns a plan with the real cutoff index."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )
        messages = _build_summarizable_messages(count=6)  # 12 messages total
        plan = await processor.plan_compression(messages)

        assert plan is not None
        # cutoff is len(messages) - keep_value, adjusted for safe cutoff.
        assert 0 < plan.cutoff_index < len(messages)
        assert len(plan.messages_to_summarize) == plan.cutoff_index
        assert len(plan.preserved_messages) == len(messages) - plan.cutoff_index
        # Slicing must reconstruct the original.
        assert plan.messages_to_summarize + plan.preserved_messages == messages

    @pytest.mark.anyio
    async def test_plan_force_bypasses_trigger_check(self):
        """force=True plans compression even when trigger conditions don't match."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 100),  # never matches
            keep=("messages", 4),
        )
        messages = _build_summarizable_messages(count=6)

        # Without force: no plan.
        assert await processor.plan_compression(messages) is None

        # With force: plan is returned.
        plan = await processor.plan_compression(messages, force=True)
        assert plan is not None
        assert plan.cutoff_index > 0

    @pytest.mark.anyio
    async def test_execute_plan_succeeds(self):
        """execute_plan returns a SummarizationResult with summarized=True."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )
        processor._summarization_agent = Agent(
            FunctionModel(lambda m, i: _MR(parts=[TextPart(content="Compact summary")]))
        )
        messages = _build_summarizable_messages(count=6)
        plan = await processor.plan_compression(messages)
        assert plan is not None

        result = await processor.execute_plan(plan, focus="api design")

        assert result.summarized is True
        assert result.summary == "Compact summary"
        assert result.cutoff_index == plan.cutoff_index
        # First message is the summary-bearing ModelRequest.
        assert isinstance(result.messages[0], ModelRequest)
        assert len(result.messages) < len(messages)

    @pytest.mark.anyio
    async def test_execute_plan_handles_llm_failure(self):
        """execute_plan returns summarized=False on LLM failure, preserving history."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )

        def boom(messages: list[ModelMessage], info: AgentInfo) -> _MR:
            raise RuntimeError("summary LLM exploded")

        processor._summarization_agent = Agent(FunctionModel(boom))
        messages = _build_summarizable_messages(count=6)
        plan = await processor.plan_compression(messages)
        assert plan is not None

        result = await processor.execute_plan(plan)

        assert result.summarized is False
        assert result.summary is None
        assert result.skip_reason == "failed"
        # Reconstructed history matches the original.
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_process_no_trigger_returns_not_triggered(self):
        """process() returns skip_reason='not_triggered' when below threshold."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 100),
            keep=("messages", 4),
        )
        messages = _build_summarizable_messages(count=2)
        result = await processor.process(messages)

        assert result.summarized is False
        assert result.skip_reason == "not_triggered"
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_process_cutoff_zero_returns_cutoff_zero(self):
        """process() returns skip_reason='cutoff_zero' when trigger fires but cutoff is 0."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 100),  # collapses cutoff to 0
        )
        messages = _build_summarizable_messages(count=6)
        result = await processor.process(messages)

        assert result.summarized is False
        assert result.skip_reason == "cutoff_zero"
        assert result.messages == messages

    @pytest.mark.anyio
    async def test_process_force_bypasses_trigger(self):
        """process(force=True) compresses even when triggers don't match."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 100),  # never matches
            keep=("messages", 4),
        )
        processor._summarization_agent = Agent(
            FunctionModel(lambda m, i: _MR(parts=[TextPart(content="Forced summary")]))
        )
        messages = _build_summarizable_messages(count=6)

        result = await processor.process(messages, force=True)

        assert result.summarized is True
        assert result.summary == "Forced summary"
        assert len(result.messages) < len(messages)

    @pytest.mark.anyio
    async def test_call_returns_messages_only(self):
        """__call__ stays backwards-compatible: returns list[ModelMessage], not result."""
        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("messages", 5),
            keep=("messages", 4),
        )
        processor._summarization_agent = Agent(
            FunctionModel(lambda m, i: _MR(parts=[TextPart(content="SUMMARY")]))
        )
        messages = _build_summarizable_messages(count=6)

        result = await processor(messages)

        # Plain list, no .summarized attribute on it.
        assert isinstance(result, list)
        assert not hasattr(result, "summarized")
