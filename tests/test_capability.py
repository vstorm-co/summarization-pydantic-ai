"""Tests for pydantic-ai capability wrappers."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ToolCallPart,
    ToolReturn,
    UserPromptPart,
)
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from pydantic_ai_summarization.capability import (
    ContextManagerCapability,
    LimitWarnerCapability,
    SlidingWindowCapability,
    SummarizationCapability,
)


def _make_ctx() -> Any:
    """Create a minimal RunContext for testing."""

    return RunContext(deps=None, model=TestModel(), usage=RunUsage())


class TestSummarizationCapability:
    """Tests for SummarizationCapability."""

    def test_default_construction(self):
        cap = SummarizationCapability()
        assert cap.trigger == ("messages", 50)
        assert cap.keep == ("messages", 10)
        assert cap._processor is not None

    @pytest.mark.anyio
    async def test_agent_runs(self):
        cap = SummarizationCapability()
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None


class TestSlidingWindowCapability:
    """Tests for SlidingWindowCapability."""

    def test_default_construction(self):
        cap = SlidingWindowCapability()
        assert cap.trigger == ("messages", 100)
        assert cap.keep == ("messages", 50)
        assert cap._processor is not None

    @pytest.mark.anyio
    async def test_agent_runs(self):
        cap = SlidingWindowCapability()
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None


class TestLimitWarnerCapability:
    """Tests for LimitWarnerCapability."""

    def test_construction(self):
        cap = LimitWarnerCapability(max_iterations=40, max_context_tokens=100_000)
        assert cap.max_iterations == 40
        assert cap._processor is not None

    @pytest.mark.anyio
    async def test_agent_runs(self):
        cap = LimitWarnerCapability(max_iterations=40)
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None


class TestContextManagerCapability:
    """Tests for ContextManagerCapability."""

    def test_default_construction(self):
        cap = ContextManagerCapability()
        assert cap.max_tokens is None
        assert cap._resolved_max_tokens == 200_000
        assert cap.compress_threshold == 0.9
        assert cap.compression_count == 0
        assert cap._summarization_processor is not None

    def test_invalid_threshold(self):
        with pytest.raises(ValueError, match="between 0 and 1"):
            ContextManagerCapability(compress_threshold=1.5)

    def test_request_compact(self):
        cap = ContextManagerCapability()
        assert cap._compact_requested is False
        cap.request_compact(focus="preserve credentials")
        assert cap._compact_requested is True
        assert cap._compact_focus == "preserve credentials"

    def test_include_compact_tool_default_false(self):
        cap = ContextManagerCapability()
        assert cap.include_compact_tool is False
        assert cap.get_toolset() is None

    def test_include_compact_tool_true(self):
        cap = ContextManagerCapability(include_compact_tool=True)
        assert cap.include_compact_tool is True
        toolset = cap.get_toolset()
        assert toolset is not None
        assert "compact_conversation" in toolset.tools

    async def test_compact_conversation_tool_calls_request_compact(self):
        cap = ContextManagerCapability(include_compact_tool=True)
        toolset = cap.get_toolset()
        tool_fn = toolset.tools["compact_conversation"].function
        result = await tool_fn()
        assert "Conversation compaction requested" in result
        assert cap._compact_requested is True

    async def test_compact_conversation_tool_with_focus(self):
        cap = ContextManagerCapability(include_compact_tool=True)
        toolset = cap.get_toolset()
        tool_fn = toolset.tools["compact_conversation"].function
        result = await tool_fn(focus="API design decisions")
        assert "Focus: API design decisions" in result
        assert cap._compact_requested is True
        assert cap._compact_focus == "API design decisions"

    def test_usage_callback(self):
        calls: list[tuple[float, int, int]] = []

        def on_usage(pct: float, current: int, max_tokens: int) -> None:
            calls.append((pct, current, max_tokens))

        cap = ContextManagerCapability(max_tokens=1000, on_usage_update=on_usage)
        assert cap.on_usage_update is not None

    @pytest.mark.anyio
    async def test_agent_runs(self):
        cap = ContextManagerCapability(max_tokens=100_000)
        agent = Agent(TestModel(), capabilities=[cap])
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_tool_output_truncation(self):
        """after_tool_execute truncates large outputs."""

        cap = ContextManagerCapability(max_tool_output_tokens=10)  # 40 chars
        ctx = _make_ctx()

        call = ToolCallPart(tool_name="grep", args={}, tool_call_id="c1")
        tool_def = ToolDefinition(name="grep", description="search")

        # Small output — unchanged
        small = "small result"
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result=small
        )
        assert result == small

        # Large output — truncated
        large = "\n".join(f"line {i}" for i in range(100))
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result=large
        )
        assert "lines omitted" in str(result)

    @pytest.mark.anyio
    async def test_no_truncation_when_disabled(self):
        """No truncation when max_tool_output_tokens is None."""

        cap = ContextManagerCapability(max_tool_output_tokens=None)
        ctx = _make_ctx()

        call = ToolCallPart(tool_name="grep", args={}, tool_call_id="c1")
        tool_def = ToolDefinition(name="grep", description="search")

        large = "x" * 100_000
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result=large
        )
        assert result == large

    @pytest.mark.anyio
    async def test_tool_return_binary_not_stringified(self):
        """A ToolReturn carrying BinaryContent must never be stringified."""
        cap = ContextManagerCapability(max_tool_output_tokens=10)  # 40 chars
        ctx = _make_ctx()
        call = ToolCallPart(tool_name="read_file", args={}, tool_call_id="c1")
        tool_def = ToolDefinition(name="read_file", description="read")

        big_pdf = b"%PDF-1.4\n" + b"\x00" * 5_000_000
        tr = ToolReturn(
            return_value="Successfully read file: doc.pdf",
            content=[
                "Content of doc.pdf:",
                BinaryContent(data=big_pdf, media_type="application/pdf"),
            ],
        )

        result = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={}, result=tr)

        assert isinstance(result, ToolReturn)
        assert result.return_value == "Successfully read file: doc.pdf"
        assert result.content is not None
        binary = result.content[1]
        assert isinstance(binary, BinaryContent)
        assert len(binary.data) == len(big_pdf)

    @pytest.mark.anyio
    async def test_tool_return_long_text_return_value_truncated(self):
        """A ToolReturn's large textual return_value is still truncated."""
        cap = ContextManagerCapability(max_tool_output_tokens=10)  # 40 chars
        ctx = _make_ctx()
        call = ToolCallPart(tool_name="grep", args={}, tool_call_id="c1")
        tool_def = ToolDefinition(name="grep", description="search")

        tr = ToolReturn(return_value="\n".join(f"line {i}" for i in range(100)))
        result = await cap.after_tool_execute(ctx, call=call, tool_def=tool_def, args={}, result=tr)
        assert isinstance(result, ToolReturn)
        assert "omitted" in str(result.return_value)
        assert len(str(result.return_value)) < 200


class TestTruncateToolOutput:
    def test_short_text_unchanged(self):
        """Short text passes through without truncation."""
        from pydantic_ai_summarization.capability import _truncate_tool_output

        result = _truncate_tool_output("line1\nline2\nline3", head_lines=5, tail_lines=5)
        assert result == "line1\nline2\nline3"

    def test_single_long_line_truncated_by_chars(self):
        """A huge single line (no newlines) is bounded by the char budget."""
        from pydantic_ai_summarization.capability import _truncate_tool_output

        result = _truncate_tool_output("x" * 100_000, head_lines=5, tail_lines=5, max_chars=1000)
        assert "characters omitted" in result
        assert len(result) < 1100


class TestSerializationNames:
    """Tests for AgentSpec serialization names."""

    def test_summarization(self):
        assert SummarizationCapability.get_serialization_name() == "SummarizationCapability"

    def test_sliding_window(self):
        assert SlidingWindowCapability.get_serialization_name() == "SlidingWindowCapability"

    def test_limit_warner(self):
        assert LimitWarnerCapability.get_serialization_name() == "LimitWarnerCapability"

    def test_context_manager(self):
        assert ContextManagerCapability.get_serialization_name() == "ContextManagerCapability"


class TestContextManagerCompact:
    """Tests for compact() method."""

    @pytest.mark.anyio
    async def test_compact_returns_messages(self):
        """compact() processes messages and returns result."""
        cap = ContextManagerCapability(max_tokens=100_000)
        # With a very simple message list, processor won't compress

        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        result = await cap.compact(messages)
        # Small input — won't actually compress, returns as-is
        assert len(result) >= 1

    def test_compact_increments_count(self):
        """compact() increments compression_count."""
        cap = ContextManagerCapability(max_tokens=100_000)
        assert cap.compression_count == 0

    @pytest.mark.anyio
    async def test_compact_forwards_focus_to_processor(self):
        """compact() threads the focus topic through to the summarization processor.

        Uses the two-phase stub to verify focus is passed to execute_plan and
        that compression_count only increments when summarized=True (issue #30).
        """
        cap = ContextManagerCapability(max_tokens=100_000)
        stub = _ProcessorStub()
        cap._summarization_processor = stub  # type: ignore[assignment]

        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        await cap.compact(messages, focus="payment flow")

        assert stub.execute_call_kwargs["focus"] == "payment flow"
        # force=True is what makes the compact tool always compress (issue #30 point #3).
        assert stub.plan_call_kwargs["force"] is True
        assert cap.compression_count == 1

    @pytest.mark.anyio
    async def test_compact_no_increment_when_not_summarized(self):
        """compact() does not increment compression_count when summary fails."""
        cap = ContextManagerCapability(max_tokens=100_000)
        stub = _ProcessorStub(summarized=False)
        cap._summarization_processor = stub  # type: ignore[assignment]

        messages = [ModelRequest(parts=[UserPromptPart(content="Hello")])]
        result = await cap.compact(messages)

        assert cap.compression_count == 0
        # When not summarized, the input messages are returned unchanged.
        assert result == messages

    @pytest.mark.anyio
    async def test_usage_callback_fires(self):
        """on_usage_update is called during before_model_request."""
        calls: list[tuple[float, int, int]] = []

        def on_usage(pct: float, current: int, max_tokens: int) -> None:
            calls.append((pct, current, max_tokens))

        cap = ContextManagerCapability(max_tokens=1000, on_usage_update=on_usage)
        agent = Agent(TestModel(), capabilities=[cap])
        await agent.run("Hello")
        assert len(calls) >= 1

    def test_auto_detect_max_tokens_default(self):
        """Default max_tokens is None, resolved to 200K fallback."""
        cap = ContextManagerCapability()
        assert cap.max_tokens is None
        assert cap._resolved_max_tokens == 200_000

    def test_explicit_max_tokens(self):
        """Explicit max_tokens is used."""
        cap = ContextManagerCapability(max_tokens=50_000)
        assert cap._resolved_max_tokens == 50_000


class _ProcessorStub:
    """Minimal stand-in for SummarizationProcessor that records calls.

    Used to test ContextManagerCapability's two-phase orchestration without
    requiring a real summary LLM. Always returns a plan from plan_compression
    (so the capability proceeds into execute_plan and fires both hooks); the
    `summarized` flag on execute_plan's return value controls whether the
    outcome is success or failure.
    """

    def __init__(self, *, summarized: bool = True, cutoff_index: int = 3) -> None:
        self._summarized = summarized
        self._cutoff_index = cutoff_index
        self.plan_call_kwargs: dict[str, Any] = {}
        self.execute_call_kwargs: dict[str, Any] = {}

    async def plan_compression(self, messages, *, force=False):  # type: ignore[no-untyped-def]
        self.plan_call_kwargs = {"force": force}
        from pydantic_ai_summarization.processor import CompressionPlan

        idx = min(self._cutoff_index, len(messages))
        return CompressionPlan(
            cutoff_index=idx,
            messages_to_summarize=messages[:idx],
            preserved_messages=messages[idx:],
            system_parts=[],
        )

    async def execute_plan(self, plan, focus=None):  # type: ignore[no-untyped-def]
        from pydantic_ai_summarization.processor import SummarizationResult

        self.execute_call_kwargs = {"focus": focus}
        if not self._summarized:
            return SummarizationResult(
                messages=[*plan.messages_to_summarize, *plan.preserved_messages],
                summarized=False,
                skip_reason="failed",
            )
        return SummarizationResult(
            messages=[*plan.preserved_messages],
            summarized=True,
            cutoff_index=plan.cutoff_index,
            summary="stub summary",
        )


class TestContextManagerHookContract:
    """Tests for the on_before_compress / on_after_compress contract (issue #30)."""

    @pytest.mark.anyio
    async def test_on_before_compress_receives_real_cutoff(self):
        """on_before_compress receives the processor's actual cutoff index, not 0."""
        before_calls: list[tuple[int, int]] = []

        def on_before_compress(messages, cutoff_index: int) -> None:
            before_calls.append((len(messages), cutoff_index))

        cap = ContextManagerCapability(max_tokens=100_000, on_before_compress=on_before_compress)
        stub = _ProcessorStub(cutoff_index=7)
        cap._summarization_processor = stub  # type: ignore[assignment]

        messages = [ModelRequest(parts=[UserPromptPart(content="x")])] * 10
        await cap.compact(messages)

        assert before_calls == [(10, 7)]

    @pytest.mark.anyio
    async def test_on_after_compress_receives_summary_and_summarized_flag(self):
        """on_after_compress is called with (messages, summarized=True, summary)."""
        after_calls: list[tuple[bool, str | None]] = []

        def on_after_compress(messages, summarized: bool, summary: str | None) -> None:
            after_calls.append((summarized, summary))

        cap = ContextManagerCapability(max_tokens=100_000, on_after_compress=on_after_compress)
        cap._summarization_processor = _ProcessorStub()  # type: ignore[assignment]

        await cap.compact([ModelRequest(parts=[UserPromptPart(content="x")])])

        assert after_calls == [(True, "stub summary")]

    @pytest.mark.anyio
    async def test_on_after_compress_fires_with_false_when_summary_fails(self):
        """on_after_compress receives summarized=False when the LLM fails."""
        after_calls: list[tuple[bool, str | None]] = []

        def on_after_compress(messages, summarized: bool, summary: str | None) -> None:
            after_calls.append((summarized, summary))

        cap = ContextManagerCapability(max_tokens=100_000, on_after_compress=on_after_compress)
        cap._summarization_processor = _ProcessorStub(summarized=False)  # type: ignore[assignment]

        await cap.compact([ModelRequest(parts=[UserPromptPart(content="x")])])

        assert after_calls == [(False, None)]
        # And compression_count is not incremented.
        assert cap.compression_count == 0

    @pytest.mark.anyio
    async def test_reinject_only_when_summarized(self):
        """The str returned by on_after_compress is re-injected only when summarized=True."""

        reinjected: list[str] = []

        def on_after_compress(messages, summarized: bool, summary):
            reinjected.append("INSTRUCTION")
            return "INSTRUCTION"

        cap = ContextManagerCapability(max_tokens=100_000, on_after_compress=on_after_compress)
        cap._summarization_processor = _ProcessorStub(summarized=False)  # type: ignore[assignment]

        messages = [ModelRequest(parts=[UserPromptPart(content="x")])]
        result = await cap.compact(messages)

        # Callback fired, returned a str, but reinject was suppressed because summarized=False.
        assert reinjected == ["INSTRUCTION"]
        # Messages unchanged.
        assert result == messages

    @pytest.mark.anyio
    async def test_reinject_applied_when_summarized(self):
        """When summarized=True and on_after_compress returns a str, it's re-injected."""
        from pydantic_ai.messages import ModelRequest, SystemPromptPart

        def on_after_compress(messages, summarized: bool, summary):
            return "CRITICAL RULES"

        cap = ContextManagerCapability(max_tokens=100_000, on_after_compress=on_after_compress)
        cap._summarization_processor = _ProcessorStub(summarized=True, cutoff_index=2)  # type: ignore[assignment]

        # Need at least cutoff_index+1 messages so preserved is non-empty.
        messages = [ModelRequest(parts=[UserPromptPart(content=f"msg {i}")]) for i in range(4)]
        result = await cap.compact(messages)

        # First message gained a SystemPromptPart with the reinjected text.
        assert isinstance(result[0], ModelRequest)
        system_contents = [p.content for p in result[0].parts if isinstance(p, SystemPromptPart)]
        assert "CRITICAL RULES" in system_contents

    @pytest.mark.anyio
    async def test_usage_update_fires_again_after_compression(self):
        """on_usage_update fires a second time after compression completes."""
        from types import SimpleNamespace

        calls: list[tuple[float, int, int]] = []

        def on_usage(pct: float, current: int, max_tokens: int) -> None:
            calls.append((pct, current, max_tokens))

        cap = ContextManagerCapability(max_tokens=100_000, on_usage_update=on_usage)
        cap._summarization_processor = _ProcessorStub(summarized=True)  # type: ignore[assignment]
        cap.request_compact()  # force compression on next before_model_request

        ctx = _make_ctx()
        request_context = SimpleNamespace(
            messages=[ModelRequest(parts=[UserPromptPart(content=f"m{i}")]) for i in range(4)]
        )
        await cap.before_model_request(ctx, request_context)

        # Two usage callbacks: pre-compression and post-compression.
        assert len(calls) == 2
        # request_context.messages was replaced with the post-compression list.
        assert len(request_context.messages) < 4

    @pytest.mark.anyio
    async def test_before_model_request_without_usage_callback(self):
        """Compression works when on_usage_update is None (no second callback attempted)."""
        from types import SimpleNamespace

        cap = ContextManagerCapability(max_tokens=100_000)  # no on_usage_update
        cap._summarization_processor = _ProcessorStub(summarized=True, cutoff_index=2)  # type: ignore[assignment]
        cap.request_compact()

        ctx = _make_ctx()
        request_context = SimpleNamespace(
            messages=[ModelRequest(parts=[UserPromptPart(content=f"m{i}")]) for i in range(4)]
        )
        await cap.before_model_request(ctx, request_context)

        # Compression happened even with no usage callback.
        assert len(request_context.messages) < 4

    @pytest.mark.anyio
    async def test_reinject_skipped_when_first_message_not_request(self):
        """Reinject is a no-op when the compressed history starts with a ModelResponse.

        Covers the defensive branch where messages[0] isn't a ModelRequest —
        e.g. when the user provides a history that begins with an assistant turn.
        """
        from pydantic_ai.messages import ModelResponse, TextPart

        def on_after_compress(messages, summarized: bool, summary):
            return "SHOULD NOT APPEAR"

        cap = ContextManagerCapability(max_tokens=100_000, on_after_compress=on_after_compress)
        cap._summarization_processor = _ProcessorStub(summarized=True, cutoff_index=2)  # type: ignore[assignment]

        # After slicing at cutoff=2, preserved_messages starts with a ModelResponse.
        messages = [
            ModelRequest(parts=[UserPromptPart(content="m0")]),
            ModelRequest(parts=[UserPromptPart(content="m1")]),
            ModelResponse(parts=[TextPart(content="a2")]),  # becomes result.messages[0]
            ModelRequest(parts=[UserPromptPart(content="m3")]),
        ]
        result = await cap.compact(messages)

        # Compression happened but reinject was skipped — no new SystemPromptPart.
        assert isinstance(result[0], ModelResponse)
        # Result is shorter than input.
        assert len(result) < len(messages)


class TestMultipleCapabilities:
    """Test combining multiple capabilities."""

    @pytest.mark.anyio
    async def test_warner_and_context_manager(self):
        """LimitWarner + ContextManager work together."""
        agent = Agent(
            TestModel(),
            capabilities=[
                LimitWarnerCapability(max_iterations=40),
                ContextManagerCapability(max_tokens=100_000),
            ],
        )
        result = await agent.run("Hello")
        assert result.output is not None

    @pytest.mark.anyio
    async def test_all_capabilities(self):
        """All four capabilities can be combined."""
        agent = Agent(
            TestModel(),
            capabilities=[
                SummarizationCapability(trigger=("messages", 100), keep=("messages", 20)),
                SlidingWindowCapability(trigger=("messages", 200), keep=("messages", 100)),
                LimitWarnerCapability(max_iterations=50),
                ContextManagerCapability(max_tokens=200_000),
            ],
        )
        result = await agent.run("Hello")
        assert result.output is not None
