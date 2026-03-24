"""Tests for pydantic-ai capability wrappers."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_ai.usage import RunUsage

from pydantic_ai_summarization.capability import (
    ContextManagerCapability,
    LimitWarnerCapability,
    SlidingWindowCapability,
    SummarizationCapability,
)


def _make_ctx() -> Any:
    """Create a minimal RunContext for testing."""
    from pydantic_ai import RunContext

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
        assert cap.max_tokens == 200_000
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
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.tools import ToolDefinition

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
        from pydantic_ai.messages import ToolCallPart
        from pydantic_ai.tools import ToolDefinition

        cap = ContextManagerCapability(max_tool_output_tokens=None)
        ctx = _make_ctx()

        call = ToolCallPart(tool_name="grep", args={}, tool_call_id="c1")
        tool_def = ToolDefinition(name="grep", description="search")

        large = "x" * 100_000
        result = await cap.after_tool_execute(
            ctx, call=call, tool_def=tool_def, args={}, result=large
        )
        assert result == large


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
