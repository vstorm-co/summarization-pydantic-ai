"""Capabilities for pydantic-ai agents.

Wraps summarization processors and context management as pydantic-ai
``AbstractCapability`` instances, removing the need for ``pydantic-ai-middleware``.

Example:
    ```python
    from pydantic_ai import Agent
    from pydantic_ai_summarization.capability import ContextManagerCapability

    agent = Agent(
        "openai:gpt-4.1",
        capabilities=[ContextManagerCapability(max_tokens=100_000)],
    )
    ```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.tools import ToolDefinition

from pydantic_ai_summarization._cutoff import async_count_tokens
from pydantic_ai_summarization.limit_warner import LimitWarnerProcessor
from pydantic_ai_summarization.processor import (
    DEFAULT_SUMMARY_PROMPT,
    SummarizationProcessor,
    count_tokens_approximately,
)
from pydantic_ai_summarization.sliding_window import SlidingWindowProcessor
from pydantic_ai_summarization.types import ContextSize, ModelType, TokenCounter, WarningOn

# Callback types (matching middleware.py but defined here to avoid the dependency)
UsageCallback = Any  # (pct: float, current: int, max_tokens: int) -> None
BeforeCompressCallback = Any  # (messages: list, cutoff_index: int) -> None
AfterCompressCallback = Any  # (messages: list) -> str | None


def _truncate_tool_output(text: str, head_lines: int, tail_lines: int) -> str:
    """Truncate tool output keeping head and tail lines."""
    lines = text.splitlines()
    total_lines = len(lines)

    if total_lines <= head_lines + tail_lines:
        return text

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    omitted = total_lines - head_lines - tail_lines

    return "\n".join([*head, f"\n... ({omitted} lines omitted) ...\n", *tail])


@dataclass
class SummarizationCapability(AbstractCapability[Any]):
    """Capability that summarizes conversation history when thresholds are reached.

    Wraps ``SummarizationProcessor`` as a pydantic-ai capability.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization.capability import SummarizationCapability

        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[SummarizationCapability(
                trigger=("messages", 50),
                keep=("messages", 10),
            )],
        )
        ```
    """

    trigger: ContextSize = ("messages", 50)
    keep: ContextSize = ("messages", 10)
    model: ModelType = "openai:gpt-4.1-mini"
    token_counter: TokenCounter = field(default=count_tokens_approximately)
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    _processor: SummarizationProcessor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._processor = SummarizationProcessor(
            trigger=self.trigger,
            keep=self.keep,
            model=self.model,
            token_counter=self.token_counter,
            summary_prompt=self.summary_prompt,
        )

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: Any,
    ) -> Any:
        assert self._processor is not None
        request_context.messages = await self._processor(request_context.messages)
        return request_context


@dataclass
class SlidingWindowCapability(AbstractCapability[Any]):
    """Capability that trims old messages using a sliding window.

    Zero-cost alternative to summarization — discards oldest messages.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization.capability import SlidingWindowCapability

        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[SlidingWindowCapability(
                trigger=("messages", 100),
                keep=("messages", 50),
            )],
        )
        ```
    """

    trigger: ContextSize = ("messages", 100)
    keep: ContextSize = ("messages", 50)
    token_counter: TokenCounter = field(default=count_tokens_approximately)
    _processor: SlidingWindowProcessor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._processor = SlidingWindowProcessor(
            trigger=self.trigger,
            keep=self.keep,
            token_counter=self.token_counter,
        )

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: Any,
    ) -> Any:
        assert self._processor is not None
        request_context.messages = await self._processor(request_context.messages)
        return request_context


@dataclass
class LimitWarnerCapability(AbstractCapability[Any]):
    """Capability that warns the agent when run limits approach.

    Injects warning ``SystemPromptPart`` into the conversation when
    iteration, context window, or total token limits are near.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization.capability import LimitWarnerCapability

        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[LimitWarnerCapability(
                max_iterations=40,
                max_context_tokens=100_000,
            )],
        )
        ```
    """

    max_iterations: int | None = None
    max_context_tokens: int | None = None
    max_total_tokens: int | None = None
    warn_on: list[WarningOn] | None = None
    warning_threshold: float = 0.7
    critical_remaining_iterations: int = 3
    token_counter: TokenCounter = field(default=count_tokens_approximately)
    _processor: LimitWarnerProcessor | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._processor = LimitWarnerProcessor(
            max_iterations=self.max_iterations,
            max_context_tokens=self.max_context_tokens,
            max_total_tokens=self.max_total_tokens,
            warn_on=self.warn_on,
            warning_threshold=self.warning_threshold,
            critical_remaining_iterations=self.critical_remaining_iterations,
            token_counter=self.token_counter,
        )

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: Any,
    ) -> Any:
        assert self._processor is not None
        request_context.messages = await self._processor(ctx, request_context.messages)
        return request_context


@dataclass
class ContextManagerCapability(AbstractCapability[Any]):
    """Full context management capability with token tracking, auto-compression,
    and tool output truncation.

    Replaces ``ContextManagerMiddleware`` + ``pydantic-ai-middleware`` with a native
    pydantic-ai capability. Uses ``before_model_request`` for history processing
    and ``after_tool_execute`` for tool output truncation.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization.capability import ContextManagerCapability

        agent = Agent(
            "openai:gpt-4.1",
            capabilities=[ContextManagerCapability(max_tokens=100_000)],
        )
        ```
    """

    max_tokens: int = 200_000
    compress_threshold: float = 0.9
    keep: ContextSize = ("messages", 0)
    summarization_model: ModelType = "openai:gpt-4.1-mini"
    token_counter: TokenCounter = field(default=count_tokens_approximately)
    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    max_tool_output_tokens: int | None = None
    tool_output_head_lines: int = 5
    tool_output_tail_lines: int = 5
    on_usage_update: UsageCallback | None = None
    on_before_compress: BeforeCompressCallback | None = None
    on_after_compress: AfterCompressCallback | None = None

    _compact_requested: bool = field(default=False, init=False, repr=False)
    _compact_focus: str | None = field(default=None, init=False, repr=False)
    _compression_count: int = field(default=0, init=False, repr=False)
    _summarization_processor: SummarizationProcessor | None = field(
        default=None, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if not 0 < self.compress_threshold <= 1:
            raise ValueError(
                f"compress_threshold must be between 0 and 1, got {self.compress_threshold}."
            )
        self._summarization_processor = SummarizationProcessor(
            trigger=("fraction", self.compress_threshold),
            keep=self.keep,
            model=self.summarization_model,
            token_counter=self.token_counter,
            summary_prompt=self.summary_prompt,
            max_input_tokens=self.max_tokens,
        )

    @property
    def compression_count(self) -> int:
        """Number of times compression has been triggered."""
        return self._compression_count

    def request_compact(self, focus: str | None = None) -> None:
        """Request manual compaction on the next model request.

        Args:
            focus: Optional focus instructions for the summary.
        """
        self._compact_requested = True
        self._compact_focus = focus

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: Any,
    ) -> Any:
        """Track tokens, auto-compress when threshold reached."""
        messages: list[ModelMessage] = request_context.messages

        total = await async_count_tokens(self.token_counter, messages)
        pct = total / self.max_tokens if self.max_tokens > 0 else 0.0

        if self.on_usage_update is not None:
            self.on_usage_update(pct, total, self.max_tokens)

        should_compress = pct >= self.compress_threshold or self._compact_requested
        if should_compress:
            focus = self._compact_focus
            self._compact_requested = False
            self._compact_focus = None

            if self.on_before_compress is not None:
                self.on_before_compress(messages, 0)

            assert self._summarization_processor is not None
            messages = await self._summarization_processor(messages)
            self._compression_count += 1

            if self.on_after_compress is not None:
                result = self.on_after_compress(messages)
                if isinstance(result, str) and messages:
                    from pydantic_ai.messages import ModelRequest, SystemPromptPart

                    first = messages[0]
                    if isinstance(first, ModelRequest):
                        messages[0] = ModelRequest(
                            parts=[*first.parts, SystemPromptPart(content=result)],
                            instructions=first.instructions,
                        )

            new_total = await async_count_tokens(self.token_counter, messages)
            new_pct = new_total / self.max_tokens if self.max_tokens > 0 else 0.0
            if self.on_usage_update is not None:
                self.on_usage_update(new_pct, new_total, self.max_tokens)

        request_context.messages = messages
        return request_context

    async def after_tool_execute(
        self,
        ctx: RunContext[Any],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Truncate large tool outputs."""
        if self.max_tool_output_tokens is None:
            return result

        result_str = str(result) if not isinstance(result, str) else result
        char_limit = self.max_tool_output_tokens * 4

        if len(result_str) <= char_limit:
            return result

        return _truncate_tool_output(
            result_str, self.tool_output_head_lines, self.tool_output_tail_lines
        )


__all__ = [
    "ContextManagerCapability",
    "LimitWarnerCapability",
    "SlidingWindowCapability",
    "SummarizationCapability",
]
