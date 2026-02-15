"""Context manager middleware for real-time conversation context management.

This module requires the ``hybrid`` extra::

    pip install summarization-pydantic-ai[hybrid]

It provides :class:`ContextManagerMiddleware`, a dual-protocol class that:

1. Acts as a pydantic-ai **history processor** (``__call__``): tracks token
   usage and auto-compresses when approaching the token limit.
2. Acts as a pydantic-ai-middleware **AgentMiddleware** (``after_tool_call``):
   optionally truncates large tool outputs inline.

Example::

    from pydantic_ai import Agent
    from pydantic_ai_middleware import MiddlewareAgent
    from pydantic_ai_summarization import (
        ContextManagerMiddleware,
        create_context_manager_middleware,
    )

    middleware = create_context_manager_middleware(
        max_tokens=200_000,
        compress_threshold=0.9,
        on_usage_update=lambda pct, cur, mx: print(f"{pct:.0%} used"),
    )

    agent = Agent("openai:gpt-4.1", history_processors=[middleware])
    wrapped = MiddlewareAgent(agent, middleware=[middleware])
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    SystemPromptPart,
)

try:
    from pydantic_ai_middleware import AgentMiddleware
except ImportError as err:
    raise ImportError(  # pragma: no cover
        "ContextManagerMiddleware requires the 'hybrid' extra: "
        "pip install summarization-pydantic-ai[hybrid]"
    ) from err

from pydantic_ai_summarization._cutoff import (
    determine_cutoff_index,
    validate_context_size,
)
from pydantic_ai_summarization.processor import (
    DEFAULT_SUMMARY_PROMPT,
    count_tokens_approximately,
    format_messages_for_summary,
)
from pydantic_ai_summarization.types import ContextSize, ModelType, TokenCounter

UsageCallback = Callable[[float, int, int], Any]
"""Callback type for usage updates: ``(percentage, current_tokens, max_tokens)``.

Supports both sync and async callables. If the callable returns an awaitable,
it will be awaited automatically.
"""


def _truncate_tool_output(text: str, head_lines: int, tail_lines: int) -> str:
    """Truncate tool output keeping head and tail lines.

    Args:
        text: Full tool output text.
        head_lines: Lines to keep from the beginning.
        tail_lines: Lines to keep from the end.

    Returns:
        Truncated text with omission indicator.
    """
    lines = text.splitlines()
    total_lines = len(lines)

    if total_lines <= head_lines + tail_lines:
        return text

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    omitted = total_lines - head_lines - tail_lines

    return "\n".join(
        [
            *head,
            f"\n... ({omitted} lines omitted) ...\n",
            *tail,
        ]
    )


@dataclass
class ContextManagerMiddleware(AgentMiddleware[Any]):  # type: ignore[misc]
    """Real-time context management middleware.

    Combines token tracking, auto-compression, and optional tool output
    truncation. Registered both as a pydantic-ai ``history_processor``
    (for per-model-call context management) and as an ``AgentMiddleware``
    (for tool output interception).

    Attributes:
        max_tokens: Maximum token budget for the conversation.
        compress_threshold: Fraction of max_tokens at which auto-compression triggers.
        keep: How much context to retain after compression.
        summarization_model: Model used for generating summaries.
        token_counter: Function to count tokens in messages.
        summary_prompt: Prompt template for summary generation.
        trim_tokens_to_summarize: Max tokens to include when generating the summary.
        max_input_tokens: Model max input tokens (for fraction-based keep).
        max_tool_output_tokens: Per-tool-output token limit before truncation.
        tool_output_head_lines: Lines from the beginning of truncated output.
        tool_output_tail_lines: Lines from the end of truncated output.
        on_usage_update: Callback invoked with usage stats before each model call.
    """

    max_tokens: int = 200_000
    """Maximum token budget for the conversation."""

    compress_threshold: float = 0.9
    """Fraction of max_tokens at which auto-compression triggers (0.0, 1.0]."""

    keep: ContextSize = ("messages", 20)
    """How much context to retain after compression."""

    summarization_model: ModelType = "openai:gpt-4.1-mini"
    """Model used for generating summaries.

    Accepts a string model name, a pydantic-ai Model instance, or a KnownModelName literal.
    """

    token_counter: TokenCounter = field(default=count_tokens_approximately)
    """Function to count tokens in messages."""

    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    """Prompt template for summary generation."""

    trim_tokens_to_summarize: int = 4000
    """Max tokens to include when generating the summary."""

    max_input_tokens: int | None = None
    """Model max input tokens (required for fraction-based keep)."""

    max_tool_output_tokens: int | None = None
    """Per-tool-output token limit before truncation. None disables truncation."""

    tool_output_head_lines: int = 5
    """Lines to show from the beginning of truncated tool output."""

    tool_output_tail_lines: int = 5
    """Lines to show from the end of truncated tool output."""

    on_usage_update: UsageCallback | None = None
    """Callback invoked with ``(percentage, current_tokens, max_tokens)``."""

    _summarization_agent: Agent[None, str] | None = field(default=None, init=False, repr=False)
    _compression_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 < self.compress_threshold <= 1:
            raise ValueError(
                f"compress_threshold must be between 0 and 1, got {self.compress_threshold}."
            )
        self.keep = validate_context_size(self.keep, "keep")
        if self.keep[0] == "fraction" and self.max_input_tokens is None:
            raise ValueError("max_input_tokens is required when using fraction-based keep.")

    # -- History processor protocol (pydantic-ai) --

    async def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """History processor: track usage and auto-compress.

        Called by pydantic-ai before every model request within a run.

        Args:
            messages: Current message history.

        Returns:
            Potentially compressed message history.
        """
        total = self.token_counter(messages)
        pct = total / self.max_tokens if self.max_tokens > 0 else 0.0

        await self._notify_usage(pct, total, self.max_tokens)

        if pct >= self.compress_threshold:
            messages = await self._compress(messages)
            self._compression_count += 1
            new_total = self.token_counter(messages)
            new_pct = new_total / self.max_tokens if self.max_tokens > 0 else 0.0
            await self._notify_usage(new_pct, new_total, self.max_tokens)

        return messages

    # -- Middleware protocol (pydantic-ai-middleware) --

    async def after_tool_call(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        result: Any,
        deps: Any | None,
        ctx: Any | None = None,
    ) -> Any:
        """Middleware hook: optionally truncate large tool outputs.

        Args:
            tool_name: Name of the tool that was called.
            tool_args: Arguments passed to the tool.
            result: The tool's return value.
            deps: Agent dependencies.
            ctx: Middleware scoped context.

        Returns:
            Original or truncated result.
        """
        if self.max_tool_output_tokens is None:
            return result

        result_str = str(result) if not isinstance(result, str) else result
        char_limit = self.max_tool_output_tokens * 4  # ~4 chars per token

        if len(result_str) <= char_limit:
            return result

        return _truncate_tool_output(
            result_str, self.tool_output_head_lines, self.tool_output_tail_lines
        )

    # -- Internal methods --

    async def _notify_usage(self, pct: float, current: int, maximum: int) -> None:
        """Call the usage callback if set, handling sync and async."""
        if self.on_usage_update is None:
            return
        result = self.on_usage_update(pct, current, maximum)
        if inspect.isawaitable(result):
            await result

    async def _compress(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Compress messages via LLM summarization."""
        cutoff_index = determine_cutoff_index(
            messages,
            self.keep,
            self.token_counter,
            self.max_input_tokens,
            default_keep=20,
        )

        if cutoff_index <= 0:
            return messages

        messages_to_summarize = messages[:cutoff_index]  # pragma: no cover
        preserved_messages = messages[cutoff_index:]  # pragma: no cover

        summary = await self._create_summary(messages_to_summarize)  # pragma: no cover

        summary_message = ModelRequest(  # pragma: no cover
            parts=[
                SystemPromptPart(content=f"Summary of previous conversation:\n\n{summary}"),
            ]
        )

        return [summary_message, *preserved_messages]  # pragma: no cover

    def _get_summarization_agent(self) -> Agent[None, str]:  # pragma: no cover
        """Get or create the summarization agent."""
        if self._summarization_agent is None:
            self._summarization_agent = Agent(
                self.summarization_model,
                instructions=(
                    "You are a context summarization assistant. "
                    "Extract the most important information from conversations."
                ),
            )
        return self._summarization_agent

    async def _create_summary(
        self, messages_to_summarize: list[ModelMessage]
    ) -> str:  # pragma: no cover
        """Generate summary for the given messages."""
        if not messages_to_summarize:
            return "No previous conversation history."

        formatted = format_messages_for_summary(messages_to_summarize)

        if self.trim_tokens_to_summarize and len(formatted) > self.trim_tokens_to_summarize * 4:
            formatted = formatted[-(self.trim_tokens_to_summarize * 4) :]

        prompt = self.summary_prompt.format(messages=formatted)

        try:
            agent = self._get_summarization_agent()
            result = await agent.run(prompt)
            return result.output.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    @property
    def compression_count(self) -> int:
        """Number of times compression has been triggered."""
        return self._compression_count


def create_context_manager_middleware(
    max_tokens: int = 200_000,
    compress_threshold: float = 0.9,
    keep: ContextSize = ("messages", 20),
    summarization_model: ModelType = "openai:gpt-4.1-mini",
    token_counter: TokenCounter | None = None,
    summary_prompt: str | None = None,
    max_tool_output_tokens: int | None = None,
    tool_output_head_lines: int = 5,
    tool_output_tail_lines: int = 5,
    on_usage_update: UsageCallback | None = None,
    max_input_tokens: int | None = None,
) -> ContextManagerMiddleware:
    """Create a :class:`ContextManagerMiddleware` with sensible defaults.

    Args:
        max_tokens: Maximum token budget for the conversation.
        compress_threshold: Fraction of max_tokens at which auto-compression triggers.
        keep: How much context to retain after compression.
        summarization_model: Model used for generating summaries.
        token_counter: Custom token counter (default: approximate char-based).
        summary_prompt: Custom prompt template for summaries.
        max_tool_output_tokens: Per-tool-output token limit before truncation.
        tool_output_head_lines: Lines from start of truncated output.
        tool_output_tail_lines: Lines from end of truncated output.
        on_usage_update: Callback for usage updates.
        max_input_tokens: Model max input tokens (for fraction-based keep).

    Returns:
        Configured ContextManagerMiddleware instance.
    """
    kwargs: dict[str, Any] = {
        "max_tokens": max_tokens,
        "compress_threshold": compress_threshold,
        "keep": keep,
        "summarization_model": summarization_model,
        "max_tool_output_tokens": max_tool_output_tokens,
        "tool_output_head_lines": tool_output_head_lines,
        "tool_output_tail_lines": tool_output_tail_lines,
        "max_input_tokens": max_input_tokens,
    }
    if token_counter is not None:
        kwargs["token_counter"] = token_counter
    if summary_prompt is not None:
        kwargs["summary_prompt"] = summary_prompt
    if on_usage_update is not None:
        kwargs["on_usage_update"] = on_usage_update

    return ContextManagerMiddleware(**kwargs)
