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
from pathlib import Path as _Path
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelMessagesTypeAdapter,
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
    async_count_tokens,
    async_determine_cutoff_index,
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

BeforeCompressCallback = Callable[[list[ModelMessage], int], Any]
"""Callback type for pre-compression archival: ``(messages_to_discard, cutoff_index)``.

Called with the messages that are about to be discarded during compression,
before the LLM summarization step. Use this to save conversation history
to persistent storage for later retrieval.

Supports both sync and async callables. If the callable returns an awaitable,
it will be awaited automatically.
"""

AfterCompressCallback = Callable[[list[ModelMessage]], Any]
"""Callback type for post-compression context re-injection.

Called with the compressed message list after compression completes.
Return a string to inject it as a ``SystemPromptPart`` into the context
(inserted after the summary, before preserved messages). Return ``None``
to skip injection.

Supports both sync and async callables.
"""


def resolve_max_tokens(model_name: str) -> int | None:
    """Resolve the context window size for a model using genai-prices.

    Parses the ``"provider:model"`` format and looks up the model's
    context window from the genai-prices package.

    Args:
        model_name: Model identifier (e.g., ``"openai:gpt-4.1"``,
            ``"anthropic:claude-sonnet-4-20250514"``).

    Returns:
        Context window size in tokens, or None if the model is not found.
    """
    try:
        from genai_prices.data_snapshot import get_snapshot
    except ImportError:
        return None

    provider_id = None
    model_ref = model_name
    if ":" in model_name:
        parts = model_name.split(":", 1)
        provider_id = parts[0]
        model_ref = parts[1]

    try:
        snapshot = get_snapshot()
        _provider, model = snapshot.find_provider_model(model_ref, None, provider_id, None)
        if model.context_window and model.context_window > 0:
            return int(model.context_window)
    except Exception:
        pass

    return None


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

    max_tokens: int | None = None
    """Maximum token budget for the conversation.

    When None (default), auto-detected from the model's context window
    via genai-prices using ``model_name``. Falls back to 200,000 if the
    model is not found.
    """

    model_name: str | None = None
    """Model name for auto-detecting ``max_tokens`` from genai-prices.

    Uses the ``"provider:model"`` format (e.g., ``"openai:gpt-4.1"``).
    Only needed when ``max_tokens`` is None (auto-detect mode).
    """

    compress_threshold: float = 0.9
    """Fraction of max_tokens at which auto-compression triggers (0.0, 1.0]."""

    keep: ContextSize = ("messages", 0)
    """How much context to retain after compression.

    Default ``("messages", 0)`` keeps only the summary — all messages are
    summarized and replaced with a single SystemPromptPart. Set to e.g.
    ``("messages", 10)`` to preserve the most recent 10 messages alongside
    the summary.
    """

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

    on_before_compress: BeforeCompressCallback | None = None
    """Callback invoked with ``(messages_to_discard, cutoff_index)`` before compression.

    Called with the messages that are about to be summarized and discarded,
    before the LLM summarization step. Use this to archive conversation
    history to persistent storage.
    """

    on_after_compress: AfterCompressCallback | None = None
    """Callback invoked with ``(compressed_messages)`` after compression.

    Return a string to re-inject it into context as a ``SystemPromptPart``
    (e.g., critical instructions that must survive compaction). Return
    ``None`` to skip injection. Inspired by Claude Code's SessionStart
    hook with compact matcher.
    """

    messages_path: str | None = None
    """Path to messages.json for persistent conversation history.

    When set, every message (user input, agent responses, tool calls) is
    saved continuously to this file. On compression, the summary is also
    appended. The file is the permanent, uncompressed record of the full
    conversation. None disables persistence.
    """

    _summarization_agent: Agent[None, str] | None = field(default=None, init=False, repr=False)
    _compression_count: int = field(default=0, init=False, repr=False)
    _full_history: list[ModelMessage] = field(default_factory=list, init=False, repr=False)
    _last_context_count: int = field(default=0, init=False, repr=False)
    _history_initialized: bool = field(default=False, init=False, repr=False)
    _compact_requested: bool = field(default=False, init=False, repr=False)
    _compact_focus: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate configuration and resolve max_tokens."""
        if not 0 < self.compress_threshold <= 1:
            raise ValueError(
                f"compress_threshold must be between 0 and 1, got {self.compress_threshold}."
            )

        # Auto-detect max_tokens from genai-prices if not set
        if self.max_tokens is None:
            resolved = None
            if self.model_name:
                resolved = resolve_max_tokens(self.model_name)
            self.max_tokens = resolved or 200_000

        self.keep = validate_context_size(self.keep, "keep")
        if self.keep[0] == "fraction" and self.max_input_tokens is None:
            raise ValueError("max_input_tokens is required when using fraction-based keep.")
        if self.messages_path is not None:
            self._load_history()

    # -- History processor protocol (pydantic-ai) --

    async def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """History processor: save messages, track usage, and auto-compress.

        Called by pydantic-ai before every model request within a run.
        Saves new messages to the persistent history file before any
        compression, so the file always has the full uncompressed record.

        Args:
            messages: Current message history.

        Returns:
            Potentially compressed message history.
        """
        # Save new messages to persistent history (before compression)
        if self.messages_path is not None:
            self._save_new_messages(messages)

        # max_tokens is guaranteed to be int after __post_init__
        assert self.max_tokens is not None
        max_tokens: int = self.max_tokens

        total = await async_count_tokens(self.token_counter, messages)
        pct = total / max_tokens if max_tokens > 0 else 0.0

        await self._notify_usage(pct, total, max_tokens)

        # Compress if threshold reached OR manually requested via request_compact()
        should_compress = pct >= self.compress_threshold or self._compact_requested
        if should_compress:
            focus = self._compact_focus
            self._compact_requested = False
            self._compact_focus = None

            messages = await self._compress(messages, focus=focus)
            self._compression_count += 1

            # Re-inject context after compression (Claude Code pattern)
            messages = await self._run_after_compress(messages)

            # Save the compression summary to persistent history
            if self.messages_path is not None and messages:  # pragma: no cover
                self._full_history.append(messages[0])  # summary message
                self._persist_history()
                self._last_context_count = len(messages)

            new_total = await async_count_tokens(self.token_counter, messages)
            new_pct = new_total / max_tokens if max_tokens > 0 else 0.0
            await self._notify_usage(new_pct, new_total, max_tokens)

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

    # -- Public compaction API --

    def request_compact(self, focus: str | None = None) -> None:
        """Request manual compaction on the next ``__call__``.

        The next time the history processor runs (i.e., the next model
        request), compression will be forced regardless of token usage.

        Args:
            focus: Optional focus instructions for the summary
                (e.g., "Focus on the API changes").
        """
        self._compact_requested = True
        self._compact_focus = focus

    async def compact(
        self,
        messages: list[ModelMessage],
        focus: str | None = None,
    ) -> list[ModelMessage]:
        """Directly compact messages with LLM summarization.

        Unlike ``request_compact()`` which defers to the next ``__call__``,
        this method compresses immediately and returns the result.
        Useful for CLI ``/compact`` commands.

        Args:
            messages: Current message history.
            focus: Optional focus instructions for the summary.

        Returns:
            Compressed message list.
        """
        compressed = await self._compress(messages, focus=focus)
        self._compression_count += 1

        # Re-inject context after compression
        compressed = await self._run_after_compress(compressed)

        # Update persistence tracking
        if self.messages_path is not None and compressed:  # pragma: no cover
            self._full_history.append(compressed[0])
            self._persist_history()
            self._last_context_count = len(compressed)

        return compressed

    # -- History persistence --

    def _save_new_messages(self, messages: list[ModelMessage]) -> None:
        """Detect and append new messages to the persistent history.

        On first call, if there's already saved history (session resume),
        skip saving — those messages are already in the file. Otherwise,
        detect messages added since the last call and append them.
        """
        if not self._history_initialized:
            self._history_initialized = True
            if self._full_history:
                # Resuming session — context messages already in history
                self._last_context_count = len(messages)
                return

        if len(messages) > self._last_context_count:
            new_msgs = messages[self._last_context_count :]
            self._full_history.extend(new_msgs)
            self._last_context_count = len(messages)
            self._persist_history()

    def _load_history(self) -> None:
        """Load existing messages from the persistent history file."""
        path = _Path(self.messages_path)  # type: ignore[arg-type]
        if path.exists():
            try:
                raw = path.read_bytes()
                if raw:
                    self._full_history = list(ModelMessagesTypeAdapter.validate_json(raw))
            except Exception:  # pragma: no cover
                pass

    def _persist_history(self) -> None:
        """Write the full history to the messages file."""
        path = _Path(self.messages_path)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        data = ModelMessagesTypeAdapter.dump_json(self._full_history)
        path.write_bytes(data)

    # -- Internal methods --

    async def _notify_usage(self, pct: float, current: int, maximum: int) -> None:
        """Call the usage callback if set, handling sync and async."""
        if self.on_usage_update is None:
            return
        result = self.on_usage_update(pct, current, maximum)
        if inspect.isawaitable(result):
            await result

    async def _compress(
        self,
        messages: list[ModelMessage],
        focus: str | None = None,
    ) -> list[ModelMessage]:
        """Compress messages via LLM summarization.

        Args:
            messages: Messages to compress.
            focus: Optional focus instructions appended to the summary prompt
                (e.g., "Focus on the API changes").
        """
        cutoff_index = await async_determine_cutoff_index(
            messages,
            self.keep,
            self.token_counter,
            self.max_input_tokens,
            default_keep=0,
        )

        if cutoff_index <= 0:
            return messages

        messages_to_summarize = messages[:cutoff_index]  # pragma: no cover
        preserved_messages = messages[cutoff_index:]  # pragma: no cover

        # Archive callback — save messages before they're summarized and discarded
        if self.on_before_compress is not None:  # pragma: no cover
            cb_result = self.on_before_compress(messages_to_summarize, cutoff_index)
            if inspect.isawaitable(cb_result):
                await cb_result

        summary = await self._create_summary(  # pragma: no cover
            messages_to_summarize, focus=focus
        )

        summary_message = ModelRequest(  # pragma: no cover
            parts=[
                SystemPromptPart(content=f"Summary of previous conversation:\n\n{summary}"),
            ]
        )

        return [summary_message, *preserved_messages]  # pragma: no cover

    async def _run_after_compress(
        self, messages: list[ModelMessage]
    ) -> list[ModelMessage]:  # pragma: no cover
        """Run the after-compress callback and optionally inject context.

        If ``on_after_compress`` returns a string, it's inserted as a
        ``SystemPromptPart`` after the summary message.
        """
        if self.on_after_compress is None:
            return messages

        result = self.on_after_compress(messages)
        if inspect.isawaitable(result):
            result = await result

        if result and isinstance(result, str):
            inject_msg = ModelRequest(parts=[SystemPromptPart(content=result)])
            # Insert after summary (index 0), before preserved messages
            messages = [messages[0], inject_msg, *messages[1:]]

        return messages

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
        self,
        messages_to_summarize: list[ModelMessage],
        focus: str | None = None,
    ) -> str:  # pragma: no cover
        """Generate summary for the given messages.

        Args:
            messages_to_summarize: Messages to summarize.
            focus: Optional focus instructions appended to the prompt.
        """
        if not messages_to_summarize:
            return "No previous conversation history."

        formatted = format_messages_for_summary(messages_to_summarize)

        if self.trim_tokens_to_summarize and len(formatted) > self.trim_tokens_to_summarize * 4:
            formatted = formatted[-(self.trim_tokens_to_summarize * 4) :]

        prompt = self.summary_prompt.format(messages=formatted)
        if focus:
            prompt += f"\n\nIMPORTANT: Focus the summary on: {focus}"

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
    max_tokens: int | None = None,
    compress_threshold: float = 0.9,
    keep: ContextSize = ("messages", 0),
    summarization_model: ModelType = "openai:gpt-4.1-mini",
    token_counter: TokenCounter | None = None,
    summary_prompt: str | None = None,
    max_tool_output_tokens: int | None = None,
    tool_output_head_lines: int = 5,
    tool_output_tail_lines: int = 5,
    on_usage_update: UsageCallback | None = None,
    on_before_compress: BeforeCompressCallback | None = None,
    on_after_compress: AfterCompressCallback | None = None,
    max_input_tokens: int | None = None,
    messages_path: str | None = None,
    model_name: str | None = None,
) -> ContextManagerMiddleware:
    """Create a :class:`ContextManagerMiddleware` with sensible defaults.

    Args:
        max_tokens: Maximum token budget for the conversation. When None
            (default), auto-detected from ``model_name`` via genai-prices.
            Falls back to 200,000 if the model is not found.
        compress_threshold: Fraction of max_tokens at which auto-compression triggers.
        keep: How much context to retain after compression.
        summarization_model: Model used for generating summaries.
        token_counter: Custom token counter (default: approximate char-based).
        summary_prompt: Custom prompt template for summaries.
        max_tool_output_tokens: Per-tool-output token limit before truncation.
        tool_output_head_lines: Lines from start of truncated output.
        tool_output_tail_lines: Lines from end of truncated output.
        on_usage_update: Callback for usage updates.
        on_before_compress: Callback invoked with ``(messages, cutoff_index)``
            before compression discards messages. Use for archival.
        on_after_compress: Callback invoked with ``(compressed_messages)``
            after compression. Return a string to re-inject into context.
        max_input_tokens: Model max input tokens (for fraction-based keep).
        messages_path: Absolute path to messages.json for persistent history.
            Every message is saved continuously. None disables persistence.
        model_name: Model identifier for auto-detecting ``max_tokens`` from
            genai-prices (e.g., ``"openai:gpt-4.1"``). Only used when
            ``max_tokens`` is None.

    Returns:
        Configured ContextManagerMiddleware instance.
    """
    kwargs: dict[str, Any] = {
        "compress_threshold": compress_threshold,
        "keep": keep,
        "summarization_model": summarization_model,
        "max_tool_output_tokens": max_tool_output_tokens,
        "tool_output_head_lines": tool_output_head_lines,
        "tool_output_tail_lines": tool_output_tail_lines,
        "max_input_tokens": max_input_tokens,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if model_name is not None:
        kwargs["model_name"] = model_name
    if token_counter is not None:
        kwargs["token_counter"] = token_counter
    if summary_prompt is not None:
        kwargs["summary_prompt"] = summary_prompt
    if on_usage_update is not None:
        kwargs["on_usage_update"] = on_usage_update
    if on_before_compress is not None:
        kwargs["on_before_compress"] = on_before_compress
    if on_after_compress is not None:
        kwargs["on_after_compress"] = on_after_compress
    if messages_path is not None:
        kwargs["messages_path"] = messages_path

    return ContextManagerMiddleware(**kwargs)
