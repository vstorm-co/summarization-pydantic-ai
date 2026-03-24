"""Summarization history processor for managing conversation context."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

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

from pydantic_ai_summarization._cutoff import (
    determine_cutoff_index as _determine_cutoff,
)
from pydantic_ai_summarization._cutoff import (
    find_safe_cutoff as _find_safe,
)
from pydantic_ai_summarization._cutoff import (
    find_token_based_cutoff as _find_token,
)
from pydantic_ai_summarization._cutoff import (
    is_safe_cutoff_point as _is_safe,
)
from pydantic_ai_summarization._cutoff import (
    should_trigger as _should_trigger,
)
from pydantic_ai_summarization._cutoff import (
    validate_context_size as _validate_ctx,
)
from pydantic_ai_summarization._cutoff import (
    validate_triggers_and_keep as _validate_trig_keep,
)
from pydantic_ai_summarization.types import ContextSize, ModelType, TokenCounter

if TYPE_CHECKING:
    pass


DEFAULT_SUMMARY_PROMPT = (
    "<role>\n"
    "Context Extraction Assistant\n"
    "</role>\n\n"
    "<primary_objective>\n"
    "Extract the most relevant context from the conversation history below.\n"
    "</primary_objective>\n\n"
    "<objective_information>\n"
    "You're nearing the token limit and must extract key information. "
    "This context will overwrite the conversation history, so include only "
    "the most important information.\n"
    "</objective_information>\n\n"
    "<instructions>\n"
    "The conversation history will be replaced with your extracted context. "
    "Extract and record the most important context. Focus on information "
    "relevant to the overall goal. Avoid repeating completed actions.\n"
    "</instructions>\n\n"
    "Read the message history carefully. Think about what is most important "
    "to preserve. Extract only essential context.\n\n"
    "Respond ONLY with the extracted context. No additional information.\n\n"
    "<messages>\n"
    "Messages to summarize:\n"
    "{messages}\n"
    "</messages>"
)
"""Default prompt template used for generating summaries."""

DEFAULT_CONTINUATION_PROMPT = "Summary of previous conversation:\n\n"

_DEFAULT_MESSAGES_TO_KEEP = 20
_DEFAULT_TRIGGER_TOKENS = 170000
_DEFAULT_TRIM_TOKEN_LIMIT = 4000
_SEARCH_RANGE_FOR_TOOL_PAIRS = 5


def count_tokens_approximately(messages: Sequence[ModelMessage]) -> int:  # pragma: no branch
    """Approximate token count based on character length.

    This is a simple heuristic: ~4 characters per token on average.
    For production use, consider using a proper tokenizer like tiktoken.

    Args:
        messages: Sequence of messages to count tokens for.

    Returns:
        Approximate token count.

    Example:
        ```python
        from pydantic_ai_summarization import count_tokens_approximately

        messages = [...]  # Your message history
        token_count = count_tokens_approximately(messages)
        print(f"Approximately {token_count} tokens")
        ```
    """
    total_chars = 0
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, UserPromptPart):
                    if isinstance(part.content, str):
                        total_chars += len(part.content)
                    else:
                        # List of content parts
                        for item in part.content:
                            if isinstance(item, dict) and "text" in item:
                                total_chars += len(str(item.get("text", "")))
                elif isinstance(part, SystemPromptPart):
                    total_chars += len(part.content)
                elif isinstance(part, ToolReturnPart):
                    total_chars += len(str(part.content))
        elif isinstance(msg, ModelResponse):
            for response_part in msg.parts:
                if isinstance(response_part, TextPart):
                    total_chars += len(response_part.content)
                elif isinstance(response_part, ToolCallPart):
                    total_chars += len(response_part.tool_name)
                    total_chars += len(str(response_part.args))

    return total_chars // 4


def _format_request_parts(msg: ModelRequest) -> list[str]:  # pragma: no branch
    """Format request message parts."""
    lines: list[str] = []
    for part in msg.parts:
        if isinstance(part, UserPromptPart):
            lines.extend(_format_user_prompt(part))
        elif isinstance(part, SystemPromptPart):
            lines.append(f"System: {part.content}")
        elif isinstance(part, ToolReturnPart):
            content_str = str(part.content)[:500]
            if len(str(part.content)) > 500:
                content_str += "..."
            lines.append(f"Tool [{part.tool_name}]: {content_str}")
    return lines


def _format_user_prompt(part: UserPromptPart) -> list[str]:
    """Format a user prompt part."""
    if isinstance(part.content, str):
        return [f"User: {part.content}"]
    # Handle multi-part content
    text_parts: list[str] = []
    for item in part.content:
        if isinstance(item, dict) and "text" in item:
            text_parts.append(str(item.get("text", "")))
    return [f"User: {' '.join(text_parts)}"] if text_parts else []


def _format_response_parts(msg: ModelResponse) -> list[str]:  # pragma: no branch
    """Format response message parts."""
    lines: list[str] = []
    for part in msg.parts:
        if isinstance(part, TextPart):
            lines.append(f"Assistant: {part.content}")
        elif isinstance(part, ToolCallPart):
            lines.append(f"Tool Call [{part.tool_name}]: {part.args}")
    return lines


def format_messages_for_summary(messages: Sequence[ModelMessage]) -> str:  # pragma: no branch
    """Format messages into a readable string for summarization.

    This function converts a sequence of ModelMessage objects into a
    human-readable format suitable for passing to an LLM for summarization.

    Args:
        messages: Sequence of messages to format.

    Returns:
        Formatted string representation of the messages.

    Example:
        ```python
        from pydantic_ai_summarization import format_messages_for_summary

        messages = [...]  # Your message history
        formatted = format_messages_for_summary(messages)
        print(formatted)
        # User: Hello
        # Assistant: Hi there!
        # Tool Call [search]: {"query": "weather"}
        # Tool [search]: Sunny, 72°F
        ```
    """
    lines: list[str] = []

    for msg in messages:
        if isinstance(msg, ModelRequest):
            lines.extend(_format_request_parts(msg))
        elif isinstance(msg, ModelResponse):
            lines.extend(_format_response_parts(msg))

    return "\n".join(lines)


@dataclass
class SummarizationProcessor:
    """History processor that summarizes conversation when limits are reached.

    This processor monitors message token counts and automatically summarizes
    older messages when a threshold is reached, preserving recent messages
    and maintaining context continuity.

    Attributes:
        model: Model to use for generating summaries.
        trigger: Threshold(s) that trigger summarization.
        keep: How much context to keep after summarization.
        token_counter: Function to count tokens in messages.
        summary_prompt: Prompt template for generating summaries.
        max_input_tokens: Maximum input tokens (required for fraction-based triggers).
        trim_tokens_to_summarize: Maximum tokens to include when generating summary.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization import SummarizationProcessor

        processor = SummarizationProcessor(
            model="openai:gpt-4.1",
            trigger=("tokens", 100000),
            keep=("messages", 10),
        )

        agent = Agent(
            "openai:gpt-4.1",
            history_processors=[processor],
        )
        ```
    """

    model: ModelType
    """Model to use for generating summaries.

    Accepts a string model name (e.g., ``"openai:gpt-4.1"``), a pydantic-ai
    :class:`~pydantic_ai.models.Model` instance, or a
    :data:`~pydantic_ai.models.KnownModelName` literal.
    """

    trigger: ContextSize | list[ContextSize] | None = None
    """Threshold(s) that trigger summarization.

    Examples:
        - ("messages", 50) - trigger when 50+ messages
        - ("tokens", 100000) - trigger when 100k+ tokens
        - ("fraction", 0.8) - trigger at 80% of max tokens (requires max_input_tokens)
    """

    keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP)
    """How much context to keep after summarization.

    Examples:
        - ("messages", 20) - keep last 20 messages
        - ("tokens", 10000) - keep last 10k tokens worth
    """

    token_counter: TokenCounter = field(default=count_tokens_approximately)
    """Function to count tokens in messages."""

    summary_prompt: str = DEFAULT_SUMMARY_PROMPT
    """Prompt template for generating summaries."""

    max_input_tokens: int | None = None
    """Maximum input tokens for the model (required for fraction-based triggers)."""

    trim_tokens_to_summarize: int | None = _DEFAULT_TRIM_TOKEN_LIMIT
    """Maximum tokens to include when generating summary. None to skip trimming."""

    _trigger_conditions: list[ContextSize] = field(default_factory=list, init=False)
    _summarization_agent: Agent[None, str] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate configuration and set up trigger conditions."""
        self._trigger_conditions, self.keep = _validate_trig_keep(
            self.trigger, self.keep, self.max_input_tokens
        )

    def _validate_context_size(self, context: ContextSize, parameter_name: str) -> ContextSize:
        """Validate context configuration tuples."""
        return _validate_ctx(context, parameter_name)

    def _should_summarize(self, messages: list[ModelMessage], total_tokens: int) -> bool:
        """Determine whether summarization should run."""
        return _should_trigger(
            self._trigger_conditions, messages, total_tokens, self.max_input_tokens
        )

    def _determine_cutoff_index(self, messages: list[ModelMessage]) -> int:
        """Choose cutoff index respecting retention configuration."""
        return _determine_cutoff(
            messages,
            self.keep,
            self.token_counter,
            self.max_input_tokens,
            _DEFAULT_MESSAGES_TO_KEEP,
        )

    def _find_token_based_cutoff(
        self, messages: list[ModelMessage], target_token_count: int
    ) -> int:
        """Find cutoff index based on target token retention."""
        return _find_token(messages, target_token_count, self.token_counter)

    def _find_safe_cutoff(self, messages: list[ModelMessage], messages_to_keep: int) -> int:
        """Find safe cutoff point that preserves AI/Tool message pairs."""
        return _find_safe(messages, messages_to_keep)

    def _is_safe_cutoff_point(self, messages: list[ModelMessage], cutoff_index: int) -> bool:
        """Check if cutting at index would separate AI/Tool message pairs."""
        return _is_safe(messages, cutoff_index)

    def _get_summarization_agent(self) -> Agent[None, str]:  # pragma: no cover
        """Get or create the summarization agent."""
        if self._summarization_agent is None:
            self._summarization_agent = Agent(
                self.model,
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

        # Trim if needed
        if self.trim_tokens_to_summarize and len(formatted) > self.trim_tokens_to_summarize * 4:
            formatted = formatted[-(self.trim_tokens_to_summarize * 4) :]

        prompt = self.summary_prompt.format(messages=formatted)

        try:
            agent = self._get_summarization_agent()
            result = await agent.run(prompt)
            return result.output.strip()
        except Exception as e:
            return f"Error generating summary: {e!s}"

    async def __call__(self, messages: list[ModelMessage]) -> list[ModelMessage]:
        """Process messages and summarize if needed.

        This is the main entry point called by pydantic-ai's history processor mechanism.

        Args:
            messages: Current message history.

        Returns:
            Processed message history, potentially with older messages summarized.
        """
        total_tokens = cast(int, self.token_counter(messages))

        if not self._should_summarize(messages, total_tokens):
            return messages

        cutoff_index = self._determine_cutoff_index(messages)

        if cutoff_index <= 0:
            return messages

        # The following code path requires an LLM call, so is covered by integration tests
        messages_to_summarize = messages[:cutoff_index]  # pragma: no cover
        preserved_messages = messages[cutoff_index:]  # pragma: no cover

        summary = await self._create_summary(messages_to_summarize)  # pragma: no cover

        # Create a summary message
        summary_message = ModelRequest(  # pragma: no cover
            parts=[
                SystemPromptPart(content=f"Summary of previous conversation:\n\n{summary}"),
            ]
        )

        return [summary_message, *preserved_messages]  # pragma: no cover


def create_summarization_processor(
    model: ModelType = "openai:gpt-4.1",
    trigger: ContextSize | list[ContextSize] | None = ("tokens", _DEFAULT_TRIGGER_TOKENS),
    keep: ContextSize = ("messages", _DEFAULT_MESSAGES_TO_KEEP),
    max_input_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    summary_prompt: str | None = None,
) -> SummarizationProcessor:
    """Create a summarization history processor.

    This is a convenience factory function for creating SummarizationProcessor
    instances with sensible defaults.

    Args:
        model: Model to use for generating summaries. Accepts a string name,
            a Model instance, or a KnownModelName. Defaults to "openai:gpt-4.1".
        trigger: When to trigger summarization. Can be:
            - ("messages", N) - trigger when N+ messages
            - ("tokens", N) - trigger when N+ tokens
            - ("fraction", F) - trigger at F fraction of max_input_tokens
            - List of tuples to trigger on any condition
            Defaults to ("tokens", 170000).
        keep: How much context to keep after summarization. Defaults to ("messages", 20).
        max_input_tokens: Maximum input tokens (required for fraction-based triggers).
        token_counter: Custom token counting function. Defaults to approximate counter.
        summary_prompt: Custom prompt for summarization. Defaults to built-in prompt.

    Returns:
        Configured SummarizationProcessor.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai_summarization import create_summarization_processor

        processor = create_summarization_processor(
            trigger=("messages", 50),
            keep=("messages", 10),
        )

        agent = Agent(
            "openai:gpt-4.1",
            history_processors=[processor],
        )
        ```
    """
    kwargs: dict[str, Any] = {
        "model": model,
        "trigger": trigger,
        "keep": keep,
    }

    if max_input_tokens is not None:
        kwargs["max_input_tokens"] = max_input_tokens

    if token_counter is not None:
        kwargs["token_counter"] = token_counter

    if summary_prompt is not None:
        kwargs["summary_prompt"] = summary_prompt

    return SummarizationProcessor(**kwargs)


__all__ = [
    "DEFAULT_SUMMARY_PROMPT",
    "SummarizationProcessor",
    "count_tokens_approximately",
    "create_summarization_processor",
    "format_messages_for_summary",
]
