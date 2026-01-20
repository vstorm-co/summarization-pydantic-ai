"""Type definitions for summarization-pydantic-ai."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from pydantic_ai.messages import ModelMessage

# Type alias for token counting functions
TokenCounter = Callable[[Sequence[ModelMessage]], int]
"""Function type that counts tokens in a sequence of messages.

Example:
    ```python
    def my_token_counter(messages: Sequence[ModelMessage]) -> int:
        return sum(len(str(msg)) for msg in messages) // 4

    processor = SummarizationProcessor(
        model="openai:gpt-4.1",
        token_counter=my_token_counter,
    )
    ```
"""

# Context size type definitions
ContextFraction = tuple[Literal["fraction"], float]
"""Context size specified as a fraction of max_input_tokens.

Example: ("fraction", 0.8) means 80% of max_input_tokens.
"""

ContextTokens = tuple[Literal["tokens"], int]
"""Context size specified as an absolute token count.

Example: ("tokens", 100000) means 100,000 tokens.
"""

ContextMessages = tuple[Literal["messages"], int]
"""Context size specified as a message count.

Example: ("messages", 50) means 50 messages.
"""

ContextSize = ContextFraction | ContextTokens | ContextMessages
"""Union type for all context size specifications.

Can be:
- `("fraction", float)` - fraction of max_input_tokens (requires max_input_tokens)
- `("tokens", int)` - absolute token count
- `("messages", int)` - message count

Examples:
    ```python
    # Trigger at 80% of context window
    trigger: ContextSize = ("fraction", 0.8)

    # Trigger at 100k tokens
    trigger: ContextSize = ("tokens", 100000)

    # Trigger at 50 messages
    trigger: ContextSize = ("messages", 50)
    ```
"""

__all__ = [
    "TokenCounter",
    "ContextFraction",
    "ContextTokens",
    "ContextMessages",
    "ContextSize",
]
