"""Type definitions for summarization-pydantic-ai."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Literal

from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import KnownModelName, Model

# Type alias for token counting functions (sync or async)
TokenCounter = (
    Callable[[Sequence[ModelMessage]], int] | Callable[[Sequence[ModelMessage]], Awaitable[int]]
)
"""Function type that counts tokens in a sequence of messages.

Supports both synchronous and asynchronous callables. When an async
callable is provided, the middleware will ``await`` the result.

Example:
    ```python
    # Sync counter (simple, fast)
    def my_token_counter(messages: Sequence[ModelMessage]) -> int:
        return sum(len(str(msg)) for msg in messages) // 4

    # Async counter (using pydantic-ai's model-based counting)
    async def model_token_counter(messages: Sequence[ModelMessage]) -> int:
        from pydantic_ai import models
        model = models.infer_model("openai:gpt-4.1")
        usage = await model.count_tokens(list(messages), None, None)
        return usage.request_tokens or 0

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

WarningOn = Literal["iterations", "context_window", "total_tokens"]
"""Warning categories supported by ``LimitWarnerProcessor``.

Can be:
- ``"iterations"`` - warn as request count approaches the configured maximum
- ``"context_window"`` - warn as the current message history approaches the
  configured context budget
- ``"total_tokens"`` - warn as cumulative run token usage approaches the configured maximum

Example:
    ```python
    warn_on: list[WarningOn] = ["iterations", "context_window"]
    ```
"""

ModelType = str | Model | KnownModelName
"""Union type for model specification.

Accepts string model names, pydantic-ai Model objects, or KnownModelName literals.
This allows using custom model configurations like Azure OpenAI providers.

Examples:
    ```python
    from pydantic_ai_summarization import ModelType

    # String model name
    model: ModelType = "openai:gpt-4.1"

    # Custom model object (e.g., Azure OpenAI)
    from pydantic_ai.models.openai import OpenAIModel
    from pydantic_ai.providers.openai import OpenAIProvider
    model: ModelType = OpenAIModel("gpt-4o", provider=OpenAIProvider(base_url=...))
    ```
"""

__all__ = [
    "TokenCounter",
    "ContextFraction",
    "ContextTokens",
    "ContextMessages",
    "ContextSize",
    "WarningOn",
    "ModelType",
]
