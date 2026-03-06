# API Reference

Complete API documentation for summarization-pydantic-ai.

## Quick Reference

```python
from pydantic_ai_summarization import (
    # Main exports - Summarization
    SummarizationProcessor,
    create_summarization_processor,

    # Main exports - Sliding Window
    SlidingWindowProcessor,
    create_sliding_window_processor,

    # Main exports - Limit Warner
    LimitWarnerProcessor,
    create_limit_warner_processor,

    # Main exports - Context Manager Middleware (requires [hybrid] extra)
    ContextManagerMiddleware,
    UsageCallback,
    create_context_manager_middleware,

    # Utilities
    count_tokens_approximately,
    format_messages_for_summary,

    # Types
    ContextSize,
    ContextFraction,
    ContextTokens,
    ContextMessages,
    WarningOn,
    TokenCounter,

    # Constants
    DEFAULT_SUMMARY_PROMPT,
)
```

## Modules

| Module | Description |
|--------|-------------|
| [Processor](processor.md) | `SummarizationProcessor` class and factory function |
| [Sliding Window](sliding-window.md) | `SlidingWindowProcessor` class and factory function |
| [Limit Warner](limit-warner.md) | `LimitWarnerProcessor` class and factory function |
| [Middleware](middleware.md) | `ContextManagerMiddleware` class and factory function (requires `[hybrid]` extra) |
| [Types](types.md) | Type definitions and aliases |

## Processors Comparison

| Feature | SummarizationProcessor | SlidingWindowProcessor | LimitWarnerProcessor | ContextManagerMiddleware |
|---------|----------------------|----------------------|----------------------|------------------------|
| LLM Cost | High | Zero | Zero | Per compression |
| Latency | High | ~0ms | ~0ms | Low (tracking) / High (compression) |
| Context Preservation | Intelligent | None | Full history | Intelligent |
| Token Tracking | No | No | Built-in warnings | Built-in |
| Usage Callbacks | No | No | No | Yes |
| Tool Output Truncation | No | No | No | Yes |
| `model` param | Required | Not needed | Not needed | Optional (for summaries) |
| Requires extra | No | No | No | `[hybrid]` |

## Quick Examples

### Summarization

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

### Sliding Window

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

processor = create_sliding_window_processor(
    trigger=("messages", 100),
    keep=("messages", 50),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

### Limit Warner

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_limit_warner_processor

processor = create_limit_warner_processor(
    max_iterations=40,
    max_context_tokens=100000,
    max_total_tokens=200000,
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```
