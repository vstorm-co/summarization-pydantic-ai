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

    # Utilities
    count_tokens_approximately,
    format_messages_for_summary,

    # Types
    ContextSize,
    ContextFraction,
    ContextTokens,
    ContextMessages,
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
| [Types](types.md) | Type definitions and aliases |

## Processors Comparison

| Feature | SummarizationProcessor | SlidingWindowProcessor |
|---------|----------------------|----------------------|
| LLM Cost | High | Zero |
| Latency | High | ~0ms |
| Context Preservation | Intelligent | None |
| `model` param | Required | Not needed |
| `summary_prompt` | Customizable | Not applicable |
| `trim_tokens_to_summarize` | Supported | Not applicable |

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
