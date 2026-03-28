# API Reference

## Quick Reference

```python
from pydantic_ai_summarization import (
    # Capabilities (recommended)
    ContextManagerCapability,
    SummarizationCapability,
    SlidingWindowCapability,
    LimitWarnerCapability,

    # Standalone processors
    SummarizationProcessor,
    SlidingWindowProcessor,
    LimitWarnerProcessor,
    create_summarization_processor,
    create_sliding_window_processor,
    create_limit_warner_processor,

    # Constants
    DEFAULT_SUMMARY_PROMPT,
    DEFAULT_CONTINUATION_PROMPT,
)
```

## Modules

| Module | Description |
|--------|-------------|
| [Capabilities](capability.md) | `ContextManagerCapability`, `SummarizationCapability`, `SlidingWindowCapability`, `LimitWarnerCapability` |
| [Processor](processor.md) | `SummarizationProcessor` class and factory function |
| [Sliding Window](sliding-window.md) | `SlidingWindowProcessor` class and factory function |
| [Limit Warner](limit-warner.md) | `LimitWarnerProcessor` class and factory function |
| [Types](types.md) | Type definitions and aliases |

## Comparison

| Feature | ContextManagerCapability | SummarizationCapability | SlidingWindowCapability | LimitWarnerCapability |
|---------|:-:|:-:|:-:|:-:|
| LLM Cost | Per compression | High | Zero | Zero |
| Tool Output Truncation | Yes | No | No | No |
| Auto-detect max_tokens | Yes | No | No | No |
| `compact()` method | Yes | No | No | No |
| Token Tracking | Yes | No | No | Warnings |
