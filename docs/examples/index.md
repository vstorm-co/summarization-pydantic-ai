# Examples

Learn by example with these practical use cases.

## Overview

| Example | Description |
|---------|-------------|
| [Basic Usage](basic-usage.md) | Getting started with both processors |
| [Advanced](advanced.md) | Custom counters, prompts, and configurations |

## Quick Examples

### Minimal Summarization

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor()

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

### Minimal Sliding Window

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

processor = create_sliding_window_processor()

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

### With Custom Trigger

=== "Summarization"

    ```python
    processor = create_summarization_processor(
        trigger=("messages", 30),
        keep=("messages", 10),
    )
    ```

=== "Sliding Window"

    ```python
    processor = create_sliding_window_processor(
        trigger=("messages", 60),
        keep=("messages", 30),
    )
    ```

### With Multiple Triggers

=== "Summarization"

    ```python
    from pydantic_ai_summarization import SummarizationProcessor

    processor = SummarizationProcessor(
        model="openai:gpt-4.1",
        trigger=[
            ("messages", 50),
            ("tokens", 100000),
        ],
        keep=("messages", 20),
    )
    ```

=== "Sliding Window"

    ```python
    from pydantic_ai_summarization import SlidingWindowProcessor

    processor = SlidingWindowProcessor(
        trigger=[
            ("messages", 100),
            ("tokens", 50000),
        ],
        keep=("messages", 30),
    )
    ```

## Choosing a Processor

| Scenario | Recommended |
|----------|-------------|
| Context quality matters | `SummarizationProcessor` |
| Speed/cost matters | `SlidingWindowProcessor` |
| Many parallel conversations | `SlidingWindowProcessor` |
| Coding assistant | `SummarizationProcessor` |
| Simple chatbot | `SlidingWindowProcessor` |

## Next Steps

- See [Basic Usage](basic-usage.md) for detailed examples
- See [Advanced](advanced.md) for custom configurations
