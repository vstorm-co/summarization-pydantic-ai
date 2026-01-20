# Advanced Usage

Advanced configurations and customizations.

## Custom Token Counter

Use your own token counting logic with either processor:

=== "With tiktoken"

    ```python
    from pydantic_ai_summarization import (
        create_summarization_processor,
        create_sliding_window_processor,
    )

    def accurate_counter(messages):
        """Count tokens using tiktoken."""
        import tiktoken
        encoding = tiktoken.encoding_for_model("gpt-4")

        total = 0
        for msg in messages:
            total += len(encoding.encode(str(msg)))
        return total

    # With summarization
    processor = create_summarization_processor(
        token_counter=accurate_counter,
        trigger=("tokens", 100000),
    )

    # With sliding window
    processor = create_sliding_window_processor(
        token_counter=accurate_counter,
        trigger=("tokens", 100000),
    )
    ```

=== "Simple counter"

    ```python
    def simple_counter(messages):
        """Simple character-based estimation."""
        return sum(len(str(msg)) for msg in messages) // 4

    processor = create_sliding_window_processor(
        token_counter=simple_counter,
    )
    ```

## Custom Summary Prompt

Customize how summaries are generated (SummarizationProcessor only):

```python
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    summary_prompt="""
    You are summarizing an agent conversation. Extract:

    1. **Key Decisions**: What was decided?
    2. **Code Changes**: What code was written/modified?
    3. **Pending Tasks**: What still needs to be done?
    4. **Important Context**: What context is crucial to preserve?

    Conversation to summarize:
    {messages}

    Provide a concise summary that preserves essential information.
    """,
)
```

## Multiple Triggers

Combine multiple trigger conditions (OR logic):

=== "Summarization"

    ```python
    from pydantic_ai_summarization import SummarizationProcessor

    processor = SummarizationProcessor(
        model="openai:gpt-4.1",
        trigger=[
            ("messages", 50),      # Trigger at 50 messages
            ("tokens", 100000),    # OR at 100k tokens
            ("fraction", 0.8),     # OR at 80% capacity
        ],
        keep=("messages", 20),
        max_input_tokens=128000,   # Required for fraction
    )
    ```

=== "Sliding Window"

    ```python
    from pydantic_ai_summarization import SlidingWindowProcessor

    processor = SlidingWindowProcessor(
        trigger=[
            ("messages", 100),     # Trigger at 100 messages
            ("tokens", 50000),     # OR at 50k tokens
            ("fraction", 0.9),     # OR at 90% capacity
        ],
        keep=("messages", 30),
        max_input_tokens=128000,   # Required for fraction
    )
    ```

## Fraction-Based Configuration

Use fractions for adaptive behavior:

=== "Summarization"

    ```python
    from pydantic_ai_summarization import SummarizationProcessor

    processor = SummarizationProcessor(
        model="openai:gpt-4.1",
        trigger=("fraction", 0.75),  # Trigger at 75% of context
        keep=("fraction", 0.25),     # Keep last 25% of context
        max_input_tokens=128000,
    )
    ```

=== "Sliding Window"

    ```python
    from pydantic_ai_summarization import SlidingWindowProcessor

    processor = SlidingWindowProcessor(
        trigger=("fraction", 0.8),   # Trigger at 80% of context
        keep=("fraction", 0.3),      # Keep last 30% of context
        max_input_tokens=128000,
    )
    ```

## Different Models for Summarization

Use a smaller/faster model for summaries:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import SummarizationProcessor

# Use GPT-4o-mini for summaries (faster, cheaper)
processor = SummarizationProcessor(
    model="openai:gpt-4o-mini",
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

# Main agent uses GPT-4
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

## Disable Trimming

By default, input to summarization is trimmed. Disable this:

```python
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trim_tokens_to_summarize=None,  # No trimming
    ...
)
```

## Multiple Processors

Chain multiple history processors:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

def logging_processor(messages):
    """Log message counts."""
    print(f"Processing {len(messages)} messages")
    return messages

window = create_sliding_window_processor()

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[logging_processor, window],
)
```

## Token-Based Keep

Keep messages based on token count:

=== "Summarization"

    ```python
    from pydantic_ai_summarization import SummarizationProcessor

    processor = SummarizationProcessor(
        model="openai:gpt-4.1",
        trigger=("tokens", 100000),
        keep=("tokens", 20000),  # Keep ~20k tokens worth
    )
    ```

=== "Sliding Window"

    ```python
    from pydantic_ai_summarization import SlidingWindowProcessor

    processor = SlidingWindowProcessor(
        trigger=("tokens", 100000),
        keep=("tokens", 50000),  # Keep ~50k tokens worth
    )
    ```

## High-Throughput Configuration

For scenarios with many parallel conversations:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import SlidingWindowProcessor

# Aggressive trimming for high-throughput
processor = SlidingWindowProcessor(
    trigger=[
        ("messages", 30),
        ("tokens", 20000),
    ],
    keep=("messages", 10),
)

agent = Agent(
    "openai:gpt-4.1-mini",  # Fast, cheap model
    history_processors=[processor],
)
```

## Integration with pydantic-deep

Use with the full agent framework:

=== "Summarization"

    ```python
    from pydantic_deep import create_deep_agent
    from pydantic_ai_summarization import create_summarization_processor

    processor = create_summarization_processor(
        trigger=("tokens", 100000),
        keep=("messages", 20),
    )

    agent = create_deep_agent(
        model="openai:gpt-4.1",
        history_processors=[processor],
    )
    ```

=== "Sliding Window"

    ```python
    from pydantic_deep import create_deep_agent
    from pydantic_ai_summarization import create_sliding_window_processor

    processor = create_sliding_window_processor(
        trigger=("messages", 100),
        keep=("messages", 50),
    )

    agent = create_deep_agent(
        model="openai:gpt-4.1",
        history_processors=[processor],
    )
    ```

## Hybrid Approach

Use both processors for different scenarios:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import (
    create_summarization_processor,
    create_sliding_window_processor,
)

# For important conversations - use summarization
important_processor = create_summarization_processor(
    trigger=("tokens", 100000),
    keep=("messages", 30),
)

# For casual conversations - use sliding window
casual_processor = create_sliding_window_processor(
    trigger=("messages", 50),
    keep=("messages", 20),
)

# Choose processor based on context
def get_agent(is_important: bool):
    processor = important_processor if is_important else casual_processor
    return Agent(
        "openai:gpt-4.1",
        history_processors=[processor],
    )
```

## Presets for Common Use Cases

### Coding Assistant

```python
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("tokens", 100000),
    keep=("messages", 30),
    summary_prompt="""
    Summarize this coding session. Focus on:
    - What files were modified
    - Key code changes made
    - Bugs fixed or introduced
    - Pending tasks

    {messages}
    """,
)
```

### Customer Support Bot

```python
from pydantic_ai_summarization import SlidingWindowProcessor

processor = SlidingWindowProcessor(
    trigger=("messages", 30),
    keep=("messages", 15),
)
```

### Research Assistant

```python
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("tokens", 80000),
    keep=("tokens", 30000),
    summary_prompt="""
    Summarize this research session. Focus on:
    - Key findings and insights
    - Sources referenced
    - Questions answered
    - Open questions remaining

    {messages}
    """,
)
```
