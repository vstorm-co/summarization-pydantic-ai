# SummarizationProcessor

The `SummarizationProcessor` is the core component that handles conversation summarization.

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

## Configuration

### Model

The model used to generate summaries:

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",  # Any pydantic-ai supported model
    ...
)
```

### Trigger

When to trigger summarization. See [Triggers](triggers.md) for details:

```python
# Single trigger
processor = SummarizationProcessor(
    trigger=("tokens", 100000),
    ...
)

# Multiple triggers (OR logic)
processor = SummarizationProcessor(
    trigger=[
        ("messages", 50),
        ("tokens", 100000),
    ],
    ...
)
```

### Keep

How much context to retain after summarization:

```python
# Keep last 20 messages
processor = SummarizationProcessor(
    keep=("messages", 20),
    ...
)

# Keep last 10k tokens worth
processor = SummarizationProcessor(
    keep=("tokens", 10000),
    ...
)
```

### Token Counter

Custom function for counting tokens:

```python
def my_counter(messages):
    # Your logic here
    return token_count

processor = SummarizationProcessor(
    token_counter=my_counter,
    ...
)
```

### Summary Prompt

Custom prompt for generating summaries:

```python
processor = SummarizationProcessor(
    summary_prompt="""
    Summarize this conversation, focusing on:
    - Key decisions made
    - Important code changes
    - Pending tasks

    {messages}
    """,
    ...
)
```

## Tool Call Safety

The processor ensures tool call/response pairs are never split:

```
❌ Bad cutoff (splits pair):
[Tool Call: search] | [Tool Result: found 5 items] [User: thanks]
                    ↑ cutoff here

✅ Good cutoff (preserves pair):
[User: find items] | [Tool Call: search] [Tool Result: found 5 items]
                   ↑ cutoff here
```

## Factory Function

Use `create_summarization_processor()` for common configurations:

```python
from pydantic_ai_summarization import create_summarization_processor

# With defaults
processor = create_summarization_processor()

# With custom settings
processor = create_summarization_processor(
    model="openai:gpt-4",
    trigger=("messages", 50),
    keep=("messages", 10),
)
```
