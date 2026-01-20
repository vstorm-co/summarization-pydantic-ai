# Triggers

Triggers define when summarization should occur.

## Trigger Types

### Message-Based

Trigger when message count exceeds a threshold:

```python
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("messages", 50),  # Trigger at 50+ messages
    ...
)
```

### Token-Based

Trigger when token count exceeds a threshold:

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("tokens", 100000),  # Trigger at 100k+ tokens
    ...
)
```

### Fraction-Based

Trigger at a percentage of the model's context window:

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("fraction", 0.8),    # Trigger at 80% capacity
    max_input_tokens=128000,       # GPT-4's context window
    ...
)
```

!!! warning "Required Parameter"
    Fraction-based triggers require `max_input_tokens` to be set.

## Multiple Triggers

Combine multiple triggers with OR logic:

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=[
        ("messages", 50),     # OR
        ("tokens", 100000),   # OR
        ("fraction", 0.8),
    ],
    max_input_tokens=128000,
    ...
)
```

## Keep Configuration

The `keep` parameter uses the same format as triggers:

```python
# Keep last 20 messages
keep=("messages", 20)

# Keep last 10k tokens
keep=("tokens", 10000)

# Keep last 20% of context
keep=("fraction", 0.2)
```

## Common Configurations

### Conservative (Long Conversations)

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("tokens", 150000),
    keep=("messages", 30),
)
```

### Aggressive (Short Context Models)

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("tokens", 30000),
    keep=("messages", 10),
)
```

### Fraction-Based (Adaptive)

```python
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("fraction", 0.75),
    keep=("fraction", 0.25),
    max_input_tokens=128000,
)
```
