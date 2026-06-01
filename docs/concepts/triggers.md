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

!!! warning "`max_input_tokens` is also required for fraction-based `keep`"
    The `max_input_tokens` requirement is not limited to fraction *triggers*. Using
    `keep=("fraction", ...)` also requires `max_input_tokens` — validation rejects any
    fraction-based trigger **or** keep value when `max_input_tokens` is unset (and it must
    be greater than 0).

## Preserving the Head (Sliding Window)

[`SlidingWindowProcessor`][pydantic_ai_summarization.sliding_window.SlidingWindowProcessor]
(and [`SlidingWindowCapability`][pydantic_ai_summarization.capability.SlidingWindowCapability])
support an optional `keep_head` parameter alongside `keep`. While `keep` retains messages
from the **tail**, `keep_head` retains messages from the **start** of the conversation —
useful for preserving a system prompt or initial instructions that should always survive
trimming:

```python
from pydantic_ai_summarization import SlidingWindowProcessor

processor = SlidingWindowProcessor(
    trigger=("messages", 100),
    keep=("messages", 50),       # keep last 50 messages
    keep_head=("messages", 1),   # always preserve the first message (system prompt)
)
```

`keep_head` accepts the same `("messages", n)`, `("tokens", n)`, or `("fraction", f)` forms.
A fraction-based `keep_head` likewise requires `max_input_tokens`.

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
