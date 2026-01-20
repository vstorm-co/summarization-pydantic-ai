# SlidingWindowProcessor

The `SlidingWindowProcessor` is a zero-cost history processor that simply discards old messages when thresholds are reached.

## Why Use Sliding Window?

| Advantage | Description |
|-----------|-------------|
| **Zero LLM Cost** | No API calls needed for processing |
| **Instant** | Near-zero latency - just array slicing |
| **Deterministic** | Predictable, reproducible behavior |
| **Simple** | Easy to understand and debug |

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import SlidingWindowProcessor

processor = SlidingWindowProcessor(
    trigger=("messages", 100),
    keep=("messages", 50),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

# When conversation reaches 100 messages,
# old messages are discarded, keeping last 50
result = await agent.run("Hello!")
```

## Configuration

### Trigger

When to trigger window trimming. See [Triggers](triggers.md) for details:

```python
# Single trigger - by message count
processor = SlidingWindowProcessor(
    trigger=("messages", 100),
    ...
)

# Single trigger - by token count
processor = SlidingWindowProcessor(
    trigger=("tokens", 50000),
    ...
)

# Multiple triggers (OR logic)
processor = SlidingWindowProcessor(
    trigger=[
        ("messages", 100),
        ("tokens", 50000),
    ],
    ...
)
```

### Keep

How much context to retain after trimming:

```python
# Keep last 50 messages
processor = SlidingWindowProcessor(
    keep=("messages", 50),
    ...
)

# Keep last 25k tokens worth
processor = SlidingWindowProcessor(
    keep=("tokens", 25000),
    ...
)

# Keep last 30% of context window
processor = SlidingWindowProcessor(
    keep=("fraction", 0.3),
    max_input_tokens=128000,
)
```

### Token Counter

Custom function for counting tokens:

```python
def my_counter(messages):
    # Your logic here
    return token_count

processor = SlidingWindowProcessor(
    token_counter=my_counter,
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

Use `create_sliding_window_processor()` for common configurations:

```python
from pydantic_ai_summarization import create_sliding_window_processor

# With defaults (trigger at 100 messages, keep 50)
processor = create_sliding_window_processor()

# With custom settings
processor = create_sliding_window_processor(
    trigger=("messages", 60),
    keep=("messages", 30),
)

# Token-based
processor = create_sliding_window_processor(
    trigger=("tokens", 100000),
    keep=("tokens", 50000),
)
```

## Comparison with SummarizationProcessor

| Aspect | SlidingWindowProcessor | SummarizationProcessor |
|--------|----------------------|----------------------|
| **Cost** | Zero | LLM API call |
| **Latency** | ~0ms | Depends on model |
| **Context Loss** | Complete (old messages gone) | Minimal (summarized) |
| **Determinism** | Fully deterministic | LLM-dependent |
| **Complexity** | Simple | More complex |
| **Best For** | Speed, cost, simplicity | Context quality |

## Example: Chat Application

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

# Simple chatbot that keeps recent context only
processor = create_sliding_window_processor(
    trigger=("messages", 50),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    system_prompt="You are a helpful assistant.",
    history_processors=[processor],
)

async def chat(user_input: str, history: list):
    result = await agent.run(user_input, message_history=history)
    return result.output, result.all_messages()
```

## Example: High-Throughput Service

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import SlidingWindowProcessor

# Aggressive trimming for high-throughput scenarios
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

## Example: Token-Based Window

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

# Keep context within token budget
processor = create_sliding_window_processor(
    trigger=("tokens", 100000),
    keep=("tokens", 50000),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

## When to Use

**Use SlidingWindowProcessor when:**

- Speed is critical
- You want to minimize costs
- Recent context is most important
- Running many parallel conversations
- Building simple chatbots
- Deterministic behavior is required

**Consider SummarizationProcessor instead when:**

- Context quality is critical
- You need to preserve key information
- Building coding assistants or complex agents
- LLM cost is acceptable
