# Core Concepts

summarization-pydantic-ai provides automatic conversation context management for pydantic-ai agents.

## Overview

When agent conversations grow long, they can exceed the model's context window. This library solves that by providing two strategies:

| Strategy | Description | Cost | Speed |
|----------|-------------|------|-------|
| [SummarizationProcessor](processor.md) | Uses LLM to summarize old messages | High | Slow |
| [SlidingWindowProcessor](sliding-window.md) | Simply discards old messages | Zero | Instant |

Both processors:

1. **Monitor** conversation length (messages or tokens)
2. **Trigger** processing when thresholds are reached
3. **Find safe cutoff** that preserves tool call pairs
4. **Process** older messages (summarize or discard)
5. **Preserve** recent messages for context continuity

## Key Components

### SummarizationProcessor

Intelligent summarization using an LLM:

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

### SlidingWindowProcessor

Zero-cost trimming without LLM:

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
```

### Factory Functions

Convenience functions with sensible defaults:

```python
from pydantic_ai_summarization import (
    create_summarization_processor,
    create_sliding_window_processor,
)

# Summarization with defaults
summarizer = create_summarization_processor(
    trigger=("messages", 50),
    keep=("messages", 10),
)

# Sliding window with defaults
window = create_sliding_window_processor(
    trigger=("messages", 100),
    keep=("messages", 50),
)
```

## How SummarizationProcessor Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Conversation                        │
├─────────────────────────────────────────────────────────────┤
│  Message 1   │  Message 2   │  ...  │  Message N-1  │ Msg N │
└──────────────┴──────────────┴───────┴───────────────┴───────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 SummarizationProcessor                       │
├─────────────────────────────────────────────────────────────┤
│  1. Count tokens                                            │
│  2. Check triggers (messages/tokens/fraction)               │
│  3. Find safe cutoff point                                  │
│  4. Generate summary via LLM                                │
│  5. Replace old messages with summary                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processed Messages                        │
├─────────────────────────────────────────────────────────────┤
│  Summary Message  │  Message N-19  │  ...  │  Message N     │
└───────────────────┴────────────────┴───────┴────────────────┘
```

## How SlidingWindowProcessor Works

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Conversation                        │
├─────────────────────────────────────────────────────────────┤
│  Message 1   │  Message 2   │  ...  │  Message N-1  │ Msg N │
└──────────────┴──────────────┴───────┴───────────────┴───────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                 SlidingWindowProcessor                       │
├─────────────────────────────────────────────────────────────┤
│  1. Count messages/tokens                                   │
│  2. Check triggers                                          │
│  3. Find safe cutoff point                                  │
│  4. Discard old messages (no LLM call)                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Processed Messages                        │
├─────────────────────────────────────────────────────────────┤
│  Message N-49  │  Message N-48  │  ...  │  Message N        │
└────────────────┴────────────────┴───────┴───────────────────┘
```

## Tool Call Safety

Both processors ensure tool call/response pairs are never split:

```
❌ Bad cutoff (splits pair):
[Tool Call: search] | [Tool Result: found 5 items] [User: thanks]
                    ↑ cutoff here

✅ Good cutoff (preserves pair):
[User: find items] | [Tool Call: search] [Tool Result: found 5 items]
                   ↑ cutoff here
```

## Choosing a Processor

| Requirement | Recommended |
|-------------|-------------|
| Context quality is critical | SummarizationProcessor |
| Speed and cost are priorities | SlidingWindowProcessor |
| Running many parallel conversations | SlidingWindowProcessor |
| Long-running coding sessions | SummarizationProcessor |
| Simple chatbots | SlidingWindowProcessor |
| Deterministic behavior needed | SlidingWindowProcessor |

## Next Steps

- Learn about [Triggers](triggers.md)
- See the [SummarizationProcessor](processor.md) details
- See the [SlidingWindowProcessor](sliding-window.md) details
- View [Examples](../examples/index.md)
