# summarization-pydantic-ai

[![PyPI version](https://img.shields.io/pypi/v/summarization-pydantic-ai.svg)](https://pypi.org/project/summarization-pydantic-ai/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Coverage Status](https://coveralls.io/repos/github/vstorm-co/summarization-pydantic-ai/badge.svg?branch=main)](https://coveralls.io/github/vstorm-co/summarization-pydantic-ai?branch=main)

Automatic conversation summarization and context management for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents.

> **Looking for a complete agent framework?** Check out [pydantic-deep](https://github.com/vstorm-co/pydantic-deep) - a full-featured deep agent framework with planning, subagents, and skills system.

> **Need file operations?** Check out [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) - file storage and sandbox backends for AI agents.

## Documentation

**[Full Documentation](https://vstorm-co.github.io/summarization-pydantic-ai/)** - Installation, concepts, examples, and API reference.

## Installation

```bash
pip install summarization-pydantic-ai

# With tiktoken for accurate token counting
pip install summarization-pydantic-ai[tiktoken]
```

## Available Processors

This library provides two history processors for managing conversation context:

| Processor | LLM Cost | Latency | Context Preservation | Use Case |
|-----------|----------|---------|---------------------|----------|
| `SummarizationProcessor` | High | High | Intelligent summary | When context quality matters |
| `SlidingWindowProcessor` | Zero | ~0ms | Discards old messages | When speed/cost matters |

## Quick Start

### SummarizationProcessor - Intelligent Summarization

Uses an LLM to create intelligent summaries of older messages:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

# Create a processor that triggers at 100k tokens and keeps 20 messages
processor = create_summarization_processor(
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

# The processor will automatically summarize older messages
# when the conversation grows too long
result = await agent.run("Hello!")
```

### SlidingWindowProcessor - Zero-Cost Trimming

Simply discards old messages without LLM calls - fastest and cheapest option:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

# Keep last 50 messages when reaching 100
processor = create_sliding_window_processor(
    trigger=("messages", 100),
    keep=("messages", 50),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

# Old messages are simply discarded - no LLM cost
result = await agent.run("Hello!")
```

### Multiple Triggers

Both processors support triggering based on multiple conditions:

```python
from pydantic_ai_summarization import SummarizationProcessor, SlidingWindowProcessor

# Summarization with multiple triggers
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=[
        ("messages", 50),    # OR 50+ messages
        ("tokens", 100000),  # OR 100k+ tokens
    ],
    keep=("messages", 10),
)

# Sliding window with multiple triggers
processor = SlidingWindowProcessor(
    trigger=[
        ("messages", 100),
        ("tokens", 50000),
    ],
    keep=("messages", 30),
)
```

### Fraction-Based Configuration

Trigger when reaching a percentage of the model's context window:

```python
from pydantic_ai_summarization import SummarizationProcessor, SlidingWindowProcessor

# Summarization at 80% of context
processor = SummarizationProcessor(
    model="openai:gpt-4.1",
    trigger=("fraction", 0.8),  # 80% of context window
    keep=("fraction", 0.2),     # Keep last 20%
    max_input_tokens=128000,    # GPT-4's context window
)

# Sliding window at 80% of context
processor = SlidingWindowProcessor(
    trigger=("fraction", 0.8),
    keep=("fraction", 0.3),
    max_input_tokens=128000,
)
```

### Custom Token Counter

Use a custom token counting function with either processor:

```python
from pydantic_ai_summarization import (
    create_summarization_processor,
    create_sliding_window_processor,
)

def my_token_counter(messages):
    # Your custom token counting logic
    return sum(len(str(msg)) for msg in messages) // 4

# With summarization
processor = create_summarization_processor(
    token_counter=my_token_counter,
)

# With sliding window
processor = create_sliding_window_processor(
    token_counter=my_token_counter,
)
```

### Custom Summary Prompt

Customize how summaries are generated (SummarizationProcessor only):

```python
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    summary_prompt="""
    Extract the key information from this conversation.
    Focus on: decisions made, code written, and pending tasks.

    Conversation:
    {messages}
    """,
)
```

## Trigger Types

| Type | Example | Description |
|------|---------|-------------|
| `messages` | `("messages", 50)` | Trigger when message count exceeds threshold |
| `tokens` | `("tokens", 100000)` | Trigger when token count exceeds threshold |
| `fraction` | `("fraction", 0.8)` | Trigger at percentage of max_input_tokens |

## Keep Types

| Type | Example | Description |
|------|---------|-------------|
| `messages` | `("messages", 20)` | Keep last N messages after processing |
| `tokens` | `("tokens", 10000)` | Keep last N tokens worth of messages |
| `fraction` | `("fraction", 0.2)` | Keep last N% of max_input_tokens |

## How It Works

### SummarizationProcessor

1. **Monitoring**: Tracks token count on every call
2. **Trigger Check**: When any trigger condition is met, summarization begins
3. **Safe Cutoff**: Finds a safe point to cut that doesn't split tool call pairs
4. **Summarization**: Uses an LLM to generate a summary of older messages
5. **Replacement**: Older messages are replaced with a summary message

### SlidingWindowProcessor

1. **Monitoring**: Tracks message/token count on every call
2. **Trigger Check**: When any trigger condition is met, trimming begins
3. **Safe Cutoff**: Finds a safe point to cut that doesn't split tool call pairs
4. **Trimming**: Older messages are simply discarded (no LLM call)

## API Reference

### `SummarizationProcessor`

```python
@dataclass
class SummarizationProcessor:
    model: str                           # Model for generating summaries
    trigger: ContextSize | list[ContextSize] | None  # When to trigger
    keep: ContextSize                    # How much to keep
    token_counter: TokenCounter          # Token counting function
    summary_prompt: str                  # Prompt template
    max_input_tokens: int | None         # Required for fraction-based
    trim_tokens_to_summarize: int | None # Limit summary input
```

### `SlidingWindowProcessor`

```python
@dataclass
class SlidingWindowProcessor:
    trigger: ContextSize | list[ContextSize] | None  # When to trigger
    keep: ContextSize                    # How much to keep
    token_counter: TokenCounter          # Token counting function
    max_input_tokens: int | None         # Required for fraction-based
```

### Factory Functions

```python
# Summarization with defaults
def create_summarization_processor(
    model: str = "openai:gpt-4.1",
    trigger: ContextSize | list[ContextSize] | None = ("tokens", 170000),
    keep: ContextSize = ("messages", 20),
    max_input_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
    summary_prompt: str | None = None,
) -> SummarizationProcessor

# Sliding window with defaults
def create_sliding_window_processor(
    trigger: ContextSize | list[ContextSize] | None = ("messages", 100),
    keep: ContextSize = ("messages", 50),
    max_input_tokens: int | None = None,
    token_counter: TokenCounter | None = None,
) -> SlidingWindowProcessor
```

## Choosing a Processor

Use **SummarizationProcessor** when:
- Context quality is important
- You need to preserve key information from long conversations
- LLM cost is acceptable

Use **SlidingWindowProcessor** when:
- Speed and cost are priorities
- Recent context is most important
- You're running many parallel conversations
- You want deterministic, predictable behavior

## Development

```bash
git clone https://github.com/vstorm-co/summarization-pydantic-ai.git
cd summarization-pydantic-ai
make install
make test
```

## Related Projects

- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** - Agent framework by Pydantic
- **[pydantic-deep](https://github.com/vstorm-co/pydantic-deep)** - Full agent framework (uses this library)
- **[pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend)** - File storage and sandbox backends
- **[pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo)** - Task planning toolset

## License

MIT License - see [LICENSE](LICENSE) for details.
