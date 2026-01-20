# summarization-pydantic-ai

Automatic conversation summarization and context management for [pydantic-ai](https://github.com/pydantic/pydantic-ai) agents.

## Why Use This Library?

Long-running agent conversations can exceed model context windows. This library provides:

- **Two Strategies** - Choose between intelligent summarization or simple sliding window
- **Flexible Triggers** - Configure when to process based on messages, tokens, or fractions
- **Safe Cutoff** - Preserves tool call/response pairs to maintain conversation integrity
- **Zero Dependencies** - Only requires pydantic-ai

## Available Processors

| Processor | LLM Cost | Latency | Context Preservation | Best For |
|-----------|----------|---------|---------------------|----------|
| [`SummarizationProcessor`](concepts/processor.md) | High | High | Intelligent summary | Quality-focused apps |
| [`SlidingWindowProcessor`](concepts/sliding-window.md) | Zero | ~0ms | Discards old messages | Speed/cost-focused apps |

## Quick Start

### Intelligent Summarization

Uses an LLM to create summaries of older messages:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

# Create a processor that triggers at 100k tokens
processor = create_summarization_processor(
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

# The processor will automatically summarize older messages
result = await agent.run("Hello!")
```

### Zero-Cost Sliding Window

Simply discards old messages - no LLM calls needed:

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

# Old messages are simply discarded - instant, free
result = await agent.run("Hello!")
```

## Features

| Feature | SummarizationProcessor | SlidingWindowProcessor |
|---------|----------------------|----------------------|
| **Message Triggers** | :material-check: | :material-check: |
| **Token Triggers** | :material-check: | :material-check: |
| **Fraction Triggers** | :material-check: | :material-check: |
| **Multiple Triggers** | :material-check: | :material-check: |
| **Custom Counters** | :material-check: | :material-check: |
| **Tool Safety** | :material-check: | :material-check: |
| **Custom Prompts** | :material-check: | :material-close: |
| **Zero LLM Cost** | :material-close: | :material-check: |

## Choosing a Processor

Use **SummarizationProcessor** when:

- Context quality is critical
- You need to preserve key information from long conversations
- LLM cost is acceptable for your use case

Use **SlidingWindowProcessor** when:

- Speed and cost are priorities
- Recent context is most important
- You're running many parallel conversations
- You want deterministic, predictable behavior

## Related Projects

- **[pydantic-ai](https://github.com/pydantic/pydantic-ai)** - Agent framework by Pydantic
- **[pydantic-deep](https://github.com/vstorm-co/pydantic-deep)** - Full agent framework
- **[pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend)** - File storage backends
- **[pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo)** - Task planning toolset

## Next Steps

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)** - Get started with installation
- :material-book-open-variant: **[Concepts](concepts/index.md)** - Learn how it works
- :material-code-tags: **[Examples](examples/index.md)** - See it in action
- :material-api: **[API Reference](api/index.md)** - Full API documentation

</div>
