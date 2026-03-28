# Context Management for Pydantic AI

Automatic conversation summarization and context management for [Pydantic AI](https://ai.pydantic.dev/) agents.

---

**Context Management for Pydantic AI** helps your agents handle long conversations without exceeding model context limits. Choose between intelligent LLM summarization or fast sliding window trimming.

<div class="grid cards" markdown>

- :material-brain: **Intelligent Summarization**

    LLM-powered compression that preserves key information

- :material-speedometer: **Sliding Window**

    Zero-cost message trimming for maximum speed

- :material-shield-check: **Safe Cutoff**

    Never breaks tool call/response pairs

- :material-tune: **Flexible Configuration**

    Message, token, or fraction-based triggers

</div>

## Quick Start — Capabilities (Recommended)

The recommended way to add context management:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import ContextManagerCapability

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[ContextManagerCapability(max_tokens=100_000)],
)
```

Combine with limit warnings:

```python
from pydantic_ai_summarization import ContextManagerCapability, LimitWarnerCapability

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        LimitWarnerCapability(max_iterations=40, max_context_tokens=100_000),
        ContextManagerCapability(max_tokens=100_000),
    ],
)
```

## Available Options

| Option | Type | LLM Cost | Best For |
|--------|------|----------|----------|
| `ContextManagerCapability` | Capability | Per compression | Production apps (recommended) |
| `SummarizationCapability` | Capability | High | Quality-focused apps |
| `SlidingWindowCapability` | Capability | Zero | Speed/cost-focused apps |
| `LimitWarnerCapability` | Capability | Zero | Warning before limits hit |
| `SummarizationProcessor` | Processor | High | Standalone use |
| `SlidingWindowProcessor` | Processor | Zero | Standalone use |
| `LimitWarnerProcessor` | Processor | Zero | Standalone use |

## Alternative: Processor API

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4o",
    history_processors=[processor],
)

result = await agent.run("Hello!")
```

### Zero-Cost Sliding Window

Simply discards old messages — no LLM calls:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

processor = create_sliding_window_processor(
    trigger=("messages", 100),
    keep=("messages", 50),
)

agent = Agent(
    "openai:gpt-4o",
    history_processors=[processor],
)

result = await agent.run("Hello!")
```

## Choosing a Processor

**Use SummarizationProcessor when:**

- Context quality is critical
- You need to preserve key information from long conversations
- LLM cost is acceptable for your use case

**Use SlidingWindowProcessor when:**

- Speed and cost are priorities
- Recent context is most important
- You're running many parallel conversations
- You want deterministic, predictable behavior

## Related Projects

| Package | Description |
|---------|-------------|
| [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) | Full agent framework (uses this library) |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage and Docker sandbox |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning toolset |
| [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) | Multi-agent orchestration |
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation — agent framework by Pydantic |

## Next Steps

<div class="grid cards" markdown>

- :material-download: **[Installation](installation.md)**

    Get started with pip or uv

- :material-book-open-variant: **[Concepts](concepts/index.md)**

    Learn how processors work

- :material-code-tags: **[Examples](examples/index.md)**

    See practical usage patterns

- :material-api: **[API Reference](api/index.md)**

    Full API documentation

</div>
