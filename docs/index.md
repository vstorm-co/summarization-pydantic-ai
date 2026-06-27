<p align="center">
  <img src="assets/social-preview.png" alt="Summarization for Pydantic AI" width="100%">
</p>

<h1 align="center">Summarization for Pydantic AI</h1>

<p align="center"><em>Unlimited context for long-running agents.</em></p>

<p align="center">
  <a href="https://pypi.org/project/summarization-pydantic-ai/"><img src="https://img.shields.io/pypi/v/summarization-pydantic-ai.svg" alt="PyPI version"></a>
  <a href="https://pepy.tech/projects/summarization-pydantic-ai"><img src="https://static.pepy.tech/badge/summarization-pydantic-ai/month" alt="PyPI Downloads"></a>
  <a href="https://github.com/vstorm-co/summarization-pydantic-ai/stargazers"><img src="https://img.shields.io/github/stars/vstorm-co/summarization-pydantic-ai?style=flat&logo=github&color=yellow" alt="GitHub Stars"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://coveralls.io/github/vstorm-co/summarization-pydantic-ai?branch=main"><img src="https://coveralls.io/repos/github/vstorm-co/summarization-pydantic-ai/badge.svg?branch=main" alt="Coverage Status"></a>
  <a href="https://github.com/vstorm-co/summarization-pydantic-ai/actions/workflows/ci.yml"><img src="https://github.com/vstorm-co/summarization-pydantic-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

---

!!! tip "Part of Pydantic Deep Agents"
    **Summarization for Pydantic AI** is one library in [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) — the open-source
    Claude Code alternative & Python agent framework. Use it standalone, or get every
    library wired together in a single `create_deep_agent()` call.

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
| [`ContextManagerCapability`][pydantic_ai_summarization.capability.ContextManagerCapability] | Capability | Per compression | Production apps (recommended) |
| [`SummarizationCapability`][pydantic_ai_summarization.capability.SummarizationCapability] | Capability | High | Quality-focused apps |
| [`SlidingWindowCapability`][pydantic_ai_summarization.capability.SlidingWindowCapability] | Capability | Zero | Speed/cost-focused apps |
| [`LimitWarnerCapability`][pydantic_ai_summarization.capability.LimitWarnerCapability] | Capability | Zero | Warning before limits hit |
| [`SummarizationProcessor`][pydantic_ai_summarization.processor.SummarizationProcessor] | Processor | High | Standalone use |
| [`SlidingWindowProcessor`][pydantic_ai_summarization.sliding_window.SlidingWindowProcessor] | Processor | Zero | Standalone use |
| [`LimitWarnerProcessor`][pydantic_ai_summarization.limit_warner.LimitWarnerProcessor] | Processor | Zero | Standalone use |

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
