<h1 align="center">Context Management for Pydantic AI</h1>

<p align="center">
  <em>Automatic Conversation Summarization and History Management</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/summarization-pydantic-ai/"><img src="https://img.shields.io/pypi/v/summarization-pydantic-ai.svg" alt="PyPI version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://github.com/vstorm-co/summarization-pydantic-ai/actions/workflows/ci.yml"><img src="https://github.com/vstorm-co/summarization-pydantic-ai/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/pydantic/pydantic-ai"><img src="https://img.shields.io/badge/Powered%20by-Pydantic%20AI-E92063?logo=pydantic&logoColor=white" alt="Pydantic AI"></a>
</p>

<p align="center">
  <b>Intelligent Summarization</b> — LLM-powered context compression
  &nbsp;&bull;&nbsp;
  <b>Sliding Window</b> — zero-cost message trimming
  &nbsp;&bull;&nbsp;
  <b>Safe Cutoff</b> — preserves tool call pairs
</p>

---

**Context Management for Pydantic AI** helps your [Pydantic AI](https://ai.pydantic.dev/) agents handle long conversations without exceeding model context limits. Choose between intelligent LLM summarization or fast sliding window trimming.

> **Full framework?** Check out [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) — complete agent framework with planning, filesystem, subagents, and skills.

## Use Cases

| What You Want to Build | How This Library Helps |
|------------------------|------------------------|
| **Long-Running Agent** | Automatically compress history when context fills up |
| **Customer Support Bot** | Preserve key details while discarding routine exchanges |
| **Code Assistant** | Keep recent code context, summarize older discussions |
| **High-Throughput App** | Zero-cost sliding window for maximum speed |
| **Cost-Sensitive App** | Choose between quality (summarization) or free (sliding window) |

## Installation

```bash
pip install summarization-pydantic-ai
```

Or with uv:

```bash
uv add summarization-pydantic-ai
```

For accurate token counting:

```bash
pip install summarization-pydantic-ai[tiktoken]
```

## Quick Start

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

**That's it.** Your agent now:

- Monitors conversation size on every turn
- Summarizes older messages when limits are reached
- Preserves tool call/response pairs (never breaks them)
- Keeps recent context intact

## Available Processors

| Processor | LLM Cost | Latency | Context Preservation |
|-----------|----------|---------|---------------------|
| `SummarizationProcessor` | High | High | Intelligent summary |
| `SlidingWindowProcessor` | Zero | ~0ms | Discards old messages |

### Intelligent Summarization

Uses an LLM to create summaries of older messages:

```python
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    trigger=("tokens", 100000),  # When to summarize
    keep=("messages", 20),       # What to keep
)
```

### Zero-Cost Sliding Window

Simply discards old messages — no LLM calls:

```python
from pydantic_ai_summarization import create_sliding_window_processor

processor = create_sliding_window_processor(
    trigger=("messages", 100),  # When to trim
    keep=("messages", 50),      # What to keep
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
| `messages` | `("messages", 20)` | Keep last N messages |
| `tokens` | `("tokens", 10000)` | Keep last N tokens worth |
| `fraction` | `("fraction", 0.2)` | Keep last N% of context |

## Advanced Configuration

### Multiple Triggers

```python
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4o",
    trigger=[
        ("messages", 50),    # OR 50+ messages
        ("tokens", 100000),  # OR 100k+ tokens
    ],
    keep=("messages", 10),
)
```

### Fraction-Based

```python
processor = SummarizationProcessor(
    model="openai:gpt-4o",
    trigger=("fraction", 0.8),  # 80% of context window
    keep=("fraction", 0.2),     # Keep last 20%
    max_input_tokens=128000,    # GPT-4's context window
)
```

### Custom Token Counter

```python
def my_token_counter(messages):
    return sum(len(str(msg)) for msg in messages) // 4

processor = create_summarization_processor(
    token_counter=my_token_counter,
)
```

### Custom Summary Prompt

```python
processor = create_summarization_processor(
    summary_prompt="""
    Extract key information from this conversation.
    Focus on: decisions made, code written, pending tasks.

    Conversation:
    {messages}
    """,
)
```

## Why Choose This Library?

| Feature | Description |
|---------|-------------|
| **Two Strategies** | Intelligent summarization or fast sliding window |
| **Flexible Triggers** | Message count, token count, or fraction-based |
| **Safe Cutoff** | Never breaks tool call/response pairs |
| **Custom Counters** | Bring your own token counting logic |
| **Custom Prompts** | Control how summaries are generated |
| **Zero Dependencies** | Only requires pydantic-ai |

## Related Projects

| Package | Description |
|---------|-------------|
| [Pydantic Deep Agents](https://github.com/vstorm-co/pydantic-deepagents) | Full agent framework (uses this library) |
| [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) | File storage and Docker sandbox |
| [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) | Task planning toolset |
| [subagents-pydantic-ai](https://github.com/vstorm-co/subagents-pydantic-ai) | Multi-agent orchestration |
| [pydantic-ai](https://github.com/pydantic/pydantic-ai) | The foundation — agent framework by Pydantic |

## Contributing

```bash
git clone https://github.com/vstorm-co/summarization-pydantic-ai.git
cd summarization-pydantic-ai
make install
make test  # 100% coverage required
```

## License

MIT — see [LICENSE](LICENSE)

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/vstorm-co">vstorm-co</a></sub>
</p>
