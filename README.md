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
  <b>Limit Warnings</b> — finish-soon guidance before hard caps
  &nbsp;&bull;&nbsp;
  <b>Context Manager</b> — real-time token tracking + tool truncation
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

For real-time token tracking and tool output truncation:

```bash
pip install summarization-pydantic-ai[hybrid]
```

## Quick Start — Capabilities (Recommended)

The recommended way to add context management is via pydantic-ai's native [Capabilities API](https://ai.pydantic.dev/capabilities/):

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import ContextManagerCapability

agent = Agent(
    "openai:gpt-4.1",
    capabilities=[ContextManagerCapability(max_tokens=100_000)],
)

result = await agent.run("Hello!")
```

**That's it.** Your agent now:

- Tracks token usage on every turn
- Auto-compresses when approaching the limit (90% by default)
- Truncates large tool outputs
- Auto-detects context window size from the model
- Preserves tool call/response pairs (never breaks them)

### Combine with Limit Warnings

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

### Alternative: Processor API

For standalone use without capabilities:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    trigger=("tokens", 100000),
    keep=("messages", 20),
)

agent = Agent("openai:gpt-4.1", history_processors=[processor])
```

## Available Processors

| Processor | LLM Cost | Latency | Context Preservation |
|-----------|----------|---------|---------------------|
| `SummarizationProcessor` | High | High | Intelligent summary |
| `SlidingWindowProcessor` | Zero | ~0ms | Discards old messages |
| `LimitWarnerProcessor` | Zero | ~0ms | Full history + warning injection |
| `ContextManagerMiddleware` | Per compression | Low tracking / High compression | Intelligent summary |

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

### Limit Warnings

Warn the agent before requests, context usage, or total tokens hit a cap:

```python
from pydantic_ai_summarization import create_limit_warner_processor

processor = create_limit_warner_processor(
    max_iterations=40,
    max_context_tokens=100000,
    max_total_tokens=200000,
)
```

### Real-Time Context Manager

Dual-protocol middleware combining token tracking, auto-compression, message persistence, and tool output truncation:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1",      # auto-detect max_tokens from genai-prices
    compress_threshold=0.9,
    messages_path="messages.json",     # persist all messages
    on_usage_update=lambda pct, cur, mx: print(f"{pct:.0%} used ({cur:,}/{mx:,})"),
    on_after_compress=lambda msgs: "Re-inject critical instructions here",
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[middleware],
)
```

Requires `pip install summarization-pydantic-ai[hybrid]`

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

### Custom Model (e.g., Azure OpenAI)

```python
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_summarization import create_summarization_processor

azure_model = OpenAIModel(
    "gpt-4o",
    provider=OpenAIProvider(
        base_url="https://my-resource.openai.azure.com/openai/deployments/gpt-4o",
        api_key="your-azure-api-key",
    ),
)

processor = create_summarization_processor(
    model=azure_model,
    trigger=("tokens", 100000),
    keep=("messages", 20),
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
| **Auto max_tokens** | Auto-detect context window from genai-prices |
| **Message Persistence** | Save all messages to JSON for session resume |
| **Guided Compaction** | Focus summaries on specific topics |
| **Callbacks** | on_before/after_compress with instruction re-injection |
| **Async Token Counting** | Sync or async token counter support |
| **Token Tracking** | Real-time usage monitoring with callbacks |
| **Tool Truncation** | Automatic truncation of large tool outputs |
| **Custom Models** | Use any pydantic-ai Model (Azure, custom providers) |
| **Lightweight** | Only requires pydantic-ai-slim (no extra model SDKs) |

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

---

<div align="center">

### Need help implementing this in your company?

<p>We're <a href="https://vstorm.co"><b>Vstorm</b></a> — an Applied Agentic AI Engineering Consultancy<br>with 30+ production AI agent implementations.</p>

<a href="https://vstorm.co/contact-us/">
  <img src="https://img.shields.io/badge/Talk%20to%20us%20%E2%86%92-0066FF?style=for-the-badge&logoColor=white" alt="Talk to us">
</a>

<br><br>

Made with ❤️ by <a href="https://vstorm.co"><b>Vstorm</b></a>

</div>
