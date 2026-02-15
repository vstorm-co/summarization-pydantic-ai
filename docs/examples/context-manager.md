# Context Manager Examples

Practical examples for using the `ContextManagerMiddleware`.

!!! note "Prerequisite"
    All examples on this page require the `hybrid` extra:

    ```bash
    pip install summarization-pydantic-ai[hybrid]
    ```

## Basic Usage

The simplest way to add real-time context management:

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

# Create middleware with defaults (200k token budget, compress at 90%)
middleware = create_context_manager_middleware()

# Register as history processor for token tracking + compression
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[middleware],
)

# Wrap with MiddlewareAgent for tool output truncation
wrapped = MiddlewareAgent(agent, middleware=[middleware])

async def main():
    result = await wrapped.run("Hello!")
    print(result.output)
```

## Custom Token Budget

Configure for a specific model's context window:

```python
from pydantic_ai_summarization import create_context_manager_middleware

# For Claude with 200K context
middleware = create_context_manager_middleware(
    max_tokens=200_000,
    compress_threshold=0.85,    # Compress at 85% (170K tokens)
    keep=("messages", 30),      # Keep last 30 messages after compression
)

# For GPT-4 with 128K context
middleware = create_context_manager_middleware(
    max_tokens=128_000,
    compress_threshold=0.9,     # Compress at 90% (115K tokens)
    keep=("messages", 20),      # Keep last 20 messages after compression
)
```

## Real-Time Usage Monitoring

Track token usage with a callback to display in a UI or log:

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

def on_usage(percentage: float, current: int, maximum: int) -> None:
    bar_length = 30
    filled = int(bar_length * percentage)
    bar = "=" * filled + "-" * (bar_length - filled)
    print(f"[{bar}] {percentage:.0%} ({current:,} / {maximum:,} tokens)")

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    on_usage_update=on_usage,
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[middleware],
)
wrapped = MiddlewareAgent(agent, middleware=[middleware])

async def chat():
    message_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        result = await wrapped.run(
            user_input,
            message_history=message_history,
        )

        print(f"Assistant: {result.output}")
        message_history = result.all_messages()

if __name__ == "__main__":
    import asyncio
    asyncio.run(chat())
```

## Async Usage Callback

Use an async callback to persist usage data:

```python
from pydantic_ai_summarization import create_context_manager_middleware

async def save_usage(percentage: float, current: int, maximum: int) -> None:
    """Save usage stats to a database or monitoring service."""
    await db.insert({
        "usage_pct": percentage,
        "tokens_used": current,
        "tokens_max": maximum,
    })

middleware = create_context_manager_middleware(
    on_usage_update=save_usage,
)
```

## Tool Output Truncation

Prevent file reads and other large tool outputs from bloating the conversation:

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    max_tool_output_tokens=2000,   # Truncate tool outputs over ~2000 tokens
    tool_output_head_lines=10,     # Show first 10 lines
    tool_output_tail_lines=10,     # Show last 10 lines
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[middleware],
)
wrapped = MiddlewareAgent(agent, middleware=[middleware])

@agent.tool_plain
def read_file(path: str) -> str:
    """Read a file from disk."""
    with open(path) as f:
        return f.read()

# If read_file returns a 10,000-line file, the middleware will
# truncate it to the first 10 + last 10 lines automatically
```

## Integration with pydantic-deep

Use the context manager middleware with the full agent framework:

```python
from pydantic_deep import create_deep_agent, DeepAgentDeps
from pydantic_ai_backends import LocalBackend
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    compress_threshold=0.9,
    keep=("messages", 20),
    max_tool_output_tokens=2000,
    on_usage_update=lambda pct, cur, mx: print(f"Usage: {pct:.0%}"),
)

agent = create_deep_agent(
    model="openai:gpt-4.1",
    history_processors=[middleware],
)

# Wrap for tool output interception
wrapped = MiddlewareAgent(agent, middleware=[middleware])

deps = DeepAgentDeps(
    backend=LocalBackend(root_dir="/home/user/project"),
)

async def main():
    result = await wrapped.run("Analyze this project", deps=deps)
    print(result.output)
    print(f"Compressions triggered: {middleware.compression_count}")
```

## Hybrid Mode: All Three Together

Combine the `ContextManagerMiddleware` with standalone processors for layered context management:

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import (
    create_context_manager_middleware,
    create_sliding_window_processor,
    create_summarization_processor,
)

# Layer 1: Sliding window as a safety net (cheap, fast)
window = create_sliding_window_processor(
    trigger=("messages", 200),
    keep=("messages", 100),
)

# Layer 2: Summarization for intelligent compression
summarizer = create_summarization_processor(
    trigger=("tokens", 150_000),
    keep=("messages", 30),
)

# Layer 3: Context manager for real-time tracking + tool truncation
middleware = create_context_manager_middleware(
    max_tokens=200_000,
    compress_threshold=0.95,     # Only compress if others didn't catch it
    keep=("messages", 20),
    max_tool_output_tokens=2000,
    on_usage_update=lambda pct, cur, mx: print(f"Usage: {pct:.0%}"),
)

# Processors run in order: window -> summarizer -> middleware
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[window, summarizer, middleware],
)

# Wrap for tool output interception via middleware protocol
wrapped = MiddlewareAgent(agent, middleware=[middleware])
```

In this setup:

1. **Sliding window** acts first as a hard cap on message count (zero cost).
2. **Summarization** triggers at a token threshold, creating an intelligent summary.
3. **Context manager middleware** provides the final safety net with real-time tracking, and handles tool output truncation via the middleware protocol.

## Custom Summarization Model

Use a different model for generating compression summaries:

```python
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    summarization_model="openai:gpt-4.1-mini",  # Fast, cheap model for summaries
)
```

## Fraction-Based Keep

Use a fraction of the model's context window for retention:

```python
from pydantic_ai_summarization import ContextManagerMiddleware

middleware = ContextManagerMiddleware(
    max_tokens=128_000,
    compress_threshold=0.9,
    keep=("fraction", 0.25),       # Keep last 25% of context window
    max_input_tokens=128_000,      # Required for fraction-based keep
)
```

!!! warning "Required Parameter"
    Fraction-based `keep` requires `max_input_tokens` to be set.
