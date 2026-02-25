# Context Manager Examples

Practical examples for using the `ContextManagerMiddleware`.

!!! note "Prerequisite"
    All examples on this page require the `hybrid` extra:

    ```bash
    pip install summarization-pydantic-ai[hybrid]
    ```

## Runnable Examples

The `examples/` directory contains 6 ready-to-run scripts:

| # | File | What it shows |
|---|------|---------------|
| 01 | `01_basic_context_manager.py` | Auto-compression when hitting token limit, usage bar |
| 02 | `02_persistence_and_resume.py` | messages.json persistence, session resume |
| 03 | `03_callbacks_and_reinjection.py` | on_before/after_compress, re-inject instructions, focused compaction |
| 04 | `04_auto_max_tokens.py` | genai-prices auto-detection of context window (no API key needed) |
| 05 | `05_interactive_chat.py` | Full interactive REPL with /compact, /context commands |
| 06 | `06_standalone_processors.py` | SummarizationProcessor vs SlidingWindowProcessor (no middleware) |

```bash
uv run python examples/01_basic_context_manager.py
uv run python examples/04_auto_max_tokens.py   # no API key needed
uv run python examples/05_interactive_chat.py   # interactive
```

## Basic Usage

The simplest way to add real-time context management:

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

# Auto-detect max_tokens from model name
middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1",
)

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

## Message Persistence and Session Resume

Persist all messages to a file and resume sessions:

```python
from pathlib import Path
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelMessagesTypeAdapter
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

MESSAGES_PATH = "/tmp/my_session/messages.json"

middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1-mini",
    messages_path=MESSAGES_PATH,
)

agent = Agent("openai:gpt-4.1-mini", history_processors=[middleware])
wrapped = MiddlewareAgent(agent, middleware=[middleware])

# --- Session 1: Start fresh ---
history: list[ModelMessage] = []
result = await wrapped.run("What is Python?", message_history=history)
history = result.all_messages()
# messages.json is written automatically

# --- Session 2: Resume ---
raw = Path(MESSAGES_PATH).read_bytes()
history = list(ModelMessagesTypeAdapter.validate_json(raw))
result = await wrapped.run("What did we discuss?", message_history=history)
```

## Callbacks and Re-Injection

Use callbacks to get notified about compression and re-inject critical instructions:

```python
from pydantic_ai.messages import ModelMessage
from pydantic_ai_summarization import create_context_manager_middleware

CRITICAL_RULES = "Always respond in English. Never use markdown."

def on_before(messages: list[ModelMessage], cutoff_index: int) -> None:
    print(f"Compressing {cutoff_index} of {len(messages)} messages")

def on_after(messages: list[ModelMessage]) -> str | None:
    print(f"Compression done — {len(messages)} messages remain")
    return CRITICAL_RULES  # re-injected as SystemPromptPart

middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1-mini",
    compress_threshold=0.7,
    on_before_compress=on_before,
    on_after_compress=on_after,
)
```

## Guided Compaction

Focus the summary on specific topics:

```python
# Direct compaction (for CLI /compact commands)
history = await middleware.compact(history, focus="API design decisions")

# Deferred compaction (triggers on next __call__)
middleware.request_compact(focus="debugging session details")
```

## Auto-Detection of max_tokens

No need to set `max_tokens` manually — pass `model_name` and it auto-detects:

```python
from pydantic_ai_summarization import resolve_max_tokens

# Standalone lookup
resolve_max_tokens("openai:gpt-4.1")         # → 1,000,000
resolve_max_tokens("openai:gpt-4o")           # → 128,000
resolve_max_tokens("anthropic:claude-sonnet-4-20250514")  # → 200,000

# In middleware — auto-detect
middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1",  # auto-detects 1M budget
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
    model_name="openai:gpt-4.1",
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

## Custom Summarization Model

Use a different model for generating compression summaries:

```python
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    summarization_model="openai:gpt-4.1-mini",  # Fast, cheap model for summaries
)
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
    model_name="openai:gpt-4.1",
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
