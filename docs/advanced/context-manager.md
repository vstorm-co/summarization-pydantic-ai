# ContextManagerMiddleware

The `ContextManagerMiddleware` is a dual-protocol class that provides real-time context management during agent execution. It combines:

1. **History processor** (pydantic-ai): Tracks token usage and auto-compresses conversation when approaching the token budget.
2. **Agent middleware** ([pydantic-ai-middleware](https://github.com/vstorm-co/pydantic-ai-middleware)): Optionally truncates large tool outputs inline before they enter the conversation history.

!!! warning "Requires the `hybrid` extra"
    This middleware requires the `hybrid` optional dependency:

    ```bash
    pip install summarization-pydantic-ai[hybrid]
    ```

    See [Installation](../installation.md#hybrid-for-context-manager-middleware) for details.

## How It Works

The middleware operates on two levels during each agent run:

```
┌──────────────────────────────────────────────────────────────┐
│                      Agent Run Loop                          │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  1. History Processor (__call__)                              │
│     ├─ Count tokens in current messages                      │
│     ├─ Notify usage callback (percentage, current, max)      │
│     ├─ If usage >= compress_threshold:                        │
│     │   ├─ Summarize older messages via LLM                  │
│     │   ├─ Replace old messages with summary                 │
│     │   └─ Notify updated usage                              │
│     └─ Return (possibly compressed) messages                 │
│                                                              │
│  2. After Tool Call (after_tool_call)                         │
│     ├─ Check if result exceeds max_tool_output_tokens        │
│     ├─ If yes: truncate to head + tail lines                 │
│     └─ Return original or truncated result                   │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Token tracking**: Before every model call, the middleware counts tokens in the current conversation and compares against `max_tokens * compress_threshold`. If the conversation is approaching the budget, it automatically compresses older messages using LLM summarization.

**Tool output truncation**: When `max_tool_output_tokens` is set, the middleware intercepts tool results via the `after_tool_call` hook and truncates any output that exceeds the token limit, keeping configurable head and tail lines.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int` | `200_000` | Maximum token budget for the conversation |
| `compress_threshold` | `float` | `0.9` | Fraction of `max_tokens` at which auto-compression triggers (0.0, 1.0] |
| `keep` | `ContextSize` | `("messages", 20)` | How much context to retain after compression |
| `summarization_model` | `str` | `"openai:gpt-4.1-mini"` | Model used for generating summaries |
| `token_counter` | `TokenCounter` | `count_tokens_approximately` | Function to count tokens in messages |
| `summary_prompt` | `str` | `DEFAULT_SUMMARY_PROMPT` | Prompt template for summary generation |
| `trim_tokens_to_summarize` | `int` | `4000` | Max tokens to include when generating the summary |
| `max_input_tokens` | `int \| None` | `None` | Model max input tokens (required for fraction-based keep) |
| `max_tool_output_tokens` | `int \| None` | `None` | Per-tool-output token limit before truncation. `None` disables truncation |
| `tool_output_head_lines` | `int` | `5` | Lines to show from the beginning of truncated tool output |
| `tool_output_tail_lines` | `int` | `5` | Lines to show from the end of truncated tool output |
| `on_usage_update` | `UsageCallback \| None` | `None` | Callback invoked with usage stats before each model call |

## UsageCallback

The `on_usage_update` parameter accepts a callable with the signature:

```python
UsageCallback = Callable[[float, int, int], Any]
```

The callback receives three arguments:

- `percentage` (`float`): Current usage as a fraction of `max_tokens` (e.g., `0.85` for 85%)
- `current_tokens` (`int`): Current token count in the conversation
- `max_tokens` (`int`): The configured maximum token budget

Both sync and async callables are supported. If the callable returns an awaitable, it will be awaited automatically.

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    compress_threshold=0.9,
    keep=("messages", 20),
)

# Register as both history processor and middleware
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[middleware],
)
wrapped = MiddlewareAgent(agent, middleware=[middleware])
```

## With Usage Callback

Track token usage in real time:

```python
from pydantic_ai_summarization import create_context_manager_middleware

def on_usage(percentage: float, current: int, maximum: int) -> None:
    print(f"Token usage: {percentage:.0%} ({current:,} / {maximum:,})")

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    on_usage_update=on_usage,
)
```

Async callbacks are also supported:

```python
async def on_usage_async(percentage: float, current: int, maximum: int) -> None:
    await save_usage_to_db(percentage, current, maximum)

middleware = create_context_manager_middleware(
    on_usage_update=on_usage_async,
)
```

## With Tool Output Truncation

Prevent large tool outputs from consuming too much of the token budget:

```python
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    max_tokens=200_000,
    max_tool_output_tokens=2000,      # Truncate outputs > ~2000 tokens
    tool_output_head_lines=10,        # Show first 10 lines
    tool_output_tail_lines=10,        # Show last 10 lines
)
```

When a tool output exceeds the limit, it is truncated to show the first and last N lines with an indicator of how many lines were omitted:

```
Line 1
Line 2
...
Line 10

... (485 lines omitted) ...

Line 496
Line 497
...
Line 505
```

## Factory Function

The [`create_context_manager_middleware()`][pydantic_ai_summarization.middleware.create_context_manager_middleware] factory function provides a convenient way to create a configured middleware instance:

```python
from pydantic_ai_summarization import create_context_manager_middleware

# With defaults
middleware = create_context_manager_middleware()

# Fully configured
middleware = create_context_manager_middleware(
    max_tokens=150_000,
    compress_threshold=0.85,
    keep=("messages", 30),
    summarization_model="openai:gpt-4.1-mini",
    max_tool_output_tokens=1000,
    tool_output_head_lines=5,
    tool_output_tail_lines=5,
    on_usage_update=lambda pct, cur, mx: print(f"{pct:.0%}"),
)
```

## Properties

### compression_count

The `compression_count` property returns the number of times compression has been triggered during the lifetime of the middleware instance:

```python
middleware = create_context_manager_middleware()

# ... after some agent runs ...

print(f"Compressed {middleware.compression_count} times")
```

## Comparison with Standalone Processors

| Feature | ContextManagerMiddleware | SummarizationProcessor | SlidingWindowProcessor |
|---------|------------------------|----------------------|----------------------|
| Token tracking | Built-in | No | No |
| Usage callbacks | Yes | No | No |
| Auto-compression | Yes (threshold-based) | Yes (trigger-based) | No |
| Tool output truncation | Yes | No | No |
| LLM cost | Per compression | Per trigger | Zero |
| Requires extra | `[hybrid]` | No | No |

## Next Steps

- [Context Manager Examples](../examples/context-manager.md) - Practical usage patterns
- [Installation](../installation.md#hybrid-for-context-manager-middleware) - Installing the hybrid extra
- [API Reference](../api/middleware.md) - Full API documentation
