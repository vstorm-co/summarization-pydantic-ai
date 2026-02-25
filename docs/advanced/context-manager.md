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
│     ├─ Persist messages to messages.json (if configured)     │
│     ├─ Notify usage callback (percentage, current, max)      │
│     ├─ If usage >= compress_threshold:                        │
│     │   ├─ Call on_before_compress callback                   │
│     │   ├─ Summarize older messages via LLM (with focus)     │
│     │   ├─ Replace old messages with summary                 │
│     │   ├─ Call on_after_compress → re-inject instructions   │
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

**Message persistence**: When `messages_path` is set, all messages are saved to a JSON file on every history processor call. This provides a permanent, uncompressed record of the full conversation — ideal for session resume.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | `int \| None` | `None` | Maximum token budget. `None` auto-detects from genai-prices (falls back to 200,000) |
| `model_name` | `str \| None` | `None` | Model name for auto-detecting `max_tokens` (e.g., `"openai:gpt-4.1"`) |
| `compress_threshold` | `float` | `0.9` | Fraction of `max_tokens` at which auto-compression triggers (0.0, 1.0] |
| `keep` | `ContextSize` | `("messages", 0)` | How much context to retain after compression. `0` = only summary survives |
| `summarization_model` | `str` | `"openai:gpt-4.1-mini"` | Model used for generating summaries |
| `token_counter` | `TokenCounter` | `count_tokens_approximately` | Function to count tokens (sync or async) |
| `summary_prompt` | `str` | `DEFAULT_SUMMARY_PROMPT` | Prompt template for summary generation |
| `trim_tokens_to_summarize` | `int` | `4000` | Max tokens to include when generating the summary |
| `max_input_tokens` | `int \| None` | `None` | Model max input tokens (required for fraction-based keep) |
| `max_tool_output_tokens` | `int \| None` | `None` | Per-tool-output token limit before truncation. `None` disables truncation |
| `tool_output_head_lines` | `int` | `5` | Lines to show from the beginning of truncated tool output |
| `tool_output_tail_lines` | `int` | `5` | Lines to show from the end of truncated tool output |
| `messages_path` | `str \| None` | `None` | Path to persist messages as JSON. Enables session resume |
| `on_usage_update` | `UsageCallback \| None` | `None` | Callback invoked with usage stats before each model call |
| `on_before_compress` | `BeforeCompressCallback \| None` | `None` | Callback before compression — receives messages and cutoff index |
| `on_after_compress` | `AfterCompressCallback \| None` | `None` | Callback after compression — return a string to re-inject into context |

## Auto-Detection of max_tokens

When `max_tokens=None` (the default), the middleware uses `resolve_max_tokens(model_name)` to look up the model's context window from `genai-prices`:

```python
from pydantic_ai_summarization import resolve_max_tokens

# Returns context window size or None
resolve_max_tokens("openai:gpt-4.1")         # → 1,000,000
resolve_max_tokens("anthropic:claude-sonnet-4-20250514")  # → 200,000
resolve_max_tokens("unknown:model")           # → None (falls back to 200,000)
```

This means you typically don't need to set `max_tokens` manually — just pass `model_name`:

```python
middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1",  # auto-detects 1M token budget
)
```

## Callbacks

### on_before_compress

Called before compression begins. Useful for logging or archival:

```python
from pydantic_ai.messages import ModelMessage

def on_before_compress(messages: list[ModelMessage], cutoff_index: int) -> None:
    print(f"About to compress {cutoff_index} messages out of {len(messages)}")
```

### on_after_compress

Called after compression. Return a string to re-inject it as a `SystemPromptPart`:

```python
CRITICAL_INSTRUCTIONS = "Always respond in English. Never use markdown."

def on_after_compress(messages: list[ModelMessage]) -> str | None:
    # Re-inject instructions that must survive compression
    return CRITICAL_INSTRUCTIONS

middleware = create_context_manager_middleware(
    on_after_compress=on_after_compress,
)
```

This is inspired by Claude Code's SessionStart hook with compact matcher — ensures critical rules survive context compression.

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

## Message Persistence

When `messages_path` is set, all messages are written to a JSON file on every history processor call:

```python
middleware = create_context_manager_middleware(
    messages_path="/tmp/session/messages.json",
)
```

The file contains the full, uncompressed conversation history. On compression, the summary message is appended — the file is always the permanent record.

To resume a session, load the file and pass it as `message_history`:

```python
from pathlib import Path
from pydantic_ai.messages import ModelMessagesTypeAdapter

raw = Path("/tmp/session/messages.json").read_bytes()
history = list(ModelMessagesTypeAdapter.validate_json(raw))
result = await agent.run("Continue...", message_history=history)
```

## Guided Compaction

Both `compact()` and `request_compact()` accept a `focus` parameter to guide the summary:

```python
# Direct compaction (for CLI commands)
history = await middleware.compact(history, focus="Focus on the API design decisions")

# Request compaction on next __call__ (deferred)
middleware.request_compact(focus="Focus on the debugging session")
```

The focus string is appended to the summary prompt, telling the LLM what to prioritize in the summary.

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_middleware import MiddlewareAgent
from pydantic_ai_summarization import create_context_manager_middleware

middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1",  # auto-detect max_tokens
    compress_threshold=0.9,
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

# With defaults (auto-detect max_tokens)
middleware = create_context_manager_middleware()

# Fully configured
middleware = create_context_manager_middleware(
    model_name="openai:gpt-4.1",
    compress_threshold=0.85,
    keep=("messages", 10),
    summarization_model="openai:gpt-4.1-mini",
    messages_path="/tmp/session/messages.json",
    max_tool_output_tokens=1000,
    on_usage_update=lambda pct, cur, mx: print(f"{pct:.0%}"),
    on_after_compress=lambda msgs: "Re-injected instructions here",
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
| Message persistence | Yes (`messages_path`) | No | No |
| Guided compaction | Yes (`focus`) | No | No |
| Callbacks | Before/after compress | No | No |
| Auto max_tokens | Yes (genai-prices) | No | No |
| Async token counter | Yes | No | No |
| LLM cost | Per compression | Per trigger | Zero |
| Requires extra | `[hybrid]` | No | No |

## Next Steps

- [Context Manager Examples](../examples/context-manager.md) - Practical usage patterns
- [Installation](../installation.md#hybrid-for-context-manager-middleware) - Installing the hybrid extra
- [API Reference](../api/middleware.md) - Full API documentation
