# Context Manager

[`ContextManagerCapability`][pydantic_ai_summarization.capability.ContextManagerCapability]
is the all-in-one capability: live token tracking, automatic compression, tool-output
truncation, and an optional agent-triggered compaction tool. This page walks through each
feature with runnable snippets.

## Live Token Tracking

Pass an `on_usage_update` callback to observe context usage on every model request. It
receives the usage fraction, the current token count, and the resolved max budget. The same
callback fires again after a compression so you can see usage drop.

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import ContextManagerCapability


def on_usage(pct: float, current: int, max_tokens: int) -> None:
    bar = "#" * int(pct * 20)
    print(f"[{bar:<20}] {pct:.0%}  ({current}/{max_tokens} tokens)")


agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        ContextManagerCapability(
            max_tokens=100_000,
            on_usage_update=on_usage,
        )
    ],
)
```

If `max_tokens` is left as `None`, the capability auto-detects the model's context window via
`genai-prices` on the first run, falling back to `200_000` when detection is unavailable.

## Automatic Compression

Compression fires automatically once usage reaches `compress_threshold` (default `0.9`).
Tune the threshold and how much of the tail survives via `keep`:

```python
cap = ContextManagerCapability(
    max_tokens=100_000,
    compress_threshold=0.85,        # compress at 85% of the budget
    keep=("messages", 6),           # keep the last 6 messages after summarizing
    summarization_model="openai:gpt-4.1-mini",
)
```

With the default `keep=("messages", 0)`, only the generated summary survives a compression.
See [Capabilities](../concepts/capability.md) for the full parameter table and a description
of the threshold mechanism.

## Tool-Output Truncation

Large tool results can dominate the context window. Set `max_tool_output_tokens` to truncate
any tool output larger than that budget (measured as roughly `tokens * 4` characters),
keeping a head and tail slice:

```python
cap = ContextManagerCapability(
    max_tokens=100_000,
    max_tool_output_tokens=2_000,   # truncate tool outputs over ~2k tokens
    tool_output_head_lines=10,      # keep first 10 lines
    tool_output_tail_lines=5,       # keep last 5 lines
)
```

Truncation runs in `after_tool_execute`, so the trimmed output is what enters the message
history. When `max_tool_output_tokens` is `None` (the default), tool outputs are never
truncated.

## Agent-Triggered Compaction

Set `include_compact_tool=True` to register a `compact_conversation` tool. The agent can then
decide to compress the conversation itself, optionally focusing the summary on a topic:

```python
cap = ContextManagerCapability(
    max_tokens=100_000,
    include_compact_tool=True,
)

agent = Agent("openai:gpt-4.1", capabilities=[cap])
```

Calling the tool does not compress immediately — it sets a flag, and compaction is applied
before the next model request. The optional `focus` argument is passed through to the
summary prompt so the model prioritizes the requested topic.

## Manual Compaction

You can also drive compaction yourself, either deferred or immediate:

```python
# Deferred: applied before the next model request (same path as the tool)
cap.request_compact(focus="the database migration plan")

# Immediate: compress a message list directly, outside agent.run()
compressed = await cap.compact(messages, focus="the database migration plan")
```

[`request_compact()`][pydantic_ai_summarization.capability.ContextManagerCapability.request_compact]
queues compaction for the next request, while
[`compact()`][pydantic_ai_summarization.capability.ContextManagerCapability.compact] runs the
summarization immediately and returns the compressed list. Both accept an optional `focus`.

## Compression Callbacks

Two callbacks let you observe or augment compression:

```python
def before_compress(messages: list, cutoff_index: int) -> None:
    print(f"About to compress {len(messages)} messages")


def after_compress(messages: list) -> str | None:
    # Returning a string re-injects it into the first request as a SystemPromptPart
    return "Reminder: keep responses concise after compaction."


cap = ContextManagerCapability(
    max_tokens=100_000,
    on_before_compress=before_compress,
    on_after_compress=after_compress,
)
```

`on_before_compress` is called with the pre-compression messages and a cutoff index just
before summarization runs. `on_after_compress` is called with the compressed messages; if it
returns a string, that text is appended to the first request as a `SystemPromptPart`.

## Next Steps

- [Capabilities](../concepts/capability.md) - full parameter reference
- [Basic Usage](basic-usage.md) - minimal setup
- [Advanced](advanced.md) - custom token counters and prompts
