# Capabilities

The recommended way to add context management to a Pydantic AI agent is via
[capabilities](https://ai.pydantic.dev/capabilities/). No middleware wrappers needed.

## Why Capabilities?

| Feature | Capabilities | Processor API |
|---------|:-:|:-:|
| Native pydantic-ai | Yes | Yes |
| Tool output truncation | `ContextManagerCapability` | No |
| Auto-detect max_tokens | `ContextManagerCapability` | No |
| `compact()` outside run | `ContextManagerCapability` | No |
| Agent-triggered compaction | `ContextManagerCapability` (with `include_compact_tool=True`) | No |
| AgentSpec YAML | Yes | No | No |

## Available Capabilities

### ContextManagerCapability

Full context management — token tracking, auto-compression, tool output truncation:

```python
from pydantic_ai_summarization import ContextManagerCapability

cap = ContextManagerCapability(
    max_tokens=100_000,          # Auto-detected if None
    compress_threshold=0.9,      # Compress at 90% usage
    max_tool_output_tokens=5000, # Truncate large tool outputs
    include_compact_tool=True,   # Add compact_conversation tool
)
```

When `include_compact_tool=True`, the agent gets a `compact_conversation(focus?)` tool
that triggers compression on the next model request. The optional `focus` parameter
guides the summary to prioritize specific topics.

#### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_tokens` | `None` | Token budget. When `None`, it is auto-detected from the model via `genai-prices` on the first run, falling back to `200_000` if detection fails. |
| `compress_threshold` | `0.9` | Fraction of `max_tokens` at which auto-compression fires. Must satisfy `0 < x <= 1` (validated in `__post_init__`). |
| `keep` | `("messages", 0)` | How much of the tail to preserve after compression. The default of `0` means only the generated summary survives. |
| `summarization_model` | `"openai:gpt-4.1-mini"` | Model used to generate the summary. |
| `token_counter` | `count_tokens_approximately` | Callable used to measure context size. Sync or async ([`TokenCounter`][pydantic_ai_summarization.types.TokenCounter]). |
| `summary_prompt` | `DEFAULT_SUMMARY_PROMPT` | Prompt template used when summarizing. |
| `max_tool_output_tokens` | `None` | When set, tool outputs larger than this (measured as `tokens * 4` chars) are truncated head/tail. `None` disables truncation. |
| `tool_output_head_lines` | `5` | Lines kept from the start of a truncated tool output. |
| `tool_output_tail_lines` | `5` | Lines kept from the end of a truncated tool output. |
| `on_usage_update` | `None` | Callback `(pct, current, max_tokens)` invoked on every model request (and again after compression) for live token tracking. |
| `on_before_compress` | `None` | Callback `(messages, cutoff_index)` invoked between plan and execute, with the **real** cutoff index the processor chose. Not called when no compression will run. |
| `on_after_compress` | `None` | Callback `(messages, summarized, summary)` invoked after the attempt. `summarized=False` covers both "trigger fired but LLM failed" cases. Returns an optional `str` re-injected as a `SystemPromptPart` on the first message (only when `summarized=True`). |
| `include_compact_tool` | `False` | When `True`, registers the `compact_conversation` agent tool. |

#### Auto-compression mechanism

On every `before_model_request`, the capability counts tokens and calls
`on_usage_update(pct, total, max_tokens)` if provided. The capability then delegates the
compress-or-not decision to its [`SummarizationProcessor`][pydantic_ai_summarization.processor.SummarizationProcessor]
(via the two-phase `plan_compression` / `execute_plan` API): the processor is the single
decision-maker for both *whether* and *where* to compress. When a plan exists,
`on_before_compress(messages, cutoff_index)` fires with the actual cutoff index, then the
summary LLM runs, then `on_after_compress(messages, summarized, summary)` fires with the
outcome. Manual compaction
([`request_compact()`][pydantic_ai_summarization.capability.ContextManagerCapability.request_compact],
the `compact_conversation` tool, or
[`compact()`][pydantic_ai_summarization.capability.ContextManagerCapability.compact])
passes `force=True`, so it always attempts compression rather than being silently vetoed by
the trigger check.

### SummarizationCapability

LLM-based history compression:

```python
from pydantic_ai_summarization import SummarizationCapability

cap = SummarizationCapability(
    trigger=("messages", 50),
    keep=("messages", 10),
)
```

!!! note "Capability defaults differ from the standalone processor"
    [`SummarizationCapability`][pydantic_ai_summarization.capability.SummarizationCapability]
    ships its own defaults — `model="openai:gpt-4.1-mini"`, `trigger=("messages", 50)`, and
    `keep=("messages", 10)` — which are not necessarily the same as the
    [`SummarizationProcessor`][pydantic_ai_summarization.processor.SummarizationProcessor]
    defaults. Set the parameters explicitly if you need a specific configuration.

### SlidingWindowCapability

Zero-cost message trimming:

```python
from pydantic_ai_summarization import SlidingWindowCapability

cap = SlidingWindowCapability(
    trigger=("messages", 100),
    keep=("messages", 50),
)
```

### LimitWarnerCapability

Warn the agent before limits hit:

```python
from pydantic_ai_summarization import LimitWarnerCapability

cap = LimitWarnerCapability(
    max_iterations=40,
    max_context_tokens=100_000,
    max_total_tokens=200_000,
)
```

## Composing Capabilities

```python
agent = Agent(
    "openai:gpt-4.1",
    capabilities=[
        LimitWarnerCapability(max_iterations=40),
        ContextManagerCapability(max_tokens=100_000),
    ],
)
```
