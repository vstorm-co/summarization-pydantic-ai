# LimitWarnerProcessor

The `LimitWarnerProcessor` is a standalone history processor that injects warning
instructions into the next model request as configured limits approach.

It is useful when you want the agent to finish efficiently before:

- request count reaches its maximum
- current message history gets too close to the context window
- cumulative run token usage reaches a budget cap

## Basic Usage

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_limit_warner_processor

processor = create_limit_warner_processor(
    max_iterations=40,
    max_context_tokens=100000,
    max_total_tokens=200000,
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)
```

## How It Works

1. Removes any warning parts it generated on earlier turns
2. Checks configured limits against the current `RunContext`
3. Measures current context size from the cleaned message history
4. Appends a new trailing `ModelRequest` whose `UserPromptPart` carries the warning text (a separate user turn, not extra system text on the last message)

The warning is ephemeral for context-window pressure: once compaction reduces the
history size, the old context warning is removed and not re-injected.

Iteration and total-token warnings are monotonic: if those metrics are still above
threshold, the processor injects an updated warning again on the next turn.

## Configuration

```python
from pydantic_ai_summarization import LimitWarnerProcessor

processor = LimitWarnerProcessor(
    max_iterations=30,
    max_context_tokens=90000,
    max_total_tokens=180000,
    warn_on=["iterations", "context_window", "total_tokens"],
    warning_threshold=0.75,
    critical_remaining_iterations=2,
)
```

### `warning_threshold`

Default `0.7`. The fraction of a configured limit at which warnings begin. For each enabled
metric the processor computes `usage = current / limit`; no warning is emitted until
`usage >= warning_threshold`. For example, with `max_iterations=40` and the default `0.7`,
warnings start once 28 of 40 requests have been used. Must satisfy `0 < x <= 1`.

### `critical_remaining_iterations`

Default `3`. Warnings carry one of two severities, `URGENT` or `CRITICAL`. A warning starts
as `URGENT` and escalates to `CRITICAL` when the situation is dire:

- **Iterations** — escalates to `CRITICAL` once the remaining request count
  (`max_iterations - requests_used`) is `<= critical_remaining_iterations`.
- **Context window** and **total tokens** — escalate to `CRITICAL` only once usage reaches
  or exceeds the limit (`usage >= 1`).

When multiple warnings are active, the combined message is `URGENT` only if *every* active
warning is `URGENT`; otherwise it is `CRITICAL`. The severity also changes the guidance text:
`URGENT` asks the agent to "complete the current task efficiently", while `CRITICAL` asks it
to "complete the current task immediately". `critical_remaining_iterations` must be
non-negative.

## Ordering with Other Processors

Processor order matters:

- Put `LimitWarnerProcessor` after trimming or summarization processors if you want
  warnings to reflect the post-compaction history.
- Put it before them only if you explicitly want a pre-compaction warning pass.
