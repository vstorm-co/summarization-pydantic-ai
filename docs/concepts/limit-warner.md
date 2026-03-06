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
4. Appends a generated `SystemPromptPart` to the trailing `ModelRequest`

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

## Ordering with Other Processors

Processor order matters:

- Put `LimitWarnerProcessor` after trimming or summarization processors if you want
  warnings to reflect the post-compaction history.
- Put it before them only if you explicitly want a pre-compaction warning pass.
