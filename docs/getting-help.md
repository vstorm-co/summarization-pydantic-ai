# Getting Help

## Documentation

This documentation is your primary resource. Use the search bar (press `/` or `s`) to find specific topics.

## GitHub Issues

For bugs, feature requests, or questions:

[:fontawesome-brands-github: Open an Issue](https://github.com/vstorm-co/summarization-pydantic-ai/issues){ .md-button }

### Before Opening an Issue

1. **Search existing issues** - Your problem may already be reported
2. **Check the docs** - The answer might be here
3. **Prepare a minimal example** - Help us reproduce the issue

### Bug Report Template

```markdown
## Description
[Clear description of the bug]

## Steps to Reproduce
1. Create processor with...
2. Add to agent...
3. Run agent...
4. Observe error...

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## Environment
- summarization-pydantic-ai version: X.X.X
- pydantic-ai version: X.X.X
- Python version: 3.XX
- OS: [e.g., macOS 14.0]
```

## Community Resources

### Pydantic AI

summarization-pydantic-ai is built on Pydantic AI. Their documentation is an excellent resource:

- [Pydantic AI Documentation](https://ai.pydantic.dev/)
- [Pydantic AI GitHub](https://github.com/pydantic/pydantic-ai)

### Related Projects

- [pydantic-deep](https://github.com/vstorm-co/pydantic-deep) - Full agent framework
- [pydantic-ai-backend](https://github.com/vstorm-co/pydantic-ai-backend) - File storage backends
- [pydantic-ai-todo](https://github.com/vstorm-co/pydantic-ai-todo) - Task planning toolset

## FAQ

### When should I use SummarizationProcessor vs SlidingWindowProcessor?

**Use SummarizationProcessor** when:

- Context quality matters (coding assistants, complex conversations)
- You need to preserve key decisions and information
- LLM cost is acceptable

**Use SlidingWindowProcessor** when:

- Speed and cost are priorities
- Recent messages are most important
- Running many parallel conversations
- You want deterministic behavior

### Can I use this with models other than OpenAI?

Yes! Any model supported by Pydantic AI works:

```python
from pydantic_ai_summarization import create_summarization_processor

# Works with any model
processor = create_summarization_processor(
    model="anthropic:claude-3-5-sonnet-20241022",
)
```

### How do I test without API calls?

Use `TestModel` from Pydantic AI:

```python
from pydantic_ai.models.test import TestModel
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(model=TestModel())
agent = Agent(TestModel(), history_processors=[processor])
```

### Can I customize how summaries are generated?

Yes! Use the `summary_prompt` parameter:

```python
processor = create_summarization_processor(
    summary_prompt="""
    Summarize this conversation, focusing on:
    - Key decisions made
    - Code written or modified
    - Outstanding questions

    Conversation:
    {messages}
    """,
)
```

### How do I handle very long conversations?

Use multiple triggers to catch different scenarios:

```python
from pydantic_ai_summarization import SummarizationProcessor

processor = SummarizationProcessor(
    model="openai:gpt-4o",
    trigger=[
        ("messages", 50),     # Many short messages
        ("tokens", 100000),   # Fewer long messages
    ],
    keep=("messages", 10),
)
```
