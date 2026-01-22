# Installation

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Install with uv (recommended)

```bash
uv add summarization-pydantic-ai
```

## Install with pip

```bash
pip install summarization-pydantic-ai
```

## Optional Dependencies

| Extra | Description | Use Case |
|-------|-------------|----------|
| `tiktoken` | Accurate token counting | OpenAI models |

### Tiktoken for Accurate Token Counting

For more accurate token counting (especially with OpenAI models):

=== "uv"

    ```bash
    uv add summarization-pydantic-ai[tiktoken]
    ```

=== "pip"

    ```bash
    pip install summarization-pydantic-ai[tiktoken]
    ```

!!! tip "When to use tiktoken"
    The default token counter uses a heuristic (~4 chars per token). For production applications with OpenAI models, tiktoken provides exact token counts.

## Environment Setup

### API Key

summarization-pydantic-ai uses Pydantic AI which supports multiple model providers. Set your API key:

=== "OpenAI"

    ```bash
    export OPENAI_API_KEY=your-api-key
    ```

=== "Anthropic"

    ```bash
    export ANTHROPIC_API_KEY=your-api-key
    ```

=== "Google"

    ```bash
    export GOOGLE_API_KEY=your-api-key
    ```

## Verify Installation

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor, __version__

print(f"summarization-pydantic-ai version: {__version__}")

async def main():
    processor = create_summarization_processor(
        trigger=("messages", 50),
        keep=("messages", 10),
    )

    agent = Agent(
        "openai:gpt-4o",
        history_processors=[processor],
    )

    result = await agent.run("Hello!")
    print(f"Agent response: {result.output}")

asyncio.run(main())
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you have the correct Python version:

```bash
python --version  # Should be 3.10+
```

### API Key Not Found

Make sure your API key is set in the environment:

```bash
echo $OPENAI_API_KEY
```

### pydantic-ai Not Found

Ensure pydantic-ai is installed:

```bash
pip install pydantic-ai
# or
uv add pydantic-ai
```

## Next Steps

- [Core Concepts](concepts/index.md) - Learn how processors work
- [Basic Usage Example](examples/basic-usage.md) - Your first processor
- [API Reference](api/index.md) - Complete API documentation
