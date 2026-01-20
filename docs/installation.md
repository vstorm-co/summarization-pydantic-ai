# Installation

## Prerequisites

- Python 3.10 or higher
- [pydantic-ai](https://github.com/pydantic/pydantic-ai) installed

## Installation Methods

=== "pip"

    ```bash
    pip install summarization-pydantic-ai
    ```

=== "uv"

    ```bash
    uv add summarization-pydantic-ai
    ```

=== "poetry"

    ```bash
    poetry add summarization-pydantic-ai
    ```

## Optional Dependencies

### Tiktoken for Accurate Token Counting

For more accurate token counting (especially with OpenAI models):

```bash
pip install summarization-pydantic-ai[tiktoken]
```

## Verification

Verify your installation:

```python
from pydantic_ai_summarization import create_summarization_processor, __version__

print(f"summarization-pydantic-ai version: {__version__}")

# Create a processor to verify everything works
processor = create_summarization_processor()
print("Installation successful!")
```

## Next Steps

- Learn about [Core Concepts](concepts/index.md)
- See [Basic Usage Examples](examples/basic-usage.md)
