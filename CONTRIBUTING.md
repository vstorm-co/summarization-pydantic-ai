# Contributing to summarization-pydantic-ai

Thank you for your interest in contributing to summarization-pydantic-ai!

## Development Setup

### Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) - Fast Python package manager

### Getting Started

```bash
# Clone the repository
git clone https://github.com/vstorm-co/summarization-pydantic-ai.git
cd summarization-pydantic-ai

# Install dependencies
uv sync --group dev

# Run tests
uv run pytest

# Run all checks (lint, format, typecheck)
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

## Development Workflow

### Running Tests

```bash
# Run all tests with coverage
uv run coverage run -m pytest
uv run coverage report

# Run specific test
uv run pytest tests/test_processor.py::TestSummarizationProcessor -v

# Run with debug output
uv run pytest -v -s
```

### Code Quality

We use the following tools:

- **ruff** - Linting and formatting
- **pyright** - Type checking
- **pytest** - Testing with 100% coverage requirement

```bash
# Format code
uv run ruff format .

# Fix lint issues
uv run ruff check --fix .

# Type check
uv run pyright
```

## Pull Request Guidelines

### Requirements

1. **100% test coverage** - All new code must be covered by tests
2. **Type annotations** - All functions must have type hints
3. **Passing CI** - All checks must pass (lint, typecheck, tests)

### Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run all checks locally
5. Commit with a descriptive message
6. Push and open a Pull Request

### Commit Messages

Follow conventional commit format:

```
feat: add new trigger type
fix: handle empty message list
docs: update README examples
test: add edge case coverage
```

## Project Structure

```
pydantic_ai_summarization/
├── __init__.py       # Public API exports
├── processor.py      # SummarizationProcessor
├── sliding_window.py # SlidingWindowProcessor
├── types.py          # ContextSize, TokenCounter types
└── utils.py          # count_tokens_approximately(), format_messages_for_summary()

tests/
├── test_processor.py
├── test_sliding_window.py
└── test_utils.py
```

## Design Principles

1. **History Processor Protocol** - Compatible with pydantic-ai's history_processors
2. **Safe Cutoff** - Never split tool call/response pairs
3. **Flexible Configuration** - Support messages, tokens, and fraction-based triggers

## Questions?

Open an issue on GitHub for questions or discussions.
