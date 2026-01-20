.PHONY: install sync test lint format typecheck all clean docs docs-serve

# Install dependencies
install:
	uv sync --all-extras
	uv run pre-commit install

# Sync dependencies
sync:
	uv sync --all-extras

# Run tests with coverage
test:
	uv run coverage run -m pytest -v
	uv run coverage report

# Run tests without coverage
test-fast:
	uv run pytest -v

# Run linter
lint:
	uv run ruff check src tests

# Format code
format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

# Type checking
typecheck:
	uv run pyright

typecheck-mypy:
	uv run mypy src

# Run all checks
all: format lint typecheck test

# Clean build artifacts
clean:
	rm -rf build dist *.egg-info
	rm -rf .coverage htmlcov .pytest_cache .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +

# Documentation
docs:
	uv run mkdocs build

docs-serve:
	uv run mkdocs serve
