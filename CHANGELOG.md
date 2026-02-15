# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2025-02-15

### Added

- **ContextManagerMiddleware** - Dual-protocol middleware for real-time context management
  - Acts as pydantic-ai `history_processor` for token tracking and auto-compression
  - Acts as pydantic-ai-middleware `AgentMiddleware` for tool output truncation
  - `on_usage_update` callback for real-time usage tracking
  - `max_tool_output_tokens` for limiting individual tool output sizes
  - `create_context_manager_middleware()` factory function
  - Requires `hybrid` extra: `pip install summarization-pydantic-ai[hybrid]`

- **Shared cutoff algorithms** - New internal `_cutoff.py` module extracted from processors
  - `validate_context_size()` - Context configuration validation
  - `should_trigger()` - Trigger condition evaluation
  - `determine_cutoff_index()` - Retention-aware cutoff calculation
  - `find_safe_cutoff()` - Tool call/response pair preservation
  - `find_token_based_cutoff()` - Binary search token cutoff
  - `is_safe_cutoff_point()` - Safety validation for cutoff points
  - `validate_triggers_and_keep()` - Configuration normalization
  - Reduces code duplication between `SummarizationProcessor` and `SlidingWindowProcessor`

- `ModelType` type alias (`str | Model | KnownModelName`) exported from the package for convenience.

### Changed

- **Lightweight dependency**: Replaced `pydantic-ai` with `pydantic-ai-slim` to avoid pulling in unnecessary model-specific SDKs (openai, anthropic, etc.). ([#4](https://github.com/vstorm-co/summarization-pydantic-ai/issues/4))
- **Custom model support**: `SummarizationProcessor.model`, `ContextManagerMiddleware.summarization_model`, and factory functions now accept `str | Model | KnownModelName` — enabling custom providers like Azure OpenAI. ([#3](https://github.com/vstorm-co/summarization-pydantic-ai/issues/3))
- **Code refactoring**: Extracted common logic from `processor.py` and `sliding_window.py` into shared `_cutoff.py` module
- **README**: Updated with ContextManagerMiddleware, hybrid extra, and new features

### Dependencies

- Added `hybrid` extra: `pydantic-ai-middleware>=0.2.0` (optional)
- `pydantic-ai-middleware` added to dev dependencies
- Replaced `pydantic-ai>=0.1.0` with `pydantic-ai-slim>=0.1.0`

## [0.0.2] - 2025-01-22

### Changed

- **README**: Complete rewrite with centered header, badges, Use Cases table, and vstorm-co branding
- **Documentation**: Updated styling to match pydantic-deep pink theme
  - Inter font for text, JetBrains Mono for code
  - Pink accent color scheme
  - Custom CSS and announcement bar
- **mkdocs.yml**: Updated with full Material theme configuration

### Added

- **Custom Styling**: docs/overrides/main.html, docs/stylesheets/extra.css
- **Abbreviations**: docs/includes/abbreviations.md for markdown expansions

## [0.0.1] - 2025-01-20

### Added

- **SummarizationProcessor** - History processor that uses LLM to intelligently summarize older messages when context limits are reached
  - Configurable triggers: message count, token count, or fraction of context window
  - Configurable retention: keep last N messages, tokens, or fraction
  - Custom token counter support
  - Custom summary prompt support
  - Safe cutoff detection - never splits tool call/response pairs

- **SlidingWindowProcessor** - Zero-cost history processor that simply discards old messages
  - Same trigger and retention options as SummarizationProcessor
  - No LLM calls - instant, deterministic processing
  - Ideal for high-throughput scenarios

- **Factory functions** for convenient processor creation:
  - `create_summarization_processor()` - with sensible defaults
  - `create_sliding_window_processor()` - with sensible defaults

- **Utility functions**:
  - `count_tokens_approximately()` - heuristic token counter (~4 chars per token)
  - `format_messages_for_summary()` - formats messages for LLM summarization

- **Type definitions**:
  - `ContextSize` - union type for trigger/keep configuration
  - `ContextFraction`, `ContextTokens`, `ContextMessages` - specific context size types
  - `TokenCounter` - callable type for custom token counters

- **Documentation**:
  - Full MkDocs documentation with Material theme
  - Concepts, examples, and API reference
  - Integration examples with pydantic-ai

### Dependencies

- Requires `pydantic-ai>=0.1.0`
- Optional `tiktoken` support for accurate token counting

[0.0.3]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.3
[0.0.2]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.2
[0.0.1]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.1
