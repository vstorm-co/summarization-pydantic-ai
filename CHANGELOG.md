# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.0.1]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.1
