# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.4] - 2026-02-25

### Added

- **`on_before_compress` callback** — called before compression with messages and cutoff index
- **`on_after_compress` callback** — called after compression; return a string to re-inject into context as `SystemPromptPart`
- **Continuous message persistence** via `messages_path` — saves all messages to JSON on every call
- **Guided compaction** — `focus` parameter on `compact()` and `request_compact()` for topic-focused summaries
- **`request_compact(focus)`** method — deferred manual compaction
- **`compact(messages, focus)`** method — direct LLM-based compaction
- **`max_tokens` auto-detection** from `genai-prices` via `resolve_max_tokens(model_name)`
- **`model_name` parameter** for auto-detection
- **Async token counting** — `TokenCounter` now accepts sync or async callables ([#6](https://github.com/vstorm-co/summarization-pydantic-ai/issues/6))
- **`async_count_tokens()`** helper function
- `BeforeCompressCallback`, `AfterCompressCallback` type aliases
- 6 runnable examples in `examples/`

### Changed

- **`max_tokens` default**: `200_000` → `None` (auto-detect, fallback 200K)
- **`keep` default**: `("messages", 20)` → `("messages", 0)` (only summary survives)
- **Validation**: 0 now allowed for messages/tokens (negative still rejected)

## [0.0.3] - 2025-02-15

### Added

- **ContextManagerMiddleware** — dual-protocol middleware (history processor + AgentMiddleware)
- **Shared cutoff algorithms** — `_cutoff.py` module
- `ModelType` type alias

### Changed

- Replaced `pydantic-ai` with `pydantic-ai-slim`
- Custom model support (`str | Model | KnownModelName`)

## [0.0.2] - 2025-01-22

### Changed

- README rewrite, documentation styling

## [0.0.1] - 2025-01-20

### Added

- `SummarizationProcessor`, `SlidingWindowProcessor`
- Factory functions, utility functions, type definitions
- Full MkDocs documentation
