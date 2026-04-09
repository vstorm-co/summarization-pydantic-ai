# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.4] - 2026-04-09

### Added

- **`keep_head` parameter on `SlidingWindowProcessor`** — preserve N messages from the start of the conversation during sliding window trimming. Prevents loss of system prompts and initial instructions when the window slides forward. Supports all `ContextSize` formats: `("messages", N)`, `("tokens", N)`, `("fraction", F)`. Automatically adjusts the head boundary to avoid splitting tool call/response pairs. ([#17](https://github.com/vstorm-co/summarization-pydantic-ai/issues/17), reported by [@zlowred](https://github.com/zlowred))

  ```python
  # Keep system prompt when trimming
  processor = SlidingWindowProcessor(
      trigger=("messages", 100),
      keep=("messages", 50),
      keep_head=("messages", 1),
  )
  ```

- **`keep_head` on `SlidingWindowCapability`** — same parameter exposed on the capability wrapper
- **`keep_head` on `create_sliding_window_processor()`** — same parameter on the factory function

## [0.1.3] - 2026-04-02

### Added

- **`compact_conversation` tool on `ContextManagerCapability`** — when `include_compact_tool=True`, the capability provides a `compact_conversation(focus?)` tool so agents can manually trigger context compression with an optional focus topic. Uses `request_compact()` internally — compression is deferred to the next model request.

### Fixed

- **`LimitWarnerProcessor` warning delivery** — warnings are injected as a trailing `UserPromptPart` in a new `ModelRequest` instead of appending a `SystemPromptPart` to the last turn, so models treat the limit notice like a distinct user message ([#14](https://github.com/vstorm-co/summarization-pydantic-ai/pull/14), by [@Gby56](https://github.com/Gby56))

## [0.1.2] - 2026-03-31

### Changed

- Bump minimum `pydantic-ai-slim` to `>=1.74.0` for compatibility with async `get_instructions` on toolsets

## [0.1.1] - 2026-03-28

### Fixed

- **System prompts preserved during compression** — previously, compression replaced the entire message history including original system prompts (tool descriptions, skill lists, agent instructions). Now `_extract_system_prompts()` preserves leading `SystemPromptPart` entries and prepends them to the summary message. ([#12](https://github.com/vstorm-co/summarization-pydantic-ai/pull/12), reported by [@ilayu-blip](https://github.com/ilayu-blip))

### Added

- **`DEFAULT_CONTINUATION_PROMPT`** constant — customizable prefix for the summary message (default: `"Summary of previous conversation:\n\n"`)

### Removed

- **`ContextManagerMiddleware`** and `pydantic-ai-middleware` dependency — replaced by `ContextManagerCapability` (pydantic-ai native capabilities). The `[hybrid]` extra is no longer needed.
- **`middleware.py`** module — all functionality now in `capability.py` and standalone processors

## [0.1.0] - 2026-03-26

### Added

- **4 pydantic-ai capabilities** — native [capabilities](https://ai.pydantic.dev/capabilities/) replacing the need for `pydantic-ai-middleware`:
  - **`SummarizationCapability`** — LLM-based history compression via `before_model_request`
  - **`SlidingWindowCapability`** — zero-cost message trimming via `before_model_request`
  - **`LimitWarnerCapability`** — warning injection when limits approach via `before_model_request`
  - **`ContextManagerCapability`** — full context management with token tracking, auto-compression (`before_model_request`), tool output truncation (`after_tool_execute`), auto-detect `max_tokens` via `for_run`, and `compact()` method callable outside `agent.run()`
- All capabilities support AgentSpec YAML serialization

### Changed

- **Minimum pydantic-ai version bumped to `>=1.71.0`** (capabilities API support)

### Deprecated

- `ContextManagerMiddleware` (the `AgentMiddleware` subclass in `middleware.py`) — use `ContextManagerCapability` instead

## [0.0.5] - 2026-03-21

### Added

- **`LimitWarnerProcessor`** — standalone history processor that injects warning `SystemPromptPart`s as request, context-window, or total-token limits approach ([#10](https://github.com/vstorm-co/summarization-pydantic-ai/pull/10), by [@Gby56](https://github.com/Gby56))
- **`create_limit_warner_processor()`** factory function
- **`WarningOn`** type alias for selecting warning categories

## [0.0.4] - 2026-02-25

### Added

- **`on_before_compress` callback** on `ContextManagerMiddleware` — called with
  `(messages_to_discard, cutoff_index)` before compression summarizes and discards
  messages. Enables persistent history archival (e.g. save full conversation to
  files before pruning).
- **`on_after_compress` callback** — called with compressed messages after
  compression. Return a string to re-inject it into context as a `SystemPromptPart`
  (inspired by Claude Code's SessionStart hook with compact matcher).
- **Continuous message persistence** via `messages_path` on `ContextManagerMiddleware` —
  every message (user input, agent responses, tool calls) is saved to a single
  `messages.json` file on every history processor call. On compression, the summary
  is appended to the same file. The file is the permanent, uncompressed record of
  the full conversation. Supports session resume (loads existing history on init).
- **Guided compaction** — `_compress()` and `_create_summary()` accept a `focus`
  parameter (e.g., "Focus on the API changes") appended to the summary prompt.
- **`request_compact(focus)`** method — request manual compaction on the next
  `__call__`, with optional focus instructions.
- **`compact(messages, focus)`** method — directly compact messages with LLM
  summarization (for CLI `/compact` commands).
- **`max_tokens` auto-detection** from `genai-prices` — when `max_tokens=None`
  (the new default), the middleware resolves the model's context window
  automatically via `genai-prices`. Falls back to 200,000 if not found.
- **`resolve_max_tokens(model_name)`** function exported from the package —
  standalone lookup of context windows from genai-prices.
- **`model_name` parameter** on `ContextManagerMiddleware` and factory — used for
  auto-detection of `max_tokens` when not explicitly set.
- **Async token counting** — `TokenCounter` type now accepts both sync and async
  callables (`Callable[..., int] | Callable[..., Awaitable[int]]`). Enables use of
  provider token counting APIs (e.g. Anthropic's `/count_tokens` endpoint) or
  pydantic-ai's `count_tokens()` method. ([#6](https://github.com/vstorm-co/summarization-pydantic-ai/issues/6))
- **`async_count_tokens()`** helper function exported from the package.
- `BeforeCompressCallback`, `AfterCompressCallback` type aliases exported.
- `messages_path`, `model_name`, `on_before_compress`, `on_after_compress`
  parameters added to `create_context_manager_middleware()` factory.
- **Examples** — 6 runnable examples in `examples/` covering all features:
  auto-compression, persistence, callbacks, auto-detection, interactive chat,
  standalone processors.

### Changed

- **`max_tokens` default** changed from `200_000` to `None` (auto-detect from
  genai-prices, fallback to 200,000).
- **`keep` default** changed from `("messages", 20)` to `("messages", 0)` —
  on compression, only the LLM summary survives (like Claude Code). This produces
  the most compact context after compression.
- **Validation** now allows `0` for messages/tokens keep and trigger values
  (previously required > 0). Negative values are still rejected.

### Dependencies

- `genai-prices` used for auto-detection of context windows (already a transitive
  dependency via pydantic-ai-middleware).

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

[0.1.4]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.1.4
[0.1.3]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.1.3
[0.1.2]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.1.2
[0.1.1]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.1.1
[0.1.0]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.1.0
[0.0.5]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.5
[0.0.4]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.4
[0.0.3]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.3
[0.0.2]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.2
[0.0.1]: https://github.com/vstorm-co/summarization-pydantic-ai/releases/tag/v0.0.1
