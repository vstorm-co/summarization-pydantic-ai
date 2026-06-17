# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - 2026-06-17

### Changed

- **`ContextManagerCapability` compress hooks now match what actually happened** ([#30](https://github.com/vstorm-co/summarization-pydantic-ai/issues/30)). The hook contract was previously misleading on every dimension: `on_before_compress` hardcoded the cutoff index to `0`, `on_after_compress` fired even when nothing compressed, the `compact_conversation` tool could be a silent no-op when the processor's trigger didn't agree with the capability's threshold, and there was no way to retrieve the generated summary text. The compress decision is now owned solely by `SummarizationProcessor`, with the capability relaying its plan:

  - `on_before_compress(messages, cutoff_index)` now receives the real cutoff index. It fires between the processor's plan and execute steps, preserving the "before compression runs" timing promise.
  - `on_after_compress(messages, summarized, summary)` now takes a boolean outcome flag and the generated summary text (or `None`). Return a `str` to re-inject it as a `SystemPromptPart`; re-injection only happens when `summarized=True`.
  - `compact()`, `request_compact()`, and the `compact_conversation` tool pass `force=True` to the processor, so manual compaction is never silently vetoed by the trigger check.
  - `compression_count` increments only when a summary was actually produced (was previously incremented unconditionally on every `compact()` call).

  This is a breaking change to the `on_after_compress` signature (old `(messages) -> str | None`; new `(messages, summarized, summary) -> str | None`). `on_before_compress`'s signature is unchanged — old callbacks keep working as long as they ignore extra args.

### Added

- **`SummarizationProcessor.plan_compression(messages, *, force=False)`** — decide whether and where to compress without running the summary LLM. Returns a `CompressionPlan` (cutoff index + sliced messages + system parts) or `None` when no trigger matches.
- **`SummarizationProcessor.execute_plan(plan, focus=None)`** — run the summary LLM and build a `SummarizationResult`. On LLM failure, returns `summarized=False` with the original history reconstructed from the plan slices.
- **`SummarizationProcessor.process(messages, focus=None, *, force=False)`** — one-shot plan + execute returning a structured `SummarizationResult`.
- **`SummarizationResult`** dataclass — `messages`, `summarized`, `cutoff_index`, `summary`, `skip_reason`. Exposes what was previously invisible to callers (whether compression happened, why not, the generated text).
- **`CompressionPlan`** dataclass — cutoff index + sliced messages, used between plan and execute.
- **`SkipReason`** literal — `"not_triggered"`, `"cutoff_zero"`, `"failed"`.
- All five names exported from the package root.

### Fixed

- **Manual `compact()` no longer silently no-ops.** Previously `compact()` called the processor without bypassing triggers, so a manual compaction request could return the input unchanged with no signal to the caller. It now uses `force=True` and increments `compression_count` only when compression actually happened.

### Backwards compatibility

`SummarizationProcessor.__call__(messages, focus=None) -> list[ModelMessage]` is preserved as a thin wrapper around `process()`, so existing pydantic-ai `history_processors=[processor]` integrations keep working unchanged. Only callers that consume `on_after_compress` need to update their callback signature.

## [0.1.7] - 2026-06-04

### Fixed

- **Async `token_counter` now works end-to-end through the compression/cutoff path** ([#28](https://github.com/vstorm-co/summarization-pydantic-ai/issues/28)). Async support added in [#7](https://github.com/vstorm-co/summarization-pydantic-ai/pull/7) only covered the gating check; once compression actually ran, `SummarizationProcessor.__call__` still called the counter synchronously — both for the trigger total and inside the binary-search cutoff — so an async counter (e.g. a provider `count_tokens` API for accurate PDF/image counting) crashed with `TypeError: '>=' not supported between instances of 'coroutine' and 'int'` and a `RuntimeWarning: coroutine '...' was never awaited`. `__call__` now routes both invocations through `async_count_tokens` / `async_determine_cutoff_index`, so async counters work all the way through summarization. The synchronous helpers are unchanged for sync counters.

## [0.1.6] - 2026-05-24

### Changed

- **Docstring and import hygiene (internal; no behavior change).** Converted reStructuredText-style double-backtick inline code in docstrings and comments to single-backtick Markdown (18 occurrences), so it renders correctly under the mkdocstrings Markdown handler. Hoisted 15 function-local imports to module top where safe; intentionally-lazy, conditional, optional-dependency (`try`/`except ImportError`), and circular-import-avoidance imports were left in place.

### Fixed

- **Manual compaction focus topic is now honored** — `request_compact(focus=...)` and the `compact_conversation` tool's focus topic are forwarded through `ContextManagerCapability` into `SummarizationProcessor`, which appends a focus block to the summary prompt. Previously the focus was silently discarded. `SummarizationProcessor.__call__` and `compact()` now accept an optional `focus` argument.
- **`max_input_tokens=0` no longer silently disables fraction-based triggers** — fraction trigger and keep checks now use explicit `is not None` checks instead of truthiness, and `validate_triggers_and_keep` rejects `max_input_tokens <= 0` when a fraction-based trigger or keep value is used.
- **Summarization errors no longer overwrite or leak into conversation history** — when summary generation fails, the original message history is now returned unchanged and the error is logged, instead of replacing the pre-cutoff history with raw exception text (which could leak sensitive details into the model prompt).

### Documentation

- **Documentation accuracy pass and new example.** Created the missing **Context Manager** example page (token tracking via `on_usage_update`, tool-output truncation, the `compact_conversation` tool, and `compact()`/`request_compact()`), removed the nonexistent `hybrid` install extra, and documented the previously-omitted `ModelType`, `DEFAULT_CONTINUATION_PROMPT`, and `async_count_tokens` exports. Corrected the `TokenCounter` type (a sync-or-async union), the changelog note about `LimitWarnerProcessor` (it appends a `UserPromptPart`, not `SystemPromptPart`s), added a full `ContextManagerCapability` parameter table with verified defaults, and documented `keep_head` and the `warning_threshold`/`critical_remaining_iterations` knobs. `mkdocs build --strict` passes with zero warnings.

## [0.1.5] - 2026-05-24

### Infrastructure

- **`renovate.json`** ([#19](https://github.com/vstorm-co/summarization-pydantic-ai/pull/19)) — Renovate config landed; first auto-PRs (#20, #21) already merged.
- **CI: bump `docs.yml` Python to `3.14`** ([#20](https://github.com/vstorm-co/summarization-pydantic-ai/pull/20), Renovate auto-PR).
- **CI: bump `actions/checkout` to `v6`** across `ci.yml`, `docs.yml`, `publish.yml` ([#21](https://github.com/vstorm-co/summarization-pydantic-ai/pull/21), Renovate auto-PR).
- **CI: bump `astral-sh/setup-uv` to `v8.1.0`** across `ci.yml` (×3) and `publish.yml` — pulled in from Renovate's [Dependency Dashboard #22](https://github.com/vstorm-co/summarization-pydantic-ai/issues/22) (rate-limited there) and folded into this release. (Pinned to the specific patch because `astral-sh/setup-uv` does not maintain a rolling `v8` tag — only the `v7` and earlier majors do.)
- **CI: bump `actions/setup-python` to `v6`** in `docs.yml` — same source as above.

No source-code changes — pure CI / dependency-bot housekeeping. Library behaviour unchanged from 0.1.4.

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
