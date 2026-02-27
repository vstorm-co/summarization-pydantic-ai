# Examples

All examples use low `max_tokens` so compression triggers quickly.

## Setup

```bash
# Install from PyPI
pip install summarization-pydantic-ai[hybrid]

# Or install locally (for development)
uv pip install -e ".[hybrid]"
```

Set your API key:
```bash
export OPENAI_API_KEY=sk-...
```

## Examples

| # | File | What it shows | API key? |
|---|------|---------------|----------|
| 01 | `01_basic_context_manager.py` | Auto-compression when hitting token limit, usage bar | Yes |
| 02 | `02_persistence_and_resume.py` | messages.json persistence, session resume | Yes |
| 03 | `03_callbacks_and_reinjection.py` | on_before/after_compress, re-inject instructions, focused compaction | Yes |
| 04 | `04_auto_max_tokens.py` | genai-prices auto-detection of context window | No |
| 05 | `05_interactive_chat.py` | Full interactive REPL with /compact, /context commands | Yes |
| 06 | `06_standalone_processors.py` | SummarizationProcessor vs SlidingWindowProcessor (no middleware) | Yes |

## Run

```bash
uv run python examples/01_basic_context_manager.py
uv run python examples/04_auto_max_tokens.py   # no API key needed
uv run python examples/05_interactive_chat.py   # interactive
```

## What to Look For

- **Example 01**: Watch the usage bar grow, then drop when auto-compression triggers
- **Example 02**: Session 2 loads messages from disk and continues the conversation
- **Example 03**: Callbacks fire before/after compression; critical instructions survive via re-injection
- **Example 04**: Context windows auto-detected for various models without any API calls
- **Example 05**: Try `/compact decorators` to focus compression, `/context` to see stats
- **Example 06**: Compare summarization (remembers all topics) vs sliding window (loses old ones)

---

<div align="center">

### Need help implementing this in your company?

<p>We're <a href="https://vstorm.co"><b>Vstorm</b></a> — an Applied Agentic AI Engineering Consultancy<br>with 30+ production AI agent implementations.</p>

<a href="https://vstorm.co/contact-us/">
  <img src="https://img.shields.io/badge/Talk%20to%20us%20%E2%86%92-0066FF?style=for-the-badge&logoColor=white" alt="Talk to us">
</a>

<br><br>

Made with ❤️ by <a href="https://vstorm.co"><b>Vstorm</b></a>

</div>
