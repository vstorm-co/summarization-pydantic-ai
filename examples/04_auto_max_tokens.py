"""Auto-detection of max_tokens from genai-prices.

Shows:
- model_name → genai-prices → context_window auto-detection
- Fallback to 200,000 when model is unknown
- resolve_max_tokens() standalone usage

Run:
    uv run python examples/04_auto_max_tokens.py
"""

from pydantic_ai_summarization import resolve_max_tokens

MODELS = [
    "openai:gpt-4.1",
    "openai:gpt-4.1-mini",
    "openai:gpt-4o",
    "openai:o3-mini",
    "anthropic:claude-sonnet-4-20250514",
    "anthropic:claude-haiku-4-5-20251001",
    "google-gla:gemini-2.0-flash",
    "openrouter:openai/gpt-4.1",
    "unknown-provider:nonexistent-model",
]


def main() -> None:
    print("Auto-detecting context windows from genai-prices")
    print("=" * 65)
    print(f"{'Model':<45} {'Context Window':>18}")
    print("-" * 65)

    for model in MODELS:
        ctx_window = resolve_max_tokens(model)
        if ctx_window is not None:
            print(f"{model:<45} {ctx_window:>15,} tokens")
        else:
            print(f"{model:<45} {'NOT FOUND (→ 200K)':>18}")

    print("-" * 65)
    print("\nWhen max_tokens=None (default), ContextManagerMiddleware")
    print("calls resolve_max_tokens(model_name) in __post_init__.")
    print("If not found, falls back to 200,000.")


if __name__ == "__main__":
    main()
