"""Tests for system prompt preservation during compression."""

from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)

from pydantic_ai_summarization.processor import (
    DEFAULT_CONTINUATION_PROMPT,
    _extract_system_prompts,
)


class TestExtractSystemPrompts:
    """Tests for _extract_system_prompts helper."""

    def test_extracts_leading_system_parts(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="You are a helper."),
                    SystemPromptPart(content="Available tools: read_file."),
                    UserPromptPart(content="Hello"),
                ]
            ),
        ]
        parts = _extract_system_prompts(messages)
        assert len(parts) == 2
        assert parts[0].content == "You are a helper."
        assert parts[1].content == "Available tools: read_file."

    def test_stops_at_non_system_part(self):
        messages: list[ModelMessage] = [
            ModelRequest(
                parts=[
                    SystemPromptPart(content="System prompt"),
                    UserPromptPart(content="User message"),
                    SystemPromptPart(content="This should NOT be extracted"),
                ]
            ),
        ]
        parts = _extract_system_prompts(messages)
        assert len(parts) == 1
        assert parts[0].content == "System prompt"

    def test_stops_at_non_request_message(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="Prompt 1")]),
            ModelResponse(parts=[TextPart(content="Response")]),
            ModelRequest(parts=[SystemPromptPart(content="Not extracted")]),
        ]
        parts = _extract_system_prompts(messages)
        assert len(parts) == 1

    def test_empty_messages(self):
        parts = _extract_system_prompts([])
        assert parts == []

    def test_no_system_parts(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[UserPromptPart(content="Hello")]),
        ]
        parts = _extract_system_prompts(messages)
        assert parts == []

    def test_multiple_request_messages_with_system(self):
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="Prompt A")]),
            ModelRequest(parts=[SystemPromptPart(content="Prompt B")]),
            ModelResponse(parts=[TextPart(content="Hi")]),
        ]
        parts = _extract_system_prompts(messages)
        assert len(parts) == 2
        assert parts[0].content == "Prompt A"
        assert parts[1].content == "Prompt B"


class TestContinuationPrompt:
    def test_default_value(self):
        assert DEFAULT_CONTINUATION_PROMPT == "Summary of previous conversation:\n\n"

    def test_exported(self):
        from pydantic_ai_summarization import DEFAULT_CONTINUATION_PROMPT as exported

        assert exported == "Summary of previous conversation:\n\n"
