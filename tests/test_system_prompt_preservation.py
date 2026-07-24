"""Tests for system prompt preservation during compression."""

from __future__ import annotations

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel

from pydantic_ai_summarization.processor import (
    DEFAULT_CONTINUATION_PROMPT,
    SummarizationProcessor,
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


class TestSummariesDoNotAccumulate:
    """Repeated compressions must not pile stale summaries into the system channel.

    `_extract_system_prompts` carries every leading `SystemPromptPart` forward, so a
    summary emitted as a system part is re-extracted on the next compression and kept
    alongside the new one, growing without bound. Keeping the summary in a
    `UserPromptPart` stops the extraction walk at the summary instead.
    """

    async def test_only_the_real_system_prompt_survives_repeated_compressions(self):
        processor = SummarizationProcessor(
            model="openai:gpt-4.1", trigger=("messages", 1), keep=("messages", 0)
        )
        messages: list[ModelMessage] = [
            ModelRequest(parts=[SystemPromptPart(content="SYS"), UserPromptPart(content="q0")]),
            ModelResponse(parts=[TextPart(content="a0")]),
        ]

        for round_number in range(1, 4):
            processor._summarization_agent = Agent(
                FunctionModel(
                    lambda m, i, n=round_number: ModelResponse(
                        parts=[TextPart(content=f"SUMMARY-{n}")]
                    )
                )
            )
            result = await processor.process(messages)
            assert result.summarized is True
            messages = [
                *result.messages,
                ModelRequest(parts=[UserPromptPart(content=f"q{round_number}")]),
                ModelResponse(parts=[TextPart(content=f"a{round_number}")]),
            ]

        system_contents = [
            part.content
            for message in result.messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if isinstance(part, SystemPromptPart)
        ]
        assert system_contents == ["SYS"]


class TestContinuationPrompt:
    def test_default_value(self):
        assert DEFAULT_CONTINUATION_PROMPT == "Summary of previous conversation:\n\n"

    def test_exported(self):
        from pydantic_ai_summarization import DEFAULT_CONTINUATION_PROMPT as exported

        assert exported == "Summary of previous conversation:\n\n"
