# Basic Usage

Get started with conversation context management.

## Simple Agent with Summarization

The simplest way to add summarization to an agent:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

# Create processor with defaults
processor = create_summarization_processor()

# Create agent with processor
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

# Use the agent normally
async def main():
    result = await agent.run("Hello!")
    print(result.output)
```

## Simple Agent with Sliding Window

Zero-cost context management:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

# Create processor with defaults
processor = create_sliding_window_processor()

# Create agent with processor
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

# Use the agent normally
async def main():
    result = await agent.run("Hello!")
    print(result.output)
```

## Custom Thresholds

Configure when processing triggers:

=== "Summarization"

    ```python
    processor = create_summarization_processor(
        trigger=("tokens", 50000),   # Trigger at 50k tokens
        keep=("messages", 15),        # Keep last 15 messages
    )
    ```

=== "Sliding Window"

    ```python
    processor = create_sliding_window_processor(
        trigger=("tokens", 50000),   # Trigger at 50k tokens
        keep=("messages", 25),        # Keep last 25 messages
    )
    ```

## Message-Based Triggers

Trigger based on message count:

=== "Summarization"

    ```python
    processor = create_summarization_processor(
        trigger=("messages", 30),    # Trigger at 30 messages
        keep=("messages", 10),       # Keep last 10 messages
    )
    ```

=== "Sliding Window"

    ```python
    processor = create_sliding_window_processor(
        trigger=("messages", 100),   # Trigger at 100 messages
        keep=("messages", 50),       # Keep last 50 messages
    )
    ```

## Conversation Loop with Summarization

Use summarization with a conversation loop:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_summarization_processor

processor = create_summarization_processor(
    trigger=("messages", 20),
    keep=("messages", 5),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

async def chat():
    message_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        result = await agent.run(
            user_input,
            message_history=message_history,
        )

        print(f"Assistant: {result.output}")
        message_history = result.all_messages()

if __name__ == "__main__":
    import asyncio
    asyncio.run(chat())
```

## Conversation Loop with Sliding Window

Use sliding window for faster processing:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

processor = create_sliding_window_processor(
    trigger=("messages", 50),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

async def chat():
    message_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break

        result = await agent.run(
            user_input,
            message_history=message_history,
        )

        print(f"Assistant: {result.output}")
        message_history = result.all_messages()

if __name__ == "__main__":
    import asyncio
    asyncio.run(chat())
```

## With Dependencies

Use with agent dependencies:

=== "Summarization"

    ```python
    from dataclasses import dataclass
    from pydantic_ai import Agent
    from pydantic_ai_summarization import create_summarization_processor

    @dataclass
    class MyDeps:
        user_id: str

    processor = create_summarization_processor()

    agent = Agent(
        "openai:gpt-4.1",
        deps_type=MyDeps,
        history_processors=[processor],
    )

    async def main():
        deps = MyDeps(user_id="user123")
        result = await agent.run("Hello!", deps=deps)
        print(result.output)
    ```

=== "Sliding Window"

    ```python
    from dataclasses import dataclass
    from pydantic_ai import Agent
    from pydantic_ai_summarization import create_sliding_window_processor

    @dataclass
    class MyDeps:
        user_id: str

    processor = create_sliding_window_processor()

    agent = Agent(
        "openai:gpt-4.1",
        deps_type=MyDeps,
        history_processors=[processor],
    )

    async def main():
        deps = MyDeps(user_id="user123")
        result = await agent.run("Hello!", deps=deps)
        print(result.output)
    ```

## With Tools

Both processors work seamlessly with tool-using agents:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import create_sliding_window_processor

processor = create_sliding_window_processor(
    trigger=("messages", 50),
    keep=("messages", 20),
)

agent = Agent(
    "openai:gpt-4.1",
    history_processors=[processor],
)

@agent.tool_plain
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

@agent.tool_plain
def search(query: str) -> str:
    """Search for information."""
    return f"Found 5 results for: {query}"

async def main():
    result = await agent.run("What's the weather in New York?")
    print(result.output)
```

!!! note "Tool Call Safety"
    Both processors automatically preserve tool call/response pairs when trimming messages. A tool call will never be separated from its response.

## Comparing Processors

Here's a side-by-side comparison of both processors with the same configuration:

```python
from pydantic_ai import Agent
from pydantic_ai_summarization import (
    create_summarization_processor,
    create_sliding_window_processor,
)

# Summarization: Uses LLM to create intelligent summary
summarizer = create_summarization_processor(
    trigger=("messages", 30),
    keep=("messages", 10),
)

# Sliding Window: Simply discards old messages (no LLM cost)
window = create_sliding_window_processor(
    trigger=("messages", 30),
    keep=("messages", 10),
)

# Choose based on your needs
agent = Agent(
    "openai:gpt-4.1",
    history_processors=[summarizer],  # or [window]
)
```
