"""Thin Anthropic API client with tool-use loop."""

import os
from typing import Any

from anthropic import Anthropic


def get_client() -> Anthropic:
    """Create an Anthropic client. Raises if ANTHROPIC_API_KEY is not set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")
    return Anthropic(api_key=api_key)


def get_model() -> str:
    return os.environ.get("RF_LLM_MODEL", "claude-sonnet-4-5-20250929")


def chat_with_tools(
    client: Anthropic,
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    execute_tool: callable,
    max_iterations: int = 5,
) -> str:
    """Send messages to the API and handle the tool-use loop.

    Returns the final text response from the model.
    """
    for _ in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system,
            messages=messages,
            tools=tools,
        )

        # If the model is done talking, return the text
        if response.stop_reason == "end_turn":
            text_blocks = [block.text for block in response.content if block.type == "text"]
            return "\n".join(text_blocks)

        # If the model wants to use a tool, execute it and continue
        if response.stop_reason == "tool_use":
            # Append the assistant's response (contains tool_use blocks)
            messages.append({"role": "assistant", "content": response.content})

            # Execute each tool call and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
            continue

        # Unexpected stop reason — return whatever text we have
        text_blocks = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_blocks) if text_blocks else "No response generated."

    return "Max tool iterations reached. Please try a simpler question."
