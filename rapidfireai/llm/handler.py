"""Request handler for the LLM chat endpoint."""

from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.llm.client import chat_with_tools, get_client, get_model
from rapidfireai.llm.tools import ALL_TOOLS, make_tool_executor

SYSTEM_PROMPT = (
    "You are an AI assistant for RapidFire AI, an experiment execution framework "
    "for LLM fine-tuning. You help users understand their training runs.\n\n"
    "You have access to a tool that retrieves run data from the current experiment. "
    "Use it to answer questions about run status, hyperparameters, progress, and errors.\n\n"
    "Keep your answers concise and focused on the data."
)


def handle_chat(db: RfDb, user_message: str) -> str:
    """Process a user chat message and return the LLM's response."""
    client = get_client()
    model = get_model()
    execute_tool = make_tool_executor(db)

    messages = [{"role": "user", "content": user_message}]

    return chat_with_tools(
        client=client,
        model=model,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=ALL_TOOLS,
        execute_tool=execute_tool,
    )
