# LLM Chat Assistant - Frontend Integration Spec

## Overview

A new dispatcher endpoint that lets users ask natural-language questions about their training runs. The backend sends the question to Claude (Anthropic API), which can call a tool to fetch run data from the database, then returns a text answer.

## Endpoint

### `POST /dispatcher/llm-chat`

**Base URL:** `http://localhost:8851`

**Request:**

```json
{
  "message": "Which run is performing best?"
}
```

| Field     | Type   | Required | Description                        |
|-----------|--------|----------|------------------------------------|
| `message` | string | Yes      | The user's question in plain text  |

**Success Response (200):**

```json
{
  "response": "Run 2 is the furthest along with 450/500 steps completed. Run 1 has completed 300/500 steps. Both are using a learning rate of 1e-4 but Run 2 has a batch size of 16 vs Run 1's 8."
}
```

| Field      | Type   | Description                          |
|------------|--------|--------------------------------------|
| `response` | string | The LLM's answer in plain text/markdown |

**Error Response (400) - Missing message:**

```json
{
  "error": "message is required"
}
```

**Error Response (500) - API key not set:**

```json
{
  "error": "ANTHROPIC_API_KEY environment variable is not set"
}
```

**Error Response (500) - General error:**

```json
{
  "error": "...",
}
```

## Behavior

- **No streaming.** The endpoint blocks until the full response is ready. Expect **3-10 seconds** latency depending on whether the LLM calls the tool.
- **No conversation memory.** Each request is independent. The frontend must not assume multi-turn context.
- **Read-only.** The LLM can only read run data. It cannot trigger IC Ops (stop, resume, clone, delete).
- **Requires a running experiment.** The tool queries `get_all_runs()` from the database, which requires an active experiment with runs.

## Data the LLM Can Access

When the LLM calls its `get_run_data` tool, it receives this structure per run:

```json
{
  "run_id": 1,
  "status": "Ongoing",
  "hyperparameters": {
    "learning_rate": 1e-4,
    "batch_size": 8,
    "num_train_epochs": 3
  },
  "completed_steps": 300,
  "total_steps": 500,
  "num_epochs_completed": 1,
  "error": null
}
```

| Field                 | Type        | Description                                         |
|-----------------------|-------------|-----------------------------------------------------|
| `run_id`              | int         | Unique run identifier                               |
| `status`              | string      | `"New"`, `"Ongoing"`, `"Stopped"`, `"Completed"`, `"Failed"`, `"Deleted"` |
| `hyperparameters`     | object      | Flat key-value pairs of the run's config             |
| `completed_steps`     | int         | Training steps completed so far                     |
| `total_steps`         | int         | Total training steps expected                       |
| `num_epochs_completed`| int         | Full epochs completed                               |
| `error`               | string/null | Error message if the run failed                     |

## Example Questions the LLM Can Answer

- "How many runs are active?"
- "Which run has the most progress?"
- "Compare the hyperparameters of run 1 and run 2"
- "Did any runs fail? What went wrong?"
- "What learning rates are being tested?"
- "Summarize the current experiment status"

## Frontend Integration Example

```typescript
const LLM_CHAT_URL = "http://localhost:8851/dispatcher/llm-chat";

async function sendChatMessage(message: string): Promise<string> {
  const response = await fetch(LLM_CHAT_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.error || "Chat request failed");
  }

  const data = await response.json();
  return data.response;
}
```

## UI Recommendations

- **Loading state:** Show a spinner or typing indicator while waiting. The request can take several seconds.
- **Error display:** Show error messages from the `error` field in the response. The most common error is a missing API key.
- **Markdown rendering:** The LLM response may contain markdown (bold, lists, code blocks). Render accordingly.
- **Input:** A single text input with a send button. No conversation history needed for MVP.
- **Availability:** The chat feature only works when an experiment is running. Consider disabling the input or showing a message when no experiment is active.

## Environment Variables (Backend)

| Variable            | Required | Default                        | Description            |
|---------------------|----------|--------------------------------|------------------------|
| `ANTHROPIC_API_KEY` | Yes      | —                              | Anthropic API key      |
| `RF_LLM_MODEL`      | No       | `claude-sonnet-4-5-20250929` | Model to use           |

## Backend Architecture (for reference)

All LLM code is isolated in `rapidfireai/llm/`:

```
rapidfireai/llm/
├── __init__.py    # empty
├── client.py      # Anthropic SDK wrapper, tool-use loop
├── tools.py       # get_run_data tool definition + executor
└── handler.py     # Builds system prompt, wires client + tools
```

The only integration point is a single route in `rapidfireai/fit/dispatcher/dispatcher.py` that calls `handle_chat()` with the existing `RfDb` instance.

### Tool-use Flow (internal)

```
Frontend POST /dispatcher/llm-chat { "message": "..." }
  → handler.py builds system prompt + messages
  → client.py sends to Anthropic API with tool definitions
  → API returns tool_use (wants to call get_run_data)
  → tools.py executes get_run_data → queries RfDb.get_all_runs()
  → client.py sends tool result back to API
  → API returns end_turn with text answer
  → handler.py returns the text
  ← { "response": "..." }
```

Max tool iterations: 5 (safety limit). In practice, the LLM calls the tool once and answers.

## Future Considerations (not in MVP)

- Streaming responses (SSE)
- Conversation history (multi-turn)
- IC Ops tools (stop/resume/clone via chat)
- MLflow metric data (loss curves, accuracy)
