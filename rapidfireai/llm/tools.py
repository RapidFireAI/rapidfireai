"""Tool definitions and execution for the LLM assistant."""

from typing import Any

from rapidfireai.fit.db.rf_db import RfDb

# Tool definition sent to the Anthropic API
GET_RUN_DATA_TOOL = {
    "name": "get_run_data",
    "description": (
        "Get data for all runs in the current experiment. "
        "Returns each run's ID, status, hyperparameters, training progress, "
        "number of completed epochs, and any error messages."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

ALL_TOOLS = [GET_RUN_DATA_TOOL]


def execute_get_run_data(db: RfDb) -> list[dict[str, Any]]:
    """Query the database and return run data formatted for the LLM."""
    runs = db.get_all_runs()
    results = []
    for run_id, run in runs.items():
        results.append(
            {
                "run_id": run_id,
                "status": run["status"].value,
                "hyperparameters": run["flattened_config"],
                "completed_steps": run["completed_steps"],
                "total_steps": run["total_steps"],
                "num_epochs_completed": run["num_epochs_completed"],
                "error": run["error"] or None,
            }
        )
    return results


def make_tool_executor(db: RfDb):
    """Return a function that dispatches tool calls by name."""

    def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> Any:
        if tool_name == "get_run_data":
            return execute_get_run_data(db)
        return {"error": f"Unknown tool: {tool_name}"}

    return execute_tool
