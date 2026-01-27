"""
Shared utilities for fit and evals dispatchers.
"""

from typing import Any, Protocol


class DatabaseWithExperiment(Protocol):
    """Protocol for database classes that support get_running_experiment."""

    def get_running_experiment(self) -> dict[str, Any] | None: ...


def check_experiment_running(db: DatabaseWithExperiment, experiment_name: str) -> bool:
    """
    Check if a specific experiment is currently running.

    Works with both fit and evals db interfaces.

    Args:
        db: Database instance with get_running_experiment() method
        experiment_name: Name of the experiment to check

    Returns:
        True if the experiment is currently running, False otherwise
    """
    try:
        running_experiment = db.get_running_experiment()
        running_name = running_experiment.get("experiment_name") if running_experiment else None
        running_status = running_experiment.get("status") if running_experiment else None
        return running_name == experiment_name and running_status == "running"
    except Exception:
        return False
