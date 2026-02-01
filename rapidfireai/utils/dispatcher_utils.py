"""
Shared utilities for fit and evals dispatchers.
"""

from typing import Any, Protocol

from rapidfireai.utils.constants import ExperimentStatus


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
        if not running_experiment:
            return False
        running_name = running_experiment.get("experiment_name")
        running_status = running_experiment.get("status")
        # Compare with enum - works for both enum values and strings due to str inheritance
        is_running = running_status == ExperimentStatus.RUNNING
        return running_name == experiment_name and is_running
    except Exception:
        return False
