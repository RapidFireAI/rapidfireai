"""
Metric Logger abstraction layer for RapidFire AI.

This module provides a unified interface for logging metrics to different backends
(MLflow, TensorBoard, Trackio, or combinations). This abstraction allows minimal changes to core ML code
while supporting multiple tracking systems.
"""

from abc import ABC, abstractmethod
from typing import Optional, TypedDict, Any
from enum import Enum


class MetricLogger(ABC):
    """
    Abstract base class for metric logging.

    Provides a unified interface for logging metrics, parameters, and managing runs
    across different tracking backends (MLflow, TensorBoard, etc.).
    """

    @abstractmethod
    def create_experiment(self, experiment_name: str) -> str:
        """
        Create a new experiment and return experiment_id.
        """
        pass

    @abstractmethod
    def get_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment by name and set it as active.
        """
        pass

    @abstractmethod
    def create_run(self, run_name: str) -> str:
        """
        Create a new run and return run_id.

        Args:
            run_name: Name for the run

        Returns:
            Run ID string
        """
        pass

    @abstractmethod
    def log_param(self, run_id: str, key: str, value: str) -> None:
        """
        Log a parameter to a specific run.

        Args:
            run_id: Run identifier
            key: Parameter name
            value: Parameter value
        """
        pass

    @abstractmethod
    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to a specific run.

        Args:
            run_id: Run identifier
            key: Metric name
            value: Metric value
            step: Optional step number for the metric
        """
        pass

    @abstractmethod
    def get_run_metrics(self, run_id: str) -> dict:
        """
        Get all metrics for a specific run.

        Args:
            run_id: Run identifier
        Returns:
            Dictionary of metrics
        """
        pass

    @abstractmethod
    def end_run(self, run_id: str, status: Optional[str] = None) -> None:
        """
        End a specific run.

        Args:
            run_id: Run identifier
            status: Optional terminal status to record at the backend. MLflow
                ``RunStatus`` string (``"FINISHED"``, ``"FAILED"``, ``"KILLED"``).
                When ``None``, backends should fall back to their default
                terminal state (today: MLflow ``FINISHED``) so external callers
                that haven't been updated yet keep working.
        """
        pass

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """
        Delete a specific run (optional, not all backends support this).

        Args:
            run_id: Run identifier
        """
        pass

    @abstractmethod
    def clear_context(self) -> None:
        """Clear the tracking context (optional, not all backends need this)."""
        pass

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """
        Set a tag on a specific run.

        Unlike :meth:`log_param`, tags are mutable -- calling ``set_tag``
        again with the same key overwrites the previous value. Used for
        properties that evolve over a run's lifetime (e.g. progress
        counters) that ``log_param`` cannot represent.

        Args:
            run_id: Run identifier.
            key: Tag name.
            value: Tag value (coerced to string by most backends).

        The default implementation is a no-op so backends without a
        native tag concept (TensorBoard, Trackio) don't need to implement
        it. MLflow overrides this with a real ``set_tag`` call.
        """
        return None

class MetricLoggerType(Enum):
    """Enum for MetricLogger types."""
    MLFLOW = "mlflow"
    TENSORBOARD = "tensorboard"
    TRACKIO = "trackio"
    MULTIPLE = "multiple"

class MetricLoggerConfig(TypedDict):
    """Config for MetricLogger."""
    type: MetricLoggerType
    config: Any
