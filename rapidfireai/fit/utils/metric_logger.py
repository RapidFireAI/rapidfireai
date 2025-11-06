"""
Metric Logger abstraction layer for RapidFire AI.

This module provides a unified interface for logging metrics to different backends
(MLflow, TensorBoard, or both). This abstraction allows minimal changes to core ML code
while supporting multiple tracking systems.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

# Note: MLflowManager is imported lazily in MLflowMetricLogger to avoid
# connection attempts when using tensorboard-only mode


class MetricLogger(ABC):
    """
    Abstract base class for metric logging.

    Provides a unified interface for logging metrics, parameters, and managing runs
    across different tracking backends (MLflow, TensorBoard, etc.).
    """

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
    def end_run(self, run_id: str) -> None:
        """
        End a specific run.

        Args:
            run_id: Run identifier
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

    def delete_run(self, run_id: str) -> None:
        """
        Delete a specific run (optional, not all backends support this).

        Args:
            run_id: Run identifier
        """
        pass

    def clear_context(self) -> None:
        """Clear the tracking context (optional, not all backends need this)."""
        pass


class MLflowMetricLogger(MetricLogger):
    """
    MLflow implementation of MetricLogger.

    Wraps the existing MLflowManager to provide the MetricLogger interface.
    """

    def __init__(self, tracking_uri: str):
        """
        Initialize MLflow metric logger.

        Args:
            tracking_uri: MLflow tracking server URI
        """
        # Lazy import to avoid connection attempts in tensorboard-only mode
        from rapidfireai.fit.utils.mlflow_manager import MLflowManager
        self.mlflow_manager = MLflowManager(tracking_uri)

    def get_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment by name.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Experiment ID
        """
        return self.mlflow_manager.get_experiment(experiment_name)

    def create_run(self, run_name: str) -> str:
        """Create a new MLflow run."""
        return self.mlflow_manager.create_run(run_name)

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Log a parameter to MLflow."""
        self.mlflow_manager.log_param(run_id, key, value)

    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """Log a metric to MLflow."""
        self.mlflow_manager.log_metric(run_id, key, value, step=step)

    def end_run(self, run_id: str) -> None:
        """End an MLflow run."""
        self.mlflow_manager.end_run(run_id)

    def get_run_metrics(self, run_id: str) -> dict:
        """Get metrics from MLflow."""
        return self.mlflow_manager.get_run_metrics(run_id)

    def delete_run(self, run_id: str) -> None:
        """Delete an MLflow run."""
        self.mlflow_manager.delete_run(run_id)

    def clear_context(self) -> None:
        """Clear MLflow context."""
        self.mlflow_manager.clear_context()


class TensorBoardMetricLogger(MetricLogger):
    """
    TensorBoard implementation of MetricLogger.

    Uses torch.utils.tensorboard.SummaryWriter to log metrics to TensorBoard.
    """

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard metric logger.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writers = {}  # Map run_id -> SummaryWriter

    def create_run(self, run_name: str) -> str:
        """
        Create a new TensorBoard run.

        For TensorBoard, we use run_name as the run_id and create a subdirectory
        in the log directory.
        """
        from torch.utils.tensorboard import SummaryWriter

        run_log_dir = self.log_dir / run_name
        run_log_dir.mkdir(parents=True, exist_ok=True)

        # Create SummaryWriter for this run
        writer = SummaryWriter(log_dir=str(run_log_dir))
        self.writers[run_name] = writer

        return run_name

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """
        Log a parameter to TensorBoard.

        TensorBoard doesn't have native parameter logging, so we log as text.
        """
        if run_id not in self.writers:
            self.create_run(run_id)

        writer = self.writers[run_id]
        writer.add_text(f"params/{key}", str(value), global_step=0)
        writer.flush()

    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """
        Log a metric to TensorBoard.

        Args:
            run_id: Run identifier
            key: Metric name
            value: Metric value
            step: Step number (required for TensorBoard time series)
        """
        if run_id not in self.writers:
            self.create_run(run_id)

        writer = self.writers[run_id]
        # Use step=0 if not provided (fallback)
        writer.add_scalar(key, value, global_step=step if step is not None else 0)
        # Flush immediately to ensure real-time updates
        writer.flush()

    def end_run(self, run_id: str) -> None:
        """End a TensorBoard run by closing the writer."""
        if run_id in self.writers:
            self.writers[run_id].close()
            del self.writers[run_id]

    def get_run_metrics(self, run_id: str) -> dict:
        """
        Get metrics from TensorBoard.

        Note: TensorBoard doesn't provide easy API access to logged metrics.
        This returns an empty dict. For viewing metrics, use TensorBoard UI.
        """
        return {}

    def delete_run(self, run_id: str) -> None:
        """
        Delete a TensorBoard run by moving its directory outside the log tree (soft delete).

        This is a soft delete - the data is moved to a sibling '{log_dir}_deleted' directory
        outside TensorBoard's scan path, so it won't appear in the UI. Data can be manually
        recovered if needed by moving it back to the log_dir.

        Args:
            run_id: Run identifier (directory name)
        """
        import shutil
        import time

        # Close and remove writer if active
        if run_id in self.writers:
            self.writers[run_id].close()
            del self.writers[run_id]

        # Move the run directory to sibling deleted folder (outside log_dir tree)
        run_log_dir = self.log_dir / run_id
        if run_log_dir.exists() and run_log_dir.is_dir():
            # Create deleted directory as sibling, not child, of log_dir
            deleted_dir = self.log_dir.parent / f"{self.log_dir.name}_deleted"
            deleted_dir.mkdir(exist_ok=True)

            # Add timestamp to avoid name collisions
            timestamp = int(time.time())
            destination = deleted_dir / f"{run_id}_{timestamp}"

            shutil.move(str(run_log_dir), str(destination))
    
    def __del__(self):
        """Clean up all writers on deletion."""
        for writer in self.writers.values():
            writer.close()


class DualMetricLogger(MetricLogger):
    """
    Dual implementation that logs to both MLflow and TensorBoard.

    This allows users to benefit from both tracking systems simultaneously:
    - MLflow for experiment comparison and model registry
    - TensorBoard for real-time training visualization (especially useful in Colab)
    """

    def __init__(self, mlflow_tracking_uri: str, tensorboard_log_dir: str):
        """
        Initialize dual metric logger.

        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            tensorboard_log_dir: Directory for TensorBoard logs
        """
        self.mlflow_logger = MLflowMetricLogger(mlflow_tracking_uri)
        self.tensorboard_logger = TensorBoardMetricLogger(tensorboard_log_dir)

    def get_experiment(self, experiment_name: str) -> str:
        """Get experiment from MLflow (TensorBoard doesn't have experiments)."""
        return self.mlflow_logger.get_experiment(experiment_name)

    def create_run(self, run_name: str) -> str:
        """Create run in both MLflow and TensorBoard."""
        mlflow_run_id = self.mlflow_logger.create_run(run_name)
        self.tensorboard_logger.create_run(run_name)
        # Return MLflow run_id as the canonical ID
        return mlflow_run_id

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Log parameter to both backends."""
        self.mlflow_logger.log_param(run_id, key, value)
        self.tensorboard_logger.log_param(run_id, key, value)

    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to both backends."""
        self.mlflow_logger.log_metric(run_id, key, value, step=step)
        self.tensorboard_logger.log_metric(run_id, key, value, step=step)

    def end_run(self, run_id: str) -> None:
        """End run in both backends."""
        self.mlflow_logger.end_run(run_id)
        self.tensorboard_logger.end_run(run_id)

    def get_run_metrics(self, run_id: str) -> dict:
        """Get metrics from MLflow (primary source)."""
        return self.mlflow_logger.get_run_metrics(run_id)

    def delete_run(self, run_id: str) -> None:
        """Delete run from both MLflow and TensorBoard."""
        self.mlflow_logger.delete_run(run_id)
        self.tensorboard_logger.delete_run(run_id)

    def clear_context(self) -> None:
        """Clear context in both backends."""
        self.mlflow_logger.clear_context()


def create_metric_logger(
    backend: str,
    mlflow_tracking_uri: Optional[str] = None,
    tensorboard_log_dir: Optional[str] = None,
) -> MetricLogger:
    """
    Factory function to create the appropriate metric logger.

    Args:
        backend: Tracking backend to use ('mlflow', 'tensorboard', or 'both')
        mlflow_tracking_uri: MLflow tracking server URI (required if backend includes MLflow)
        tensorboard_log_dir: TensorBoard log directory (required if backend includes TensorBoard)

    Returns:
        MetricLogger instance

    Raises:
        ValueError: If backend is invalid or required parameters are missing
    """
    backend = backend.lower()

    if backend == "mlflow":
        if not mlflow_tracking_uri:
            raise ValueError("mlflow_tracking_uri required for MLflow backend")
        return MLflowMetricLogger(mlflow_tracking_uri)

    elif backend == "tensorboard":
        if not tensorboard_log_dir:
            raise ValueError("tensorboard_log_dir required for TensorBoard backend")
        return TensorBoardMetricLogger(tensorboard_log_dir)

    elif backend == "both":
        if not mlflow_tracking_uri or not tensorboard_log_dir:
            raise ValueError("Both mlflow_tracking_uri and tensorboard_log_dir required for dual backend")
        return DualMetricLogger(mlflow_tracking_uri, tensorboard_log_dir)

    else:
        raise ValueError(f"Invalid backend: {backend}. Must be 'mlflow', 'tensorboard', or 'both'")
