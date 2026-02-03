"""
TensorBoard implementation of MetricLogger.

Uses torch.utils.tensorboard.SummaryWriter to log metrics to TensorBoard.
"""

from rapidfireai.utils.metric_logger import MetricLogger, MetricLoggerType
from pathlib import Path
from typing import Optional, Any
import os
from rapidfireai.utils.os_utils import mkdir_p
from rapidfireai.evals.utils.logger import RFLogger

class TensorBoardMetricLogger(MetricLogger):
    """
    TensorBoard implementation of MetricLogger.

    Uses torch.utils.tensorboard.SummaryWriter to log metrics to TensorBoard.
    """

    def __init__(self, log_dir: str, logger: RFLogger = None, init_kwargs: dict[str, Any] = None):
        """
        Initialize TensorBoard metric logger.

        Args:
            log_dir: Directory for TensorBoard logs
            init_kwargs: Initialization kwargs for TensorBoard
        """
        from torch.utils.tensorboard import SummaryWriter

        self.type = MetricLoggerType.TENSORBOARD
        self.log_dir = Path(log_dir)
        self.logger = logger if logger is not None else RFLogger()
        self.init_kwargs = init_kwargs # Not currently used
        try:
            mkdir_p(self.log_dir, notify=False)
        except (PermissionError, OSError) as e:
            self.logger.error(f"Error creating directory: {e}")
            raise
        self.writers = {}  # Map run_id -> SummaryWriter

    def create_experiment(self, experiment_name: str) -> str:
        """
        Create a new TensorBoard experiment.
        """
        return experiment_name

    def get_experiment(self, experiment_name: str) -> str:
        """
        Get existing TensorBoard experiment by name and set it as active.
        """
        return experiment_name

    def create_run(self, run_name: str) -> str:
        """
        Create a new TensorBoard run.

        For TensorBoard, we use run_name as the run_id and create a subdirectory
        in the log directory.
        """
        from torch.utils.tensorboard import SummaryWriter

        run_log_dir = os.path.join(self.log_dir, run_name)
        try:
            mkdir_p(run_log_dir, notify=False)
        except (PermissionError, OSError) as e:
            self.logger.error(f"Error creating directory: {e}")
            raise

        # Create SummaryWriter for this run
        writer = SummaryWriter(log_dir=run_log_dir)
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

    def get_run_metrics(self, run_id: str) -> dict:
        """
        Get metrics from TensorBoard.

        Note: TensorBoard doesn't provide easy API access to logged metrics.
        This returns an empty dict. For viewing metrics, use TensorBoard UI.
        """
        return {}
    
    def end_run(self, run_id: str) -> None:
        """End a TensorBoard run by closing the writer."""
        if run_id in self.writers:
            self.writers[run_id].close()
            del self.writers[run_id]


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
        run_log_dir = os.path.join(self.log_dir, run_id)
        if os.path.exists(run_log_dir) and os.path.isdir(run_log_dir):
            # Create deleted directory as sibling, not child, of log_dir
            deleted_dir = os.path.join(self.log_dir.parent, f"{self.log_dir.name}_deleted")
            try:
                mkdir_p(deleted_dir, notify=False)
            except (PermissionError, OSError) as e:
                self.logger.error(f"Error creating directory: {e}")
                raise

            # Add timestamp to avoid name collisions
            timestamp = int(time.time())
            destination = os.path.join(deleted_dir, f"{run_id}_{timestamp}")

            shutil.move(run_log_dir, destination)
    
    def __del__(self):
        """Clean up all writers on deletion."""
        for writer in self.writers.values():
            writer.close()

    def clear_context(self) -> None:
        """Clear the TensorBoard context."""
        if self.writers:
            # Iterate over a snapshot since `end_run()` mutates `self.writers`.
            for run_id in list(self.writers):
                self.end_run(run_id)
        else:
            self.logger.info("No active TensorBoard runs to clear")

