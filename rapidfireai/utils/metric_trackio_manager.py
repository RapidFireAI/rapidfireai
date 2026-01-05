"""This module contains the TrackIOManager class which is responsible for managing the TrackIO runs."""

import trackio
from typing import Any
from rapidfireai.utils.metric_logger import MetricLogger, MetricLoggerType
from rapidfireai.evals.utils.logger import RFLogger


class TrackIOMetricLogger(MetricLogger):
    def __init__(self, experiment_name: str, logger: RFLogger = None, init_kwargs: dict[str, Any] = None):
        """
        Initialize TrackIO Manager.

        Args:
            init_kwargs: Initialization kwargs for TrackIO
        """
        self.init_kwargs = init_kwargs
        self.type = MetricLoggerType.TRACKIO
        if self.init_kwargs is None:
            self.init_kwargs = {"embed": False}
        if not isinstance(self.init_kwargs, dict):
            raise ValueError("init_kwargs must be a dictionary")
        self.experiment_name = experiment_name
        self.logger = logger if logger is not None else RFLogger()
        self.active_runs = {}  # Map run_id -> run_name
        self.run_params = {}  # Map run_id -> dict of params to log on init
        
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Ensure TrackIO is initialized with the experiment."""
        if not self._initialized and self.experiment_name:
            trackio.init(project=self.experiment_name, **self.init_kwargs)
            self._initialized = True

    def create_experiment(self, experiment_name: str) -> str:
        """Create a new experiment and set it as active."""
        self.experiment_name = experiment_name
        self._ensure_initialized()
        return experiment_name

    def get_experiment(self, experiment_name: str) -> str:
        """Get existing experiment by name and set it as active."""
        self.experiment_name = experiment_name
        self._ensure_initialized()
        return experiment_name

    def create_run(self, run_name: str) -> str:
        """Create a new run and return run_name as there is no run_id in TrackIO"""
        self._ensure_initialized()
        
        # TrackIO uses run names directly, so we use run_name as the run_id
        # Try to finish any existing run first
        try:
            trackio.finish()
        except Exception:
            pass  # No active run to finish
        
        # Initialize a new run with the run name
        try:
            trackio.init(project=self.experiment_name, name=run_name, **self.init_kwargs)
        except Exception:
            # If init doesn't accept name, try without it
            trackio.init(project=self.experiment_name, **self.init_kwargs)
        
        self.active_runs[run_name] = run_name
        # Log any pending params for this run
        if run_name in self.run_params:
            trackio.log(self.run_params[run_name])
            del self.run_params[run_name]
        
        return run_name

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Log parameters to a specific run."""
        # TrackIO logs params via the log() method
        # Try to log immediately, or store for later if run not active
        try:
            self._ensure_initialized()
            trackio.log({key: value})
        except Exception:
            # Run not active, store for later when run is created
            if run_id not in self.run_params:
                self.run_params[run_id] = {}
            self.run_params[run_id][key] = value

    def log_metric(self, _: str, key: str, value: float, step: int = None) -> None:
        """Log a metric to a specific run."""
        # TrackIO uses log() with step in the dict
        log_dict = {key: value}
        if step is not None:
            log_dict["step"] = step
        self._ensure_initialized()
        trackio.log(log_dict)

    def get_run_metrics(self, run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a specific run.
        
        Note: TrackIO stores metrics locally. This method returns an empty dict
        as TrackIO doesn't provide a direct API to retrieve historical metrics.
        Metrics can be viewed using `trackio.show()`.
        """
        # TrackIO doesn't provide a direct API to retrieve metrics programmatically
        # Metrics are stored locally and can be viewed via trackio.show()
        return {}

    def end_run(self, run_id: str) -> None:
        """End a specific run."""
        try:
            trackio.finish()
            if run_id in self.active_runs:
                del self.active_runs[run_id]
        except Exception as e:
            print(f"Error ending TrackIO run {run_id}: {e}")

    def delete_run(self, run_id: str) -> None:
        """Delete a specific run."""
        try:
            # TrackIO stores runs locally, deletion would require file system operations
            # For now, we just remove from tracking
            if run_id in self.active_runs:
                del self.active_runs[run_id]
            # Note: TrackIO doesn't have a delete_run API, runs are stored as local files
            print(f"Note: TrackIO runs are stored locally. To delete run '{run_id}', remove its files manually.")
        except Exception as e:
            raise ValueError(f"Run '{run_id}' not found: {e}")

    def clear_context(self) -> None:
        """Clear the TrackIO context by ending any active run."""
        try:
            trackio.finish()
            print("TrackIO context cleared successfully")
        except Exception:
            print("No active TrackIO run to clear")

