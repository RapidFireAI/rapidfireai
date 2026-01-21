"""This module contains the TrackioManager class which is responsible for managing the Trackio runs."""

import time
import trackio
from typing import Any
from rapidfireai.utils.metric_logger import MetricLogger, MetricLoggerType
from rapidfireai.evals.utils.logger import RFLogger
import warnings

warnings.filterwarnings("ignore", message="Reserved keys renamed")

class TrackioMetricLogger(MetricLogger):
    def __init__(self, experiment_name: str, logger: RFLogger = None, init_kwargs: dict[str, Any] = None):
        """
        Initialize Trackio Manager.

        Args:
            init_kwargs: Initialization kwargs for Trackio
        """
        self.init_kwargs = init_kwargs
        self.type = MetricLoggerType.TRACKIO
        if self.init_kwargs is None:
            self.init_kwargs = {"embed": False}
        if not isinstance(self.init_kwargs, dict):
            raise ValueError("init_kwargs must be a dictionary")
        self.api = trackio.Api()
        self.experiment_name = experiment_name
        self.logger = logger if logger is not None else RFLogger()
        self.active_runs = {}  # Map run_id -> run_name
        self.run_params = {}  # Map run_id -> dict of params to log on init
        
        self._initialized = False

    def create_experiment(self, experiment_name: str) -> str:
        """Create a new experiment and set it as active."""
        # No need to create an experiment in Trackio, it is created automatically when the first run is created, so we just set the experiment name
        self.experiment_name = experiment_name
        return experiment_name

    def get_experiment(self, experiment_name: str) -> str:
        """Get existing experiment by name and set it as active."""
        # No specific experiment with Trackio, so we just set the experiment name
        self.experiment_name = experiment_name
        return experiment_name

    def create_run(self, run_name: str) -> str:
        """Create a new run and return run_name as there is no run_id in Trackio"""
        self.logger.info(f"Creating a run for Trackio: {run_name}")
        # Initialize a new run with the run name 
        try:
            self.active_runs[run_name] = trackio.init(project=self.experiment_name, name=run_name, **self.init_kwargs)
        except Exception:
            raise ValueError(f"Exception in calling trackio.init() to create new run: {run_name} with {self.init_kwargs=}")
        
        # Log any pending params for this run
        if run_name in self.run_params:
            trackio.log(self.run_params[run_name])
            del self.run_params[run_name]
        
        return run_name

    def log_param(self, run_name: str, key: str, value: str) -> None:
        """Log parameters to a specific run."""
        # Trackio logs params via the log() method
        # Try to log immediately, or store for later if run not active
        try:
            self.active_runs[run_name].log({key: value})
        except Exception:
            # Run not active, store for later when run is created
            if run_name not in self.run_params:
                self.run_params[run_name] = {}
            self.run_params[run_name][key] = value

    def log_metric(self, run_name: str, key: str, value: float, step: int = None) -> None:
        """Log a metric to a specific run."""
        # Trackio uses log() with step in the dict
        log_dict = {key: value}
        if step is not None:
            log_dict["step"] = step
        try:
            self.active_runs[run_name].log(log_dict)
        except Exception:
            raise ValueError(f"Error logging metric in log_metric, is there not an active run?: {run_name=}, {key} = {value}, {step=}")

    def get_run_metrics(self, run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a specific run.
        
        Note: Trackio stores metrics locally. This method returns an empty dict
        as Trackio doesn't provide a direct API to retrieve historical metrics.
        Metrics can be viewed using `trackio.show()`.
        """
        # Trackio doesn't provide a direct API to retrieve metrics programmatically
        # Metrics are stored locally and can be viewed via trackio.show()
        return {}

    def end_run(self, run_name: str) -> None:
        """End a specific run."""
        try:
            self.logger.info(f"Ending Trackio run: {run_name}")
            self.active_runs[run_name].finish()
            # Allow background thread to complete sending data before program exit
            time.sleep(0.5)
            if run_name in self.active_runs:
                del self.active_runs[run_name]
        except Exception as e:
            self.logger.error(f"Error ending Trackio run {run_name}: {e}")

    def delete_run(self, run_name: str) -> None:
        """Delete a specific run."""
        try:
            runs = self.api.runs(self.experiment_name)
            for run in runs:
                if run.name == run_name:
                    run.delete()
                    break
            else:
                self.logger.warning(f"Trackio run '{run_name}' not found")
            if run_name in self.active_runs:
                del self.active_runs[run_name]
        except Exception as e:
            raise ValueError(f"Trackio run '{run_name}' not found: {e}")

    def clear_context(self) -> None:
        """Clear the Trackio context by ending all active runs."""
        try:
            active_run_keys = list(self.active_runs.keys())
            for run_name in active_run_keys:
                self.logger.info(f"Clearing Trackio context calling trackio.finish() for {run_name=}")
                self.end_run(run_name)
        
            self.logger.info("Trackio context cleared successfully")
        except Exception:
            self.logger.info("No active Trackio run to clear")

