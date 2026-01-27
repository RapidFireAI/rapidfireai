"""This module contains the TrackioManager class which is responsible for managing the Trackio runs."""

import time
import trackio
import io
from contextlib import redirect_stdout
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
        self.active_runs = {}  # Map run_id -> runs
        self.run_params = {}  # Map run_id -> dict of params to log on init

    def _capture_trackio_output(self, func, *args, **kwargs):
        """Execute a trackio function while capturing and logging its stdout output."""
        captured_output = io.StringIO()
        with redirect_stdout(captured_output):
            result = func(*args, **kwargs)
        output = captured_output.getvalue().strip()
        if output:
            for line in output.split('\n'):
                self.logger.info(f"[trackio] {line}")
                if "Running on public URL" in line or "trackio show --project" in line:
                    print(f"Trackio: {line}")
        return result

    def _ensure_initialized(self, run_name: str) -> bool:
        """Check if a run is initialized."""
        if run_name in self.active_runs:
            return run_name
        self.logger.info(f"Could not find run {run_name} initializing...")
        return self.create_run(run_name)

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
        # Capture stdout to redirect trackio's print statements to the logger
        try:
            self.active_runs[run_name] = self._capture_trackio_output(
                trackio.init, project=self.experiment_name, name=run_name, resume="allow", **self.init_kwargs
            )
            time.sleep(1)
            self.logger.debug(f"Trackio run {self.active_runs[run_name].name} created successfully")
        except Exception as exc:
            raise ValueError(
                f"Exception in calling trackio.init() to create new run: {run_name} "
                f"with self.init_kwargs={self.init_kwargs!r}: {exc}"
            ) from exc

        # Log any pending params for this run 
        if run_name in self.run_params:
            self.active_runs[run_name].log(self.run_params[run_name])
            del self.run_params[run_name]

        return run_name

    def log_param(self, run_id: str, key: str, value: str) -> None:
        try:
            self._ensure_initialized(run_id)
            self.active_runs[run_id].config[key] = value
        except Exception as _:
            # Run not active, store for later when run is created
            if run_id not in self.run_params:
                self.run_params[run_id] = {}
            self.run_params[run_id][key] = value

    def log_metric(self, run_id: str, key: str, value: float, step: int = None) -> None:
        """Log a metric to a specific run."""

        step = step if step is not None else 0
        try:
            step = int(step)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"step must be an integer, got {step!r} (type: {type(step).__name__})"
            ) from exc
        
        log_dict = {key: value}
        try: 
            self._ensure_initialized(run_id)
            self.active_runs[run_id].log(log_dict, step=step)
        except Exception as exc:
            raise ValueError(
                f"Error logging metric in log_metric, is there not an active run?: "
                f"run_id={run_id!r}, {key} = {value}, step={step!r}: {exc}"
            ) from exc

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

    def end_run(self, run_id: str) -> None:
        """End a specific run."""
        try:
            self.logger.info(f"Ending Trackio run: {run_id}")
            self._capture_trackio_output(self.active_runs[run_id].finish)
            # Allow background thread to complete sending data before program exit
            time.sleep(0.5)
            if run_id in self.active_runs:
                del self.active_runs[run_id]
        except Exception as exc:
            self.logger.error(f"Error ending Trackio run {run_id}: {exc}")

    def delete_run(self, run_id: str) -> None:
        """Delete a specific run."""
        try:
            runs = self.api.runs(self.experiment_name)
            for run in runs:
                if run.name == run_id:
                    run.delete()
                    break
            else:
                self.logger.warning(f"Trackio run '{run_id}' not found")
            if run_id in self.active_runs:
                del self.active_runs[run_id]
        except Exception as exc:
            raise ValueError(f"Trackio run '{run_id}' not found: {exc}") from exc

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

