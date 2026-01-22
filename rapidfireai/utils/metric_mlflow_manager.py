"""This module contains the MLflowManager class which is responsible for managing the MLflow runs."""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Any
from rapidfireai.utils.metric_logger import MetricLogger, MetricLoggerType
from rapidfireai.utils.ping import ping_server
from rapidfireai.utils.constants import MLFlowConfig
from rapidfireai.evals.utils.logger import RFLogger


class MLflowMetricLogger(MetricLogger):
    def __init__(self, tracking_uri: str, logger: RFLogger = None, init_kwargs: dict[str, Any] = None):
        """
        Initialize MLflow Manager with tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI
            init_kwargs: Initialization kwargs for MLflow
        """
        self.type = MetricLoggerType.MLFLOW
        self.client = None
        self.logger = logger if logger is not None else RFLogger()
        self.init_kwargs = init_kwargs # Not currently used
        if not ping_server(MLFlowConfig.HOST, MLFlowConfig.PORT, 2):
            raise ConnectionRefusedError(f"MLflow server not available at {MLFlowConfig.URL}. MLflow logging will be disabled")
        else:
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = None

    def create_experiment(self, experiment_name: str) -> str:
        """Create a new experiment and set it as active."""
        self.experiment_id = self.client.create_experiment(experiment_name)
        # IMPORTANT: Set this as the active experiment in MLflow context
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def get_experiment(self, experiment_name: str) -> str:
        """Get existing experiment by name and set it as active."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        self.experiment_id = experiment.experiment_id
        return self.experiment_id

    def create_run(self, run_name: str) -> str:
        """Create a new run and return metric_run_id."""
        if self.experiment_id is None:
            raise ValueError("No experiment set. Call create_experiment() or get_experiment() first.")
        run = self.client.create_run(self.experiment_id, run_name=run_name)
        return run.info.run_id

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Log parameters to a specific run."""
        self.client.log_param(run_id, key, value)

    def log_metric(self, run_id: str, key: str, value: float, step: int = None) -> None:
        """Log a metric to a specific run."""
        self.client.log_metric(run_id, key, value, step=step)

    def get_run_metrics(self, run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a specific run.
        """
        try:
            run = self.client.get_run(run_id)
            if run is None:
                return {}

            run_data = run.data
            metric_dict = {}
            for metric_key in run_data.metrics.keys():
                try:
                    metric_history = self.client.get_metric_history(run_id, metric_key)
                    metric_dict[metric_key] = [(metric.step, metric.value) for metric in metric_history]
                except Exception as e:
                    self.logger.error(f"Error getting metric history for {metric_key}: {e}")
                    continue
            return metric_dict
        except Exception as e:
            self.logger.error(f"Error getting metrics for run {run_id}: {e}")
            return {}

    def end_run(self, run_id: str) -> None:
        """End a specific run."""
        # Check if run exists before terminating
        run = self.client.get_run(run_id)
        if run is not None:
            # First terminate the run on the server
            self.client.set_terminated(run_id)

            # Then clear the local MLflow context if this is the active run
            try:
                current_run = mlflow.active_run()
                # Make sure we end the run on the correct worker
                if current_run and current_run.info.run_id == run_id:
                    mlflow.end_run()
                else:
                    self.logger.warning(f"Run {run_id} is not the active run, no local context to clear")
            except Exception as e:
                self.logger.error(f"Error clearing local MLflow context: {e}")
        else:
            self.logger.warning(f"MLflow run {run_id} not found, cannot terminate")

    def delete_run(self, run_id: str) -> None:
        """Delete a specific run."""
        # Check if run exists before deleting
        run = self.client.get_run(run_id)
        if run is not None:
            self.client.delete_run(run_id)
        else:
            raise ValueError(f"Run '{run_id}' not found")

    def clear_context(self) -> None:
        """Clear the MLflow context by ending any active run."""
        try:
            current_run = mlflow.active_run()
            if current_run:
                run_id = current_run.info.run_id

                # Try to end the run properly using the client first
                try:
                    self.client.end_run(run_id)
                except Exception:
                    # Fallback to global mlflow.end_run()
                    mlflow.end_run()
                    self.logger.info(f"Run {run_id} ended using global mlflow.end_run")

                self.logger.info("MLflow context cleared successfully")
            else:
                self.logger.info("No active MLflow run to clear")
        except Exception as e:
            self.logger.error(f"Error clearing MLflow context: {e}")
