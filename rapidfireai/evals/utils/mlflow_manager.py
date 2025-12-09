"""
MLflow Manager for RapidFire Evals.

This module provides MLflow integration for evals experiments, enabling:
- Experiment tracking and organization
- Pipeline metrics logging (accuracy, latency, throughput, etc.)
- Parameter logging for pipeline configurations
- Results persistence and comparison
"""

import mlflow
from mlflow.tracking import MlflowClient
from typing import Any


class MLflowManager:
    """
    Manager class for MLflow operations in evals mode.

    Handles experiment creation, pipeline run tracking, and metrics logging
    for evaluation pipelines.
    """

    def __init__(self, tracking_uri: str):
        """
        Initialize MLflow Manager with tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI (e.g., "http://127.0.0.1:8852")
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = None
        mlflow.set_tracking_uri(tracking_uri)

    def create_experiment(self, experiment_name: str) -> str:
        """
        Create a new experiment and set it as active.

        Args:
            experiment_name: Name for the new experiment

        Returns:
            The experiment ID
        """
        self.experiment_id = self.client.create_experiment(experiment_name)
        # Set this as the active experiment in MLflow context
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def get_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment by name and set it as active.

        Args:
            experiment_name: Name of the experiment to retrieve

        Returns:
            The experiment ID

        Raises:
            ValueError: If experiment not found
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        self.experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def get_or_create_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment or create new one if it doesn't exist.

        Args:
            experiment_name: Name of the experiment

        Returns:
            The experiment ID
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return self.create_experiment(experiment_name)
        self.experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def create_run(self, run_name: str, tags: dict[str, str] = None) -> str:
        """
        Create a new MLflow run for a pipeline.

        Args:
            run_name: Name for the run (typically pipeline name)
            tags: Optional tags to add to the run

        Returns:
            The MLflow run ID

        Raises:
            ValueError: If no experiment is set
        """
        if self.experiment_id is None:
            raise ValueError("No experiment set. Call create_experiment() or get_experiment() first.")

        run = self.client.create_run(
            self.experiment_id,
            run_name=run_name,
            tags=tags or {}
        )
        return run.info.run_id

    def log_param(self, mlflow_run_id: str, key: str, value: Any) -> None:
        """
        Log a parameter to a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            key: Parameter name
            value: Parameter value (will be converted to string)
        """
        self.client.log_param(mlflow_run_id, key, str(value))

    def log_params(self, mlflow_run_id: str, params: dict[str, Any]) -> None:
        """
        Log multiple parameters to a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            params: Dictionary of parameter names to values
        """
        for key, value in params.items():
            self.log_param(mlflow_run_id, key, value)

    def log_metric(self, mlflow_run_id: str, key: str, value: float, step: int = None) -> None:
        """
        Log a metric to a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            key: Metric name
            value: Metric value
            step: Optional step number for time-series metrics
        """
        self.client.log_metric(mlflow_run_id, key, value, step=step)

    def log_metrics(self, mlflow_run_id: str, metrics: dict[str, float], step: int = None) -> None:
        """
        Log multiple metrics to a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            metrics: Dictionary of metric names to values
            step: Optional step number for time-series metrics
        """
        for key, value in metrics.items():
            if value is not None:
                self.log_metric(mlflow_run_id, key, value, step=step)

    def get_run_metrics(self, mlflow_run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a specific run.

        Args:
            mlflow_run_id: The MLflow run ID

        Returns:
            Dictionary mapping metric names to list of (step, value) tuples
        """
        try:
            run = self.client.get_run(mlflow_run_id)
            if run is None:
                return {}

            run_data = run.data
            metric_dict = {}
            for metric_key in run_data.metrics.keys():
                try:
                    metric_history = self.client.get_metric_history(mlflow_run_id, metric_key)
                    metric_dict[metric_key] = [(metric.step, metric.value) for metric in metric_history]
                except Exception as e:
                    print(f"Error getting metric history for {metric_key}: {e}")
                    continue
            return metric_dict
        except Exception as e:
            print(f"Error getting metrics for run {mlflow_run_id}: {e}")
            return {}

    def get_run_params(self, mlflow_run_id: str) -> dict[str, str]:
        """
        Get all parameters for a specific run.

        Args:
            mlflow_run_id: The MLflow run ID

        Returns:
            Dictionary of parameter names to values
        """
        try:
            run = self.client.get_run(mlflow_run_id)
            if run is None:
                return {}
            return dict(run.data.params)
        except Exception as e:
            print(f"Error getting params for run {mlflow_run_id}: {e}")
            return {}

    def end_run(self, mlflow_run_id: str, status: str = "FINISHED") -> None:
        """
        End a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            status: Run status ("FINISHED", "FAILED", "KILLED")
        """
        try:
            run = self.client.get_run(mlflow_run_id)
            if run is not None:
                self.client.set_terminated(mlflow_run_id, status=status)
            else:
                print(f"MLflow run {mlflow_run_id} not found, cannot terminate")
        except Exception as e:
            print(f"Error ending run {mlflow_run_id}: {e}")

    def fail_run(self, mlflow_run_id: str) -> None:
        """Mark a run as failed."""
        self.end_run(mlflow_run_id, status="FAILED")

    def kill_run(self, mlflow_run_id: str) -> None:
        """Mark a run as killed (user-terminated)."""
        self.end_run(mlflow_run_id, status="KILLED")

    def delete_run(self, mlflow_run_id: str) -> None:
        """
        Delete a specific run.

        Args:
            mlflow_run_id: The MLflow run ID

        Raises:
            ValueError: If run not found
        """
        run = self.client.get_run(mlflow_run_id)
        if run is not None:
            self.client.delete_run(mlflow_run_id)
        else:
            raise ValueError(f"Run '{mlflow_run_id}' not found")

    def set_run_tag(self, mlflow_run_id: str, key: str, value: str) -> None:
        """
        Set a tag on a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            key: Tag name
            value: Tag value
        """
        self.client.set_tag(mlflow_run_id, key, value)

    def clear_context(self) -> None:
        """Clear the MLflow context by ending any active run."""
        try:
            current_run = mlflow.active_run()
            if current_run:
                run_id = current_run.info.run_id
                try:
                    self.client.set_terminated(run_id)
                except Exception:
                    mlflow.end_run()
                print("MLflow context cleared successfully")
            else:
                print("No active MLflow run to clear")
        except Exception as e:
            print(f"Error clearing MLflow context: {e}")
