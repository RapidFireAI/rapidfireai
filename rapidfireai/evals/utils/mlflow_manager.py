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

    Wraps the MLflow client API to provide a simplified interface for:
    - Creating and managing experiments
    - Creating runs for each pipeline
    - Logging parameters and metrics
    - Ending runs with appropriate status
    """

    def __init__(self, tracking_uri: str):
        """
        Initialize MLflow Manager with tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI (e.g., http://127.0.0.1:8852)
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
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def get_experiment(self, experiment_name: str) -> str:
        """
        Get existing experiment by name and set it as active.

        Args:
            experiment_name: Name of existing experiment

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
            experiment_name: Name for the experiment

        Returns:
            The experiment ID
        """
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def create_run(self, run_name: str, tags: dict[str, str] = None) -> str:
        """
        Create a new MLflow run for a pipeline.

        Args:
            run_name: Display name for the run (e.g., "pipeline_1")
            tags: Optional tags to attach to the run

        Returns:
            The MLflow run ID

        Raises:
            ValueError: If no experiment has been set
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
        Log a single parameter to a specific run.

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
            params: Dictionary of parameter name -> value
        """
        for key, value in params.items():
            self.client.log_param(mlflow_run_id, key, str(value))

    def log_metric(self, mlflow_run_id: str, key: str, value: float, step: int = None) -> None:
        """
        Log a single metric to a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            key: Metric name
            value: Metric value
            step: Optional step number (e.g., shard number)
        """
        self.client.log_metric(mlflow_run_id, key, value, step=step)

    def log_metrics(self, mlflow_run_id: str, metrics: dict[str, float], step: int = None) -> None:
        """
        Log multiple metrics to a specific run.

        Args:
            mlflow_run_id: The MLflow run ID
            metrics: Dictionary of metric name -> value
            step: Optional step number (e.g., shard number)
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.client.log_metric(mlflow_run_id, key, float(value), step=step)

    def end_run(self, mlflow_run_id: str, status: str = "FINISHED") -> None:
        """
        End an MLflow run with specified status.

        Args:
            mlflow_run_id: The MLflow run ID
            status: Run status - "FINISHED", "FAILED", or "KILLED"
        """
        try:
            run = self.client.get_run(mlflow_run_id)
            if run is not None:
                self.client.set_terminated(mlflow_run_id, status=status)
        except Exception as e:
            print(f"Error ending MLflow run {mlflow_run_id}: {e}")

    def get_run_metrics(self, mlflow_run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a run, grouped by metric name.

        Args:
            mlflow_run_id: The MLflow run ID

        Returns:
            Dictionary mapping metric name to list of (step, value) tuples
        """
        try:
            run = self.client.get_run(mlflow_run_id)
            if run is None:
                return {}

            metrics = {}
            for key in run.data.metrics.keys():
                try:
                    history = self.client.get_metric_history(mlflow_run_id, key)
                    metrics[key] = [(m.step, m.value) for m in history]
                except Exception as e:
                    print(f"Error getting metric history for {key}: {e}")
                    continue
            return metrics
        except Exception as e:
            print(f"Error getting metrics for run {mlflow_run_id}: {e}")
            return {}

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
        except Exception as e:
            print(f"Error clearing MLflow context: {e}")
