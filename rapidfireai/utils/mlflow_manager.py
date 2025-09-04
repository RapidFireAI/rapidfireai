"""This module contains the MLflowManager class which is responsible for managing the MLflow runs."""

import mlflow
from mlflow.tracking import MlflowClient


class MLflowManager:
    def __init__(self, tracking_uri: str):
        """
        Initialize MLflow Manager with tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI
        """
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
        """Create a new run and return mlflow_run_id."""
        if self.experiment_id is None:
            raise ValueError("No experiment set. Call create_experiment() or get_experiment() first.")
        run = self.client.create_run(self.experiment_id, run_name=run_name)
        return run.info.run_id

    def log_param(self, mlflow_run_id: str, key: str, value: str) -> None:
        """Log parameters to a specific run."""
        self.client.log_param(mlflow_run_id, key, value)

    def log_metric(self, mlflow_run_id: str, key: str, value: float, step: int = None) -> None:
        """Log a metric to a specific run."""
        self.client.log_metric(mlflow_run_id, key, value, step=step)

    def get_run_metrics(self, mlflow_run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a specific run.
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

    def end_run(self, mlflow_run_id: str) -> None:
        """End a specific run."""
        # Check if run exists before terminating
        run = self.client.get_run(mlflow_run_id)
        if run is not None:
            # First terminate the run on the server
            self.client.set_terminated(mlflow_run_id)

            # Then clear the local MLflow context if this is the active run
            try:
                current_run = mlflow.active_run()
                # Make sure we end the run on the correct worker
                if current_run and current_run.info.run_id == mlflow_run_id:
                    mlflow.end_run()
                else:
                    print(f"Run {mlflow_run_id} is not the active run, no local context to clear")
            except Exception as e:
                print(f"Error clearing local MLflow context: {e}")
        else:
            print(f"MLflow run {mlflow_run_id} not found, cannot terminate")

    def delete_run(self, mlflow_run_id: str) -> None:
        """Delete a specific run."""
        # Check if run exists before deleting
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
                print(f"[DEBUG] Active run ID: {run_id}")

                # Try to end the run properly using the client first
                try:
                    self.client.end_run(run_id)
                except Exception:
                    # Fallback to global mlflow.end_run()
                    mlflow.end_run()
                    print(f"Run {run_id} ended using global mlflow.end_run")

                print("MLflow context cleared successfully")
            else:
                print("No active MLflow run to clear")
        except Exception as e:
            print(f"Error clearing MLflow context: {e}")
