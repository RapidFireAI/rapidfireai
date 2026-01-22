"""
This module contains the RFMetricLogger class which is responsible for managing the metric loggers.
"""

from typing import Optional
from pathlib import Path
from rapidfireai.utils.metric_logger import MetricLogger, MetricLoggerConfig, MetricLoggerType
from rapidfireai.utils.metric_mlflow_manager import MLflowMetricLogger
from rapidfireai.utils.metric_tensorboard_manager import TensorBoardMetricLogger
from rapidfireai.utils.metric_trackio_manager import TrackioMetricLogger
from rapidfireai.evals.utils.logger import RFLogger
from rapidfireai.utils.constants import (
    MLFlowConfig,
    RF_MLFLOW_ENABLED,
    RF_TENSORBOARD_ENABLED,
    RF_TRACKIO_ENABLED,
    RF_TENSORBOARD_LOG_DIR
)

class RFMetricLogger(MetricLogger):
    """
    Implementation of MetricLogger that logs to multiple backends.  Currently no
    more than one of each type is supported.

         This allows users to benefit from multiple tracking systems simultaneously:
         - MLflow for experiment comparison and model registry
         - TensorBoard for real-time training visualization (especially useful in Colab)
         - Trackio for local-first experiment tracking
    """

    def __init__(self, metric_loggers: dict[str, MetricLoggerConfig], logger: RFLogger = None):
        """
        Initialize RFMetricLogger.

        Args:
            metric_loggers: Dictionary of metric loggers to use:
            - "name": {"type": MetricLoggerType.MLFLOW, "config": {"tracking_uri": "http://localhost:8852"}}
            - "name": {"type": MetricLoggerType.TENSORBOARD, "config": {"log_dir": "logs/tensorboard"}}
            - "name": {"type": MetricLoggerType.TRACKIO, "config": {"tracking_uri": None}}
        """
        self.type = MetricLoggerType.MULTIPLE
        self.logger = logger if logger is not None else RFLogger() 
        if not isinstance(metric_loggers, dict):
            raise ValueError("metric_loggers must be a dictionary")
        if len(metric_loggers) == 0:
            raise ValueError("metric_loggers must contain at least one metric logger")
        self.metric_loggers = {}
        for metric_logger_name, metric_logger_config in metric_loggers.items():
            if metric_logger_config.get("type") not in MetricLoggerType:
                raise ValueError(f"metric_logger_config for {metric_logger_name} must be a valid MetricLoggerType")
            if metric_logger_config.get("type") == MetricLoggerType.MLFLOW:
                try:
                    self.metric_loggers[metric_logger_name] = MLflowMetricLogger(metric_logger_config["config"]["tracking_uri"], logger=self.logger)
                    self.logger.info(f"Initialized MLflowMetricLogger: {metric_logger_name}")
                except ConnectionRefusedError as e:
                    self.logger.warning(f"Failed to initialize MLflowMetricLogger: {e}. MLflow logging is disabled.")
            elif metric_logger_config.get("type") == MetricLoggerType.TENSORBOARD:
                self.metric_loggers[metric_logger_name] = TensorBoardMetricLogger(metric_logger_config["config"]["log_dir"], logger=self.logger)
                self.logger.info(f"Initialized TensorBoardMetricLogger: {metric_logger_name}")
            elif metric_logger_config.get("type") == MetricLoggerType.TRACKIO:
                self.metric_loggers[metric_logger_name] = TrackioMetricLogger(
                    experiment_name=metric_logger_config["config"]["experiment_name"],
                    logger=self.logger,
                    init_kwargs=metric_logger_config["config"].get("init_kwargs")
                )
                self.logger.info(f"Initialized TrackioMetricLogger: {metric_logger_name}")
            else:
                raise ValueError(f"metric_logger_config for {metric_logger_name} must be a valid MetricLoggerType")
    
    def add_logger(self, metric_logger_name: str, metric_logger_config: MetricLoggerConfig) -> None:
        """Add a metric logger to the dictionary."""
        if metric_logger_config.get("type") not in MetricLoggerType:
            raise ValueError(f"metric_logger_config for {metric_logger_name} must be a valid MetricLoggerType")
        if metric_logger_config.get("type") == MetricLoggerType.MLFLOW:
            self.metric_loggers[metric_logger_name] = MLflowMetricLogger(metric_logger_config["config"]["tracking_uri"])
        elif metric_logger_config.get("type") == MetricLoggerType.TENSORBOARD:
            self.metric_loggers[metric_logger_name] = TensorBoardMetricLogger(metric_logger_config["config"]["log_dir"])
        elif metric_logger_config.get("type") == MetricLoggerType.TRACKIO:
            self.metric_loggers[metric_logger_name] = TrackioMetricLogger(
                experiment_name=metric_logger_config["config"]["experiment_name"],
                init_kwargs=metric_logger_config["config"].get("init_kwargs")
            )
        else:
            raise ValueError(f"metric_logger_config for {metric_logger_name} must be a valid MetricLoggerType")

    def create_experiment(self, experiment_name: str) -> str:
        """Create experiment in MetricLogger."""
        for metric_logger in self.metric_loggers.values():
            if metric_logger.type == MetricLoggerType.MLFLOW:
                self.logger.info(f"Creating MLflow experiment: {experiment_name}")
                return metric_logger.create_experiment(experiment_name)
        return experiment_name
    
    def get_experiment(self, experiment_name: str) -> str:
        """Get experiment from MetricLogger(TensorBoard doesn't have experiments)."""
        for metric_logger in self.metric_loggers.values():
            if metric_logger.type == MetricLoggerType.MLFLOW:
                return metric_logger.get_experiment(experiment_name)
        return experiment_name
    
    def create_run(self, run_name: str) -> str:
        """Create run in MetricLogger.

        When MLflow is enabled, we first create the MLflow run to get its ID,
        then use that ID for all other loggers to ensure consistency.
        """
        # First, create MLflow run to get its ID (if MLflow is enabled)
        mlflow_run_id = None
        for metric_logger in self.metric_loggers.values():
            if metric_logger.type == MetricLoggerType.MLFLOW:
                mlflow_run_id = metric_logger.create_run(run_name)
                self.logger.info(f"Created MLflow run: {mlflow_run_id}")
                break

        # Use MLflow run ID for all other loggers, or fall back to run_name
        canonical_run_id = mlflow_run_id if mlflow_run_id is not None else run_name

        # Create runs in other loggers using the canonical ID
        for metric_logger in self.metric_loggers.values():
            if metric_logger.type != MetricLoggerType.MLFLOW:
                metric_logger.create_run(canonical_run_id)

        return canonical_run_id
    
    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Log parameter to MetricLogger."""
        for metric_logger_name, metric_logger in self.metric_loggers.items():
            if hasattr(metric_logger, "log_param"):
                metric_logger.log_param(run_id, key, value)
            else:
                raise ValueError(f"metric_logger for {metric_logger_name} does not support log_param")
    
    def log_metric(self, run_id: str, key: str, value: float, step: Optional[int] = None) -> None:
        """Log metric to MetricLogger."""
        for metric_logger_name, metric_logger in self.metric_loggers.items():
            if hasattr(metric_logger, "log_metric"):
                metric_logger.log_metric(run_id, key, value, step=step)
            else:
                raise ValueError(f"metric_logger for {metric_logger_name} does not support log_metric")
    
    def get_run_metrics(self, run_id: str) -> dict:
        """Get metrics from MetricLogger."""
        for metric_logger in self.metric_loggers.values():
            if metric_logger.type == MetricLoggerType.MLFLOW:
                return metric_logger.get_run_metrics(run_id)
        return {}

    def end_run(self, run_id: str) -> None:
        """End run in MetricLogger."""
        for metric_logger_name, metric_logger in self.metric_loggers.items():
            self.logger.info(f"Ending run: {run_id} in {metric_logger_name}")
            if hasattr(metric_logger, "end_run"):
                metric_logger.end_run(run_id)
            else:
                raise ValueError(f"metric_logger for {metric_logger_name} does not support end_run")

    def delete_run(self, run_id: str) -> None:
        """Delete run from MetricLogger."""
        for metric_logger_name, metric_logger in self.metric_loggers.items():
            if hasattr(metric_logger, "delete_run"):
                metric_logger.delete_run(run_id)
            else:
                raise ValueError(f"metric_logger for {metric_logger_name} does not support delete_run")
        return None
    
    def clear_context(self) -> None:
        """Clear context in MetricLogger."""
        for metric_logger_name, metric_logger in self.metric_loggers.items():
            if hasattr(metric_logger, "clear_context"):
                metric_logger.clear_context()
            else:
                raise ValueError(f"metric_logger for {metric_logger_name} does not support clear_context")  
        return None
    
    @classmethod
    def get_default_metric_loggers(cls, experiment_name: str) -> dict[str, MetricLoggerConfig]:
        """Get default metric loggers."""
        metric_loggers = {}
        if RF_MLFLOW_ENABLED == "true":
            metric_loggers["rf_mlflow"] = {
                "type": MetricLoggerType.MLFLOW,
                "config": {
                    "tracking_uri": MLFlowConfig.URL,
                },
            }
        if RF_TENSORBOARD_ENABLED == "true":
            metric_loggers["rf_tensorboard"] = {
                "type": MetricLoggerType.TENSORBOARD,
                "config": {
                    "log_dir": Path(RF_TENSORBOARD_LOG_DIR) / experiment_name,
                },
            }
        if RF_TRACKIO_ENABLED == "true":
            metric_loggers["rf_trackio"] = {
                "type": MetricLoggerType.TRACKIO,
                "config": {
                    "experiment_name": experiment_name,
                },
            }
        return metric_loggers
