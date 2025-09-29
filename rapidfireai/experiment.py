"""
This module contains the Experiment class which manages the entire experiment lifecycle.
"""

import multiprocessing as mp
import os
import traceback
from collections.abc import Callable
from typing import Any

import pandas as pd
from torch.utils.data import Dataset

from rapidfireai.backend.controller import Controller
from rapidfireai.db.rf_db import RfDb
from rapidfireai.utils.constants import MLFLOW_URL
from rapidfireai.utils.exceptions import ExperimentException
from rapidfireai.utils.experiment_utils import ExperimentUtils
from rapidfireai.utils.logging import RFLogger
from rapidfireai.utils.mlflow_manager import MLflowManager
from rapidfireai.version import __version__


class Experiment:
    """Class to manage the entire experiment lifecycle."""

    def __init__(
        self,
        experiment_name: str,
        experiments_path: str = os.getenv("RF_EXPERIMENT_PATH", "./rapidfire_experiments"),
    ) -> None:
        """
        Args:
            experiment_name: The name of the experiment.
            experiments_path: The base path to the experiments directory.
        """
        # initialize experiment variables
        self.experiment_name: str = experiment_name
        self.experiment_id: int | None = None
        self.log_server_process: mp.Process | None = None
        self.worker_processes: list[mp.Process] = []

        # create db tables
        try:
            RfDb().create_tables()
        except Exception as e:
            raise ExperimentException(f"Error creating db tables: {e}, traceback: {traceback.format_exc()}") from e

        # create experiment utils object
        self.experiment_utils = ExperimentUtils()

        # create experiment
        try:
            self.experiment_id, self.experiment_name, log_messages = self.experiment_utils.create_experiment(
                given_name=self.experiment_name,
                experiments_path=os.path.abspath(experiments_path),
            )
        except Exception as e:
            raise ExperimentException(f"Error creating experiment: {e}, traceback: {traceback.format_exc()}") from e

        # create logger
        try:
            self.logger = RFLogger().create_logger("experiment")
            for msg in log_messages:
                self.logger.info(msg)
            # Log the version of rapidfireai that is running
            self.logger.info(f"Running RapidFire AI version {__version__}")
        except Exception as e:
            raise ExperimentException(f"Error creating logger: {e}, traceback: {traceback.format_exc()}") from e

        # setup signal handlers for graceful shutdown
        try:
            self.experiment_utils.setup_signal_handlers(self.worker_processes)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error setting up signal handlers: {e}")
            raise ExperimentException(
                f"Error setting up signal handlers: {e}, traceback: {traceback.format_exc()}"
            ) from e

    def run_fit(
        self,
        param_config: Any,
        create_model_fn: Callable,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_chunks: int,
        seed: int = 42,
    ) -> None:
        """Run the fit"""
        try:
            controller = Controller(self.experiment_id, self.experiment_name)
            controller.run_fit(param_config, create_model_fn, train_dataset, eval_dataset, num_chunks, seed)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error running fit: {e}")
            raise ExperimentException(f"Error running fit: {e}, traceback: {traceback.format_exc()}") from e

    def get_results(self) -> pd.DataFrame:
        """
        Get the MLflow training metrics for all runs in the experiment.
        """
        try:
            runs_info_df = self.experiment_utils.get_runs_info()
            mlflow_manager = MLflowManager(MLFLOW_URL)

            metrics_data = []

            for _, run_row in runs_info_df.iterrows():
                run_id = run_row["run_id"]
                mlflow_run_id = run_row.get("mlflow_run_id")

                if not mlflow_run_id:
                    continue

                run_metrics = mlflow_manager.get_run_metrics(mlflow_run_id)

                step_metrics = {}
                for metric_name, metric_values in run_metrics.items():
                    for step, value in metric_values:
                        if step not in step_metrics:
                            step_metrics[step] = {"run_id": run_id, "step": step}
                        step_metrics[step][metric_name] = value

                metrics_data.extend(step_metrics.values())

            if metrics_data:
                return pd.DataFrame(metrics_data).sort_values(["run_id", "step"])
            else:
                return pd.DataFrame(columns=["run_id", "step"])

        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error getting results: {e}")
            raise ExperimentException(f"Error getting results: {e}, traceback: {traceback.format_exc()}") from e

    def cancel_current(self) -> None:
        """Cancel the current task"""
        try:
            self.experiment_utils.cancel_current(internal=False)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error canceling current task: {e}")
            raise ExperimentException(f"Error canceling current task: {e}, traceback: {traceback.format_exc()}") from e

    def get_runs_info(self) -> pd.DataFrame:
        """Get the run info"""
        try:
            return self.experiment_utils.get_runs_info()
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error getting run info: {e}")
            raise ExperimentException(f"Error getting run info: {e}, traceback: {traceback.format_exc()}") from e

    def end(self) -> None:
        """End the experiment"""
        try:
            self.experiment_utils.end_experiment(internal=False)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error ending experiment: {e}")
            raise ExperimentException(f"Error ending experiment: {e}, traceback: {traceback.format_exc()}") from e

        # shutdown all child processes
        try:
            self.experiment_utils.shutdown_workers(self.worker_processes)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error shutting down RapidFire processes: {e}")
            raise ExperimentException(
                f"Error shutting down RapidFire processes: {e}, traceback: {traceback.format_exc()}"
            ) from e
