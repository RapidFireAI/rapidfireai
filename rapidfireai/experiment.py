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

from rapidfireai.fit.backend.controller import Controller
from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.fit.utils.constants import MLFLOW_URL
from rapidfireai.fit.utils.exceptions import ExperimentException
from rapidfireai.fit.utils.experiment_utils import ExperimentUtils
from rapidfireai.fit.utils.logging import RFLogger
from rapidfireai.version import __version__

# Note: MLflowManager is imported lazily in get_results() to avoid
# connection attempts when using tensorboard-only mode


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
        self._training_thread: Any = None  # Track background training thread (Colab only)

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

        # Check if training is already running
        if self._training_thread is not None and self._training_thread.is_alive():
            print("⚠️  Training is already running in background. Please wait for it to complete.")
            return

        # Detect if running in Google Colab
        try:
            import google.colab

            in_colab = True
        except ImportError:
            in_colab = False

        if in_colab:
            # Run Controller in background thread to keep kernel responsive
            import sys
            import threading
            from io import StringIO

            from IPython.display import HTML, display

            def _run_controller_background():
                """Run controller in background thread with output suppression"""
                # Suppress stdout to avoid print statements appearing in wrong cells
                old_stdout = sys.stdout
                sys.stdout = StringIO()

                try:
                    controller = Controller(self.experiment_id, self.experiment_name)
                    controller.run_fit(param_config, create_model_fn, train_dataset, eval_dataset, num_chunks, seed)
                except Exception as e:
                    # Restore stdout for error logging
                    sys.stdout = old_stdout
                    if hasattr(self, "logger"):
                        self.logger.opt(exception=True).error(f"Error in background training: {e}")
                    display(HTML(f'<p style="color: red; font-weight: bold;">❌ Error in background training: {e}</p>'))
                finally:
                    # Restore stdout
                    sys.stdout = old_stdout
                    # Display completion message
                    display(
                        HTML(
                            '<p style="color: blue; font-weight: bold;">🎉 Training completed! Check InteractiveController for final results.</p>'
                        )
                    )
                    self._training_thread = None

            self._training_thread = threading.Thread(target=_run_controller_background, daemon=True)
            self._training_thread.start()

            # Use IPython display for reliable output in Colab
            display(
                HTML(
                    '<div style="padding: 10px; background-color: #d4edda; border: 1px solid #28a745; border-radius: 5px; color: #155724;">'
                    "<b>✓ Training started in background</b><br>"
                    "Use InteractiveController to monitor progress. The notebook kernel will remain responsive while training runs.<br>"
                    "<small>Tip: Interact with InteractiveController periodically to keep Colab active.</small>"
                    "</div>"
                )
            )
        else:
            # Original blocking behavior for non-Colab environments
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

            # Check if there are any mlflow_run_ids before importing MLflow
            has_mlflow_runs = (
                runs_info_df.get("mlflow_run_id") is not None and runs_info_df["mlflow_run_id"].notna().any()
            )

            if not has_mlflow_runs:
                # No MLflow runs to fetch, return empty DataFrame
                return pd.DataFrame(columns=["run_id", "step"])

            # Lazy import - only import when we actually have MLflow runs to fetch
            from rapidfireai.fit.utils.mlflow_manager import MLflowManager

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
