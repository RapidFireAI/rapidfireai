"""This module contains utility functions for the experiment."""

import multiprocessing as mp
import os
import re
import signal
import sys
import warnings
from typing import Any

import pandas as pd
import torch
from IPython.display import display
from tqdm import tqdm
from transformers import logging as transformers_logging

from rapidfireai.utils.constants import MLFlowConfig, RF_MLFLOW_ENABLED
from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger
from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.fit.utils.constants import ExperimentStatus, ExperimentTask
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.exceptions import DBException, ExperimentException
from rapidfireai.fit.utils.logging import RFLogger

# Note: mlflow and MLflowManager are imported lazily inside conditional blocks
# to avoid MLflow connection attempts when using tensorboard-only mode


class ExperimentUtils:
    """Class to contain utility functions for the experiment."""

    def __init__(self) -> None:
        # initialize database handler
        self.db = RfDb()

    def _disable_ml_warnings_display(self) -> None:
        """Disable notebook display"""
        tqdm.disable = True

        # Suppress the transformers logging
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"
        transformers_logging.set_verbosity_error()

        # Suppress the torch warnings
        torch.set_warn_always(False)

        # Suppress the FutureWarning
        warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
        warnings.filterwarnings("ignore", message=".*torch.amp.autocast.*")
        warnings.filterwarnings("ignore", message=".*fan_in_fan_out is set to False.*")
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
        warnings.filterwarnings("ignore", message=".*decoder-only architecture.*")
        warnings.filterwarnings("ignore", message=".*attention mask is not set.*")

    def setup_signal_handlers(
        self,
        worker_processes: list[mp.Process],
    ) -> None:
        """Setup signal handlers for graceful shutdown on the main process."""

        def signal_handler(signum, frame):
            """Handle SIGINT and SIGTERM signals"""
            signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
            print(f"\nReceived {signal_name}, shutting down gracefully...")

            try:
                # Cancel current task if any
                self.cancel_current()

                self.shutdown_workers(worker_processes)

                print("Graceful shutdown completed.")
                sys.exit(0)
            except Exception as e:
                print(f"Error during graceful shutdown: {e}")
                sys.exit(1)

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def create_experiment(self, given_name: str, experiments_path: str) -> tuple[int, str, list[str]]:
        """Create a new experiment. Returns the experiment id, name, and log messages."""
        log_messages: list[str] = []

        # disable warnings from notebook
        self._disable_ml_warnings_display()

        # Clear any existing MLflow context before starting new experiment
        # Only if using MLflow backend
        if RF_MLFLOW_ENABLED=="true":
            import mlflow  # Lazy import to avoid connection attempts in tensorboard-only mode

            try:
                if mlflow.active_run():
                    print("Clearing existing MLflow context before starting new experiment")
                    mlflow.end_run()
            except Exception as e:
                print(f"Error clearing existing MLflow context: {e}")

        # check if experiment is already running
        running_experiment = None
        try:
            running_experiment = self.db.get_running_experiment()
        except DBException:
            pass
        if running_experiment:
            # check if the running experiment is the same as the new experiment
            if given_name == running_experiment["experiment_name"]:
                msg = (
                    f"Experiment {running_experiment['experiment_name']} is currently running."
                    " Returning the same experiment object."
                )
                print(msg)
                log_messages.append(msg)

                # check if the running experiment is also running a task
                current_task = self.db.get_experiment_current_task()
                if current_task != ExperimentTask.IDLE:
                    msg = f"Task {current_task.value} that was running has been cancelled."
                    print(msg)
                    log_messages.append(msg)
                self.cancel_current(internal=True)

                # get experiment id
                experiment_id, experiment_name = (
                    running_experiment["experiment_id"],
                    running_experiment["experiment_name"],
                )
            else:
                self.end_experiment(internal=True)
                experiment_id, experiment_name, metric_experiment_id = self._create_experiment_internal(
                    given_name,
                    experiments_path,
                )
                if metric_experiment_id:
                    msg = (
                        f"The previously running experiment {running_experiment['experiment_name']} was forcibly ended."
                        f" Created a new experiment '{experiment_name}' with Experiment ID: {experiment_id}"
                        f" and Metric Experiment ID: {metric_experiment_id} at {experiments_path}/{experiment_name}"
                    )
                else:
                    msg = (
                        f"The previously running experiment {running_experiment['experiment_name']} was forcibly ended."
                        f" Created a new experiment '{experiment_name}' with Experiment ID: {experiment_id}"
                        f" at {experiments_path}/{experiment_name} (TensorBoard-only mode)"
                    )
                print(msg)
                log_messages.append(msg)
        # check if experiment name already exists
        elif given_name in self.db.get_all_experiment_names():
            experiment_id, experiment_name, metric_experiment_id = self._create_experiment_internal(
                given_name,
                experiments_path,
            )
            if metric_experiment_id:
                msg = (
                    "An experiment with the same name already exists."
                    f" Created a new experiment '{experiment_name}' with Experiment ID: {experiment_id}"
                    f" and Metric Experiment ID: {metric_experiment_id} at {experiments_path}/{experiment_name}"
                )
            else:
                msg = (
                    "An experiment with the same name already exists."
                    f" Created a new experiment '{experiment_name}' with Experiment ID: {experiment_id}"
                    f" at {experiments_path}/{experiment_name} (TensorBoard-only mode)"
                )
            print(msg)
            log_messages.append(msg)
        else:
            experiment_id, experiment_name, metric_experiment_id = self._create_experiment_internal(
                given_name,
                experiments_path,
            )
            if metric_experiment_id:
                msg = (
                    f"Experiment {experiment_name} created with Experiment ID: {experiment_id}"
                    f" and Metric Experiment ID: {metric_experiment_id} at {experiments_path}/{experiment_name}"
                )
            else:
                msg = (
                    f"Experiment {experiment_name} created with Experiment ID: {experiment_id}"
                    f" at {experiments_path}/{experiment_name} (TensorBoard-only mode)"
                )
            print(msg)
            log_messages.append(msg)

        # initialize the data paths and create directories
        try:
            DataPath.initialize(experiment_name, experiments_path)
        except Exception as e:
            raise ExperimentException(f"Failed to initialize data paths: {e}")

        return experiment_id, experiment_name, log_messages

    def end_experiment(self, internal: bool = False) -> None:
        """End the experiment"""
        # check if experiment is running
        try:
            current_experiment = self.db.get_running_experiment()
        except DBException:
            if not internal:
                print("No experiment is currently running. Nothing to end.")
            return

        # create logger
        experiment_name = current_experiment["experiment_name"]
        logger = RFLogger().create_logger(experiment_name)

        # cancel current task if there's any
        self.cancel_current(internal=True)

        # reset DB states
        self.db.set_experiment_status(current_experiment["experiment_id"], ExperimentStatus.COMPLETED)
        self.db.reset_all_tables()

        # Clear MLflow context only if using MLflow backend
        if RF_MLFLOW_ENABLED=="true":
            import mlflow  # Lazy import to avoid connection attempts in tensorboard-only mode

            try:
                if mlflow.active_run():
                    print("Ending active MLflow run before ending experiment")
                    mlflow.end_run()

                # Also clear context through RFMetricLogger if available
                try:
                    metric_logger_config = RFMetricLogger.get_default_metric_loggers(experiment_name=experiment_name)
                    metric_logger = RFMetricLogger(metric_logger_config, logger=logger)
                    metric_logger.clear_context()
                except Exception as e2:
                    print(f"Error clearing Metric context through RFMetricLogger: {e2}")
            except Exception as e:
                print(f"Error clearing Metric context: {e}")

        # print experiment ended message
        msg = f"Experiment {experiment_name} ended"
        if not internal:
            print(msg)
        logger.info(msg)

    def cancel_current(self, internal: bool = False) -> None:
        """Cancel the current task"""
        # check if experiment is running
        try:
            current_experiment = self.db.get_running_experiment()
        except DBException:
            if not internal:
                print("No experiment is currently running. Nothing to cancel.")
            return

        # create logger
        logger = RFLogger().create_logger(current_experiment["experiment_name"])

        try:
            current_task = self.db.get_experiment_current_task()
        except DBException:
            msg = "No task is currently running. Nothing to cancel."
            if not internal:
                print(msg)
            logger.info(msg)
            return

        # reset experiment states and set current task to idle
        self.db.reset_experiment_states()
        self.db.set_experiment_current_task(ExperimentTask.IDLE)
        if current_task != ExperimentTask.IDLE:
            msg = f"Task {current_task.value} cancelled"
            print(msg)
            logger.info(msg)
        logger.debug("Reset experiment states and set current experiment task to idle")

    def shutdown_workers(
        self,
        worker_processes: list[mp.Process],
    ) -> None:
        """Shutdown the workers"""
        # stop workers
        for worker_process in worker_processes:
            worker_process.terminate()
        print("Workers stopped")

    def get_runs_info(self) -> pd.DataFrame:
        """Get the run info"""
        try:
            runs = self.db.get_all_runs()
            runs_info = {}
            for run_id, run_details in runs.items():
                new_run_details = {k: v for k, v in run_details.items() if k not in ("flattened_config", "config_leaf")}
                if "config_leaf" in run_details:
                    config_leaf = run_details["config_leaf"].copy()
                    config_leaf.pop("additional_kwargs", None)
                    new_run_details["config"] = config_leaf

                runs_info[run_id] = new_run_details

            if runs_info:
                df = pd.DataFrame.from_dict(runs_info, orient="index")
                df = df.reset_index().rename(columns={"index": "run_id"})
                cols = ["run_id"] + [col for col in df.columns if col != "run_id"]
                df = df[cols]
                return df
            else:
                return pd.DataFrame(columns=["run_id"])

        except DBException as e:
            raise ExperimentException("Error getting runs info") from e

    def _display_runs_info(self, runs_info: dict[int, dict[str, Any]]) -> pd.DataFrame:
        """Fetch runs info, display as a pandas DataFrame, and return the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing all runs information with run_id as first column.
        """
        try:
            # Convert the runs info to a pandas DataFrame
            df = pd.DataFrame.from_dict(runs_info, orient="index")

            # Reset index to make run_id a regular column for better display
            df = df.reset_index().rename(columns={"index": "run_id"})

            # Reorder columns to put run_id first
            cols = ["run_id"] + [col for col in df.columns if col != "run_id"]
            df = df[cols]

            # Set pandas display options for better notebook viewing
            pd.set_option("display.max_columns", None)
            pd.set_option("display.max_rows", None)
            pd.set_option("display.width", None)
            pd.set_option("display.max_colwidth", 50)

            # Display the results
            print(f"Total runs: {len(df)}")
            print("\n" + "=" * 80)

            try:
                display(df)  # For notebook environments
            except NameError:
                print(df.to_string())  # Fallback for non-notebook environments

            return df

        except ExperimentException as e:
            print(f"Error displaying runs info: {e}")
            raise

    def _create_experiment_internal(self, given_name: str, experiments_path: str) -> tuple[int, str, str | None]:
        """Create new experiment -
        if given_name already exists - increment suffix and create new experiment
        if given_name is new - create new experiment with given name
        Returns: experiment_id, experiment_name, metric_experiment_id (or None)
        """
        try:
            given_name = given_name if given_name else "rf-exp"
            experiment_name = self._generate_unique_experiment_name(given_name, self.db.get_all_experiment_names())

            # Create Metricexperiment only if available
            metric_experiment_id = None
            if RF_MLFLOW_ENABLED=="true":
                import mlflow  # Lazy import to avoid connection attempts in tensorboard-only mode

                try:
                    # create logger
                    logger = RFLogger().create_logger(experiment_name)
                    metric_logger_config = RFMetricLogger.get_default_metric_loggers(experiment_name=experiment_name)
                    metric_logger = RFMetricLogger(metric_logger_config, logger=logger)
                    metric_experiment_id = metric_logger.create_experiment(experiment_name)
                    mlflow.tracing.disable_notebook_display()
                except Exception as e:
                    # Catch MLflow-specific exceptions (mlflow.exceptions.RestException, etc.)
                    raise ExperimentException(f"Error creating Metric experiment: {e}") from e

            # write new experiment details to database
            experiment_id = self.db.create_experiment(
                experiment_name,
                metric_experiment_id,  # Will be None for tensorboard-only
                config_options={"experiments_path": experiments_path},
            )
            return experiment_id, experiment_name, metric_experiment_id
        except ExperimentException:
            # Re-raise ExperimentExceptions (including MLflow errors from above)
            raise
        except Exception as e:
            # Catch any other unexpected errors
            raise ExperimentException(f"Error in _create_experiment_internal: {e}") from e

    def _generate_unique_experiment_name(self, name: str, existing_names: list[str]) -> str:
        """Increment the suffix of the name after the last '_' till it is unique"""
        if not name:
            name = "rf-exp"

        pattern = r"^(.+?)(_(\d+))?$"
        max_attempts = 1000  # Prevent infinite loops
        attempts = 0

        new_name = name
        while new_name in existing_names and attempts < max_attempts:
            match = re.match(pattern, new_name)

            if match:
                base_name = match.group(1)
                current_suffix = match.group(3)
                if current_suffix:
                    try:
                        new_suffix = int(current_suffix) + 1
                    except ValueError:
                        # If suffix is not a valid integer, start from 1
                        new_suffix = 1
                else:
                    new_suffix = 1
                new_name = f"{base_name}_{new_suffix}"
            else:
                new_name = f"{new_name}_1"

            attempts += 1

        if attempts >= max_attempts:
            raise ExperimentException("Could not generate unique experiment name")

        return new_name
