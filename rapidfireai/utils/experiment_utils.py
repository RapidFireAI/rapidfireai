"""
RapidFire AI Experiment Utilities

Provides experiment creation and management.
"""

import os
import re
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

from rapidfireai.db.rf_db import RfDb
from rapidfireai.utils.constants import (
    ExperimentStatus,
    ExperimentTask,
    RF_EXPERIMENT_PATH,
    RF_MLFLOW_ENABLED,
)
from rapidfireai.utils.exceptions import DBException, ExperimentException
from rapidfireai.utils.logging import RFLogger
from rapidfireai.utils.os_utils import mkdir_p


class ExperimentUtils:
    """
    Experiment utilities for creation, naming, cancellation, and cleanup.
    """

    def __init__(self):
        """Initialize with database handler."""
        self.db = RfDb()

    def _disable_ml_warnings_display(self) -> None:
        """Disable ML-related warnings and verbose output."""
        try:
            from tqdm import tqdm
            tqdm.disable = True
        except ImportError:
            pass

        try:
            import torch
            torch.set_warn_always(False)
        except ImportError:
            pass

        try:
            from transformers import logging as transformers_logging
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"
            transformers_logging.set_verbosity_error()
        except ImportError:
            pass

        # Suppress common warnings
        warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
        warnings.filterwarnings("ignore", message=".*torch.amp.autocast.*")
        warnings.filterwarnings("ignore", message=".*fan_in_fan_out is set to False.*")
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
        warnings.filterwarnings("ignore", message=".*decoder-only architecture.*")
        warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
        warnings.filterwarnings("ignore", message=".*Unable to register cuDNN factory.*")
        warnings.filterwarnings("ignore", message=".*Unable to register cuBLAS factory.*")
        warnings.filterwarnings("ignore", message=".*All log messages before absl::InitializeLog.*")
        warnings.filterwarnings("ignore", message=".*resource_tracker: process died unexpectedly.*")
        warnings.filterwarnings("ignore", message=".*computation placer already registered.*")
        warnings.filterwarnings("ignore", message=".*Rank 0 is connected to 0 peer ranks.*")
        warnings.filterwarnings("ignore", message=".*No cudagraph will be used.*")
        warnings.filterwarnings("ignore", module="multiprocessing.resource_tracker")

    def create_experiment(
        self,
        given_name: str,
        experiments_path: str,
    ) -> tuple[int, str, list[str]]:
        """
        Create a new experiment. Returns the experiment id, name, and log messages.
        
        This method handles:
        - Checking for already running experiments
        - Generating unique experiment names
        - Creating experiment record in database
        - Creating experiment directories

        Args:
            given_name: Desired experiment name
            experiments_path: Path to experiments directory

        Returns:
            Tuple of (experiment_id, experiment_name, log_messages)
        """
        log_messages: list[str] = []

        # Disable warnings
        self._disable_ml_warnings_display()

        # Clear any existing MLflow context before starting new experiment
        if RF_MLFLOW_ENABLED == "true":
            try:
                import mlflow
                if mlflow.active_run():
                    print("Clearing existing MLflow context before starting new experiment")
                    mlflow.end_run()
            except Exception as e:
                print(f"Error clearing existing MLflow context: {e}")

        # Check if experiment is already running
        running_experiment = None
        try:
            running_experiment = self.db.get_running_experiment()
        except Exception:
            pass

        if running_experiment:
            if given_name == running_experiment["experiment_name"]:
                # Same experiment - reuse it
                msg = (
                    f"Experiment {running_experiment['experiment_name']} is currently running."
                    " Returning the same experiment object."
                )
                print(msg)
                log_messages.append(msg)

                # Check if there's a running task and cancel it
                try:
                    current_task = self.db.get_experiment_current_task()
                    if current_task != ExperimentTask.IDLE:
                        msg = f"Task {current_task.value} that was running has been cancelled."
                        print(msg)
                        log_messages.append(msg)
                except Exception:
                    pass

                self.cancel_current(internal=True)

                experiment_id = running_experiment["experiment_id"]
                experiment_name = running_experiment["experiment_name"]
            else:
                # Different experiment - end the previous one
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
                        f" at {experiments_path}/{experiment_name}"
                    )
                print(msg)
                log_messages.append(msg)
        elif given_name in self.db.get_all_experiment_names():
            # Name exists - create with incremented suffix
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
                    f" at {experiments_path}/{experiment_name}"
                )
            print(msg)
            log_messages.append(msg)
        else:
            # New experiment
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
                    f" at {experiments_path}/{experiment_name}"
                )
            print(msg)
            log_messages.append(msg)

        # Create experiment directory
        try:
            experiment_dir = Path(experiments_path) / experiment_name
            mkdir_p(experiment_dir)
        except Exception as e:
            raise ExperimentException(f"Failed to create experiment directories: {e}") from e

        return experiment_id, experiment_name, log_messages

    def end_experiment(self, internal: bool = False) -> None:
        """
        End the experiment and clean up resources.

        Args:
            internal: If True, suppress output messages
        """
        try:
            current_experiment = self.db.get_running_experiment()
        except Exception:
            if not internal:
                print("No experiment is currently running. Nothing to end.")
            return

        experiment_name = current_experiment["experiment_name"]
        experiments_path = current_experiment.get("experiments_path", RF_EXPERIMENT_PATH)

        # Create logger
        logger = RFLogger(
            experiment_name=experiment_name,
            experiment_path=experiments_path,
        ).get_logger("ExperimentUtils")

        # Cancel current task if any
        self.cancel_current(internal=True)

        # Reset DB states
        self.db.set_experiment_status(current_experiment["experiment_id"], ExperimentStatus.COMPLETED)
        self.db.reset_all_tables()

        # Clear MLflow context if enabled
        if RF_MLFLOW_ENABLED == "true":
            try:
                import mlflow
                if mlflow.active_run():
                    print("Ending active MLflow run before ending experiment")
                    mlflow.end_run()
            except Exception as e:
                print(f"Error clearing MLflow context: {e}")

        msg = f"Experiment {experiment_name} ended"
        if not internal:
            print(msg)
        logger.info(msg)

    def cancel_current(self, internal: bool = False) -> None:
        """
        Cancel the current task.

        Args:
            internal: If True, suppress output messages
        """
        try:
            current_experiment = self.db.get_running_experiment()
        except Exception:
            if not internal:
                print("No experiment is currently running. Nothing to cancel.")
            return

        experiment_name = current_experiment["experiment_name"]
        experiments_path = current_experiment.get("experiments_path", RF_EXPERIMENT_PATH)

        # Create logger
        logger = RFLogger(
            experiment_name=experiment_name,
            experiment_path=experiments_path,
        ).get_logger("ExperimentUtils")

        try:
            current_task = self.db.get_experiment_current_task()
        except Exception:
            if not internal:
                print("No task is currently running. Nothing to cancel.")
            return

        # Reset experiment states and set current task to idle
        self.db.reset_experiment_states()
        self.db.set_experiment_current_task(ExperimentTask.IDLE)

        if current_task != ExperimentTask.IDLE:
            msg = f"Task {current_task.value} cancelled"
            print(msg)
            logger.info(msg)

        logger.debug("Reset experiment states and set current experiment task to idle")

    def get_runs_info(self) -> pd.DataFrame:
        """
        Get run info for all runs in the experiment (fit mode).

        Returns:
            DataFrame with run information
        """
        try:
            runs = self.db.get_all_runs()
            runs_info = {}

            for run_id, run_details in runs.items():
                new_run_details = {
                    k: v for k, v in run_details.items()
                    if k not in ("flattened_config", "config_leaf")
                }
                if "config_leaf" in run_details:
                    config_leaf = run_details["config_leaf"].copy() if run_details["config_leaf"] else {}
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

        except Exception as e:
            raise ExperimentException(f"Error getting runs info: {e}") from e

    def _create_experiment_internal(
        self,
        given_name: str,
        experiments_path: str,
    ) -> tuple[int, str, str | None]:
        """
        Create new experiment - generate unique name and write to database.

        Args:
            given_name: Desired experiment name
            experiments_path: Path to experiments directory

        Returns:
            Tuple of (experiment_id, experiment_name, metric_experiment_id or None)
        """
        try:
            given_name = given_name if given_name else "rf-exp"
            experiment_name = self._generate_unique_experiment_name(
                given_name,
                self.db.get_all_experiment_names(),
            )

            # Clear all non-experiment tables before creating new experiment
            self.db.reset_all_tables(experiments_table=False)

            # Create MLflow experiment if enabled
            metric_experiment_id = None
            if RF_MLFLOW_ENABLED == "true":
                try:
                    import mlflow

                    from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger

                    logger = RFLogger(
                        experiment_name=experiment_name,
                        experiment_path=experiments_path,
                    ).get_logger("ExperimentUtils")

                    metric_logger_config = RFMetricLogger.get_default_metric_loggers(
                        experiment_name=experiment_name
                    )
                    metric_logger = RFMetricLogger(metric_logger_config, logger=logger)
                    metric_experiment_id = metric_logger.create_experiment(experiment_name)
                    mlflow.tracing.disable_notebook_display()
                except Exception as e:
                    raise ExperimentException(f"Error creating Metric experiment: {e}") from e

            # Write new experiment to database
            experiment_id = self.db.create_experiment(
                experiment_name=experiment_name,
                experiments_path=os.path.abspath(experiments_path),
                metric_experiment_id=metric_experiment_id,
                status=ExperimentStatus.RUNNING,
            )

            return experiment_id, experiment_name, metric_experiment_id

        except ExperimentException:
            raise
        except Exception as e:
            raise ExperimentException(f"Error in _create_experiment_internal: {e}") from e

    def _generate_unique_experiment_name(self, name: str, existing_names: list[str]) -> str:
        """
        Generate a unique experiment name by incrementing suffix if needed.

        Args:
            name: Desired base name
            existing_names: List of existing experiment names

        Returns:
            Unique experiment name
        """
        if not name:
            name = "rf-exp"

        pattern = r"^(.+?)(_(\d+))?$"
        max_attempts = 1000
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


__all__ = ["ExperimentUtils"]
