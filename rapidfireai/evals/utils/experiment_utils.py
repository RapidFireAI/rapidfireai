"""This module contains utility functions for the evals experiment."""

import os
import re
import warnings
<<<<<<< Updated upstream
from pathlib import Path
=======
from typing import Any

import pandas as pd
>>>>>>> Stashed changes

from rapidfireai.utils.constants import RF_EXPERIMENT_PATH
from rapidfireai.evals.db.rf_db import RFDatabase
from rapidfireai.evals.utils.constants import ExperimentStatus, MLFlowConfig
from rapidfireai.evals.utils.logger import RFLogger
from rapidfireai.evals.utils import MLflowManager


class ExperimentUtils:
    """Class to contain utility functions for the experiment."""

    def __init__(self) -> None:
        # initialize database handler
        self.db = RFDatabase()

    def _disable_ml_warnings_display(self) -> None:
        """Disable warnings"""
        # Suppress warnings
        warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*")
        warnings.filterwarnings("ignore", message=".*generation flags are not valid.*")
        warnings.filterwarnings("ignore", message=".*decoder-only architecture.*")
        warnings.filterwarnings("ignore", message=".*attention mask is not set.*")
        warnings.filterwarnings("ignore", message=".*Unable to register cuDNN factory.*")
        warnings.filterwarnings("ignore", message=".*Unable to register cuBLAS factory.*")
        warnings.filterwarnings("ignore", message=".*All log messages before absl::InitializeLog.*")
        warnings.filterwarnings("ignore", message=".*resource_tracker: process died unexpectedly.*")
        warnings.filterwarnings("ignore", message=".*computation placer already registered.")
        warnings.filterwarnings("ignore", message=".*Rank 0 is connected to 0 peer ranks.*")
        warnings.filterwarnings("ignore", message=".*No cudagraph will be used.*")
        warnings.filterwarnings("ignore", module="multiprocessing.resource_tracker")

    def create_experiment(self, given_name: str, experiments_path: str) -> tuple[int, str, list[str]]:
        """
        Create a new experiment. Returns the experiment id, name, and log messages.

        Args:
            given_name: Desired experiment name
            experiments_path: Path to experiments directory

        Returns:
            Tuple of (experiment_id, experiment_name, log_messages)
        """
        log_messages: list[str] = []

        # disable warnings
        self._disable_ml_warnings_display()

        # check if experiment is already running
        running_experiment = None
        try:
            running_experiment = self.db.get_running_experiment()
        except Exception:
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

                # Cancel any running tasks
                msg = "Any running tasks have been cancelled."
                print(msg)
                log_messages.append(msg)
                self.cancel_current(internal=True)

                # get experiment id
                experiment_id, experiment_name = (
                    running_experiment["experiment_id"],
                    running_experiment["experiment_name"],
                )
            else:
                # Different experiment - end the previous one and create new
                self.end_experiment(internal=True)
                experiment_id, experiment_name = self._create_experiment(given_name, experiments_path)
                msg = (
                    f"The previously running experiment {running_experiment['experiment_name']} was forcibly ended."
                    f" Created a new experiment '{experiment_name}' with Experiment ID: {experiment_id}"
                    f" at {experiments_path}/{experiment_name}"
                )
                print(msg)
                log_messages.append(msg)
        # check if experiment name already exists
        elif given_name in self.db.get_all_experiment_names():
            experiment_id, experiment_name = self._create_experiment(given_name, experiments_path)
            msg = (
                "An experiment with the same name already exists."
                f" Created a new experiment '{experiment_name}' with Experiment ID: {experiment_id}"
                f" at {experiments_path}/{experiment_name}"
            )
            print(msg)
            log_messages.append(msg)
        else:
            # New experiment
            experiment_id, experiment_name = self._create_experiment(given_name, experiments_path)
            msg = (
                f"Experiment {experiment_name} created with Experiment ID: {experiment_id}"
                f" at {experiments_path}/{experiment_name}"
            )
            print(msg)
            log_messages.append(msg)

        # Create experiment directory
        try:
            experiment_dir = Path(experiments_path) / experiment_name
            experiment_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise Exception(f"Failed to create experiment directories: {e}")

        return experiment_id, experiment_name, log_messages

    def end_experiment(self, internal: bool = False) -> None:
        """End the experiment"""
        # check if experiment is running
        try:
            current_experiment = self.db.get_running_experiment()
        except Exception:
            if not internal:
                print("No experiment is currently running. Nothing to end.")
            return

        # Check if there's actually a running experiment
        if current_experiment is None:
            if not internal:
                print("No experiment is currently running. Nothing to end.")
            return

        # create logger
        experiment_name = current_experiment["experiment_name"]
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=RF_EXPERIMENT_PATH)
        logger = logging_manager.get_logger("ExperimentUtils")

        # cancel current tasks if any
        self.cancel_current(internal=True)

        # reset DB states
        self.db.set_experiment_status(current_experiment["experiment_id"], ExperimentStatus.COMPLETED)
        self.db.reset_all_tables()

        # print experiment ended message
        msg = f"Experiment {experiment_name} ended"
        if not internal:
            print(msg)
        logger.info(msg)

    def cancel_current(self, internal: bool = False) -> None:
        """Cancel the current task - marks experiment as cancelled and resets pipeline/context states"""
        # check if experiment is running
        try:
            current_experiment = self.db.get_running_experiment()
        except Exception:
            if not internal:
                print("No experiment is currently running. Nothing to cancel.")
            return

        # Check if there's actually a running experiment
        if current_experiment is None:
            if not internal:
                print("No experiment is currently running. Nothing to cancel.")
            return

        # create logger
        experiment_name = current_experiment["experiment_name"]
        logging_manager = RFLogger(experiment_name=experiment_name, experiment_path=RF_EXPERIMENT_PATH)
        logger = logging_manager.get_logger("ExperimentUtils")

        try:
            # Reset experiment states (mark pipelines/contexts/tasks as failed)
            self.db.reset_experiment_states()
            logger.info("Reset experiment states - marked ongoing pipelines, contexts, and tasks as failed")

            # Mark experiment as cancelled
            self.db.set_experiment_status(current_experiment["experiment_id"], ExperimentStatus.CANCELLED)

            msg = "Experiment marked as cancelled. Ongoing pipelines, contexts, and tasks have been marked as failed."
            if not internal:
                print(msg)
            logger.info(msg)
        except Exception as e:
            msg = f"Error cancelling experiment: {e}"
            if not internal:
                print(msg)
            logger.error(msg)

    def _create_experiment(self, given_name: str, experiments_path: str) -> tuple[int, str]:
        """
        Create new experiment - if given_name already exists, increment suffix and create new experiment.

        Args:
            given_name: Desired experiment name
            experiments_path: Path to experiments directory

        Returns:
            Tuple of (experiment_id, experiment_name)
        """
        try:
            given_name = given_name if given_name else "rf-exp"
            experiment_name = self._generate_unique_experiment_name(given_name, self.db.get_all_experiment_names())

            # Clear all tables except experiments table before creating new experiment
            # This ensures a clean slate for the new experiment
            self.db.reset_all_tables(experiments_table=False)

            # write new experiment details to database
            experiment_id = self.db.create_experiment(
                experiment_name=experiment_name,
                num_actors=0,  # Will be updated in run_evals
                status=ExperimentStatus.RUNNING,
            )
            return experiment_id, experiment_name
        except Exception as e:
            raise Exception(f"Error in _create_experiment: {e}") from e

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
            raise Exception("Could not generate unique experiment name")

        return new_name

    def get_results(self, experiment_name: str = None) -> pd.DataFrame:
        """
        Get experiment results as a pandas DataFrame.

        Fetches all pipelines for the experiment and their metrics from the database
        and MLflow, returning a DataFrame with pipeline information and evaluation metrics.

        Args:
            experiment_name: Optional experiment name. If not provided, uses the
                           currently running experiment.

        Returns:
            pd.DataFrame with columns:
                - pipeline_id: Pipeline identifier
                - status: Pipeline status (completed, failed, etc.)
                - mlflow_run_id: MLflow run ID for this pipeline
                - model_name: Name of the model used
                - shards_completed: Number of shards processed
                - total_samples_processed: Total samples evaluated
                - Additional metric columns from MLflow (accuracy, latency, etc.)

        Raises:
            Exception: If no experiment is found or database error occurs
        """
        try:
            # Get experiment info
            if experiment_name:
                # Find experiment by name (would need a new DB method)
                # For now, just use running experiment
                experiment = self.db.get_running_experiment()
                if experiment and experiment["experiment_name"] != experiment_name:
                    raise Exception(f"Experiment '{experiment_name}' is not the running experiment")
            else:
                experiment = self.db.get_running_experiment()

            if not experiment:
                raise Exception("No running experiment found. Please run an experiment first.")

            # Get all pipelines for this experiment
            pipelines = self.db.get_all_pipelines()

            if not pipelines:
                return pd.DataFrame(columns=["pipeline_id", "status", "mlflow_run_id"])

            # Build results list
            results = []
            mlflow_manager = MLflowManager(tracking_uri=MLFlowConfig.URL)

            for pipeline in pipelines:
                row = {
                    "pipeline_id": pipeline.get("pipeline_id"),
                    "status": pipeline.get("status"),
                    "mlflow_run_id": pipeline.get("mlflow_run_id"),
                    "shards_completed": pipeline.get("shards_completed", 0),
                    "total_samples_processed": pipeline.get("total_samples_processed", 0),
                }

                # Try to get pipeline config JSON for additional info
                try:
                    config_json = self.db.get_pipeline_config_json(pipeline["pipeline_id"])
                    if config_json:
                        # Extract model name from config
                        if "model_config" in config_json and "model" in config_json["model_config"]:
                            row["model_name"] = config_json["model_config"]["model"]
                        # Extract other relevant config
                        if "sampling_params" in config_json:
                            row["sampling_params"] = str(config_json["sampling_params"])
                except Exception:
                    pass  # Config not available, skip

                # Fetch metrics from MLflow if available
                mlflow_run_id = pipeline.get("mlflow_run_id")
                if mlflow_run_id:
                    try:
                        metrics = mlflow_manager.get_run_metrics(mlflow_run_id)
                        if metrics:
                            # Add each metric as a column
                            for metric_name, metric_value in metrics.items():
                                # Clean metric name for DataFrame column
                                clean_name = metric_name.replace(" ", "_").lower()
                                row[clean_name] = metric_value
                    except Exception:
                        pass  # MLflow metrics not available, skip

                results.append(row)

            # Convert to DataFrame
            df = pd.DataFrame(results)

            # Reorder columns: pipeline_id first, then status, then rest
            if not df.empty:
                priority_cols = ["pipeline_id", "status", "model_name", "mlflow_run_id",
                               "shards_completed", "total_samples_processed"]
                existing_priority = [c for c in priority_cols if c in df.columns]
                other_cols = [c for c in df.columns if c not in priority_cols]
                df = df[existing_priority + other_cols]

            return df

        except Exception as e:
            raise Exception(f"Error getting results: {e}") from e

    def display_results(self, experiment_name: str = None) -> pd.DataFrame:
        """
        Get and display experiment results.

        Convenience method that calls get_results() and displays the DataFrame
        in a notebook-friendly format.

        Args:
            experiment_name: Optional experiment name

        Returns:
            pd.DataFrame with results
        """
        df = self.get_results(experiment_name)

        # Set pandas display options for better viewing
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.width", None)
        pd.set_option("display.max_colwidth", 50)

        print(f"Total pipelines: {len(df)}")
        print("\n" + "=" * 80)

        try:
            from IPython.display import display
            display(df)
        except ImportError:
            print(df.to_string())

        return df