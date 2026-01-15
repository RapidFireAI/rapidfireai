"""
This module contains the unified Experiment class for both fit and evals modes.
"""

import logging
import multiprocessing as mp
import os
import time
import traceback
from collections.abc import Callable
from typing import Any
from pathlib import Path
from rapidfireai.utils.ping import ping_server

import pandas as pd
from rapidfireai.utils.constants import (
    ColabConfig, 
    RayConfig, 
    RF_EXPERIMENT_PATH, 
    RF_LOG_FILENAME, 
    RF_TRAINING_LOG_FILENAME, 
    RF_LOG_PATH,
    RF_MLFLOW_ENABLED
)


class Experiment:
    """Unified Experiment class for both fit and evals modes."""

    def __init__(
        self,
        experiment_name: str,
        mode: str = "fit",
        experiment_path: str = RF_EXPERIMENT_PATH,
    ) -> None:
        """
        Initialize an experiment.

        Args:
            experiment_name: Name of the experiment
            mode: Either "fit" or "evals"
            experiment_path: Path to store experiment artifacts (default: $RF_HOME/rapidfire_experiments)

        Raises:
            ValueError: If mode is not "fit" or "evals"
        """
        # Validate mode
        if mode not in ["fit", "evals"]:
            raise ValueError(f"Invalid mode: {mode}. Must be 'fit' or 'evals'")

        self.mode = mode
        self.experiment_name = experiment_name
        self.experiment_path = experiment_path
        self.experiment_id = None

        # Initialize based on mode
        if mode == "fit":
            self._init_fit_mode()
        else:
            self._init_evals_mode()

    def _init_fit_mode(self) -> None:
        """Initialize fit-specific components."""
        # Import fit-specific modules
        from rapidfireai.fit.db.rf_db import RfDb
        from rapidfireai.fit.utils.exceptions import ExperimentException
        from rapidfireai.fit.utils.experiment_utils import ExperimentUtils
        from rapidfireai.fit.utils.logging import RFLogger
        from rapidfireai.version import __version__

        # Store exception class for use in methods
        self._ExperimentException = ExperimentException

        # Initialize fit-specific attributes
        self.log_server_process: mp.Process | None = None
        self.worker_processes: list[mp.Process] = []
        self._training_thread: Any = None  # Track background training thread (Colab only)

        # Create db tables
        try:
            RfDb().create_tables()
        except Exception as e:
            raise ExperimentException(f"Error creating db tables: {e}, traceback: {traceback.format_exc()}") from e

        # Store database reference
        self.db = RfDb()

        # Create experiment utils object
        self.experiment_utils = ExperimentUtils()

        # Create experiment
        try:
            self.experiment_id, self.experiment_name, log_messages = self.experiment_utils.create_experiment(
                given_name=self.experiment_name,
                experiments_path=os.path.abspath(self.experiment_path),
            )
        except Exception as e:
            raise ExperimentException(f"Error creating experiment: {e}, traceback: {traceback.format_exc()}") from e

        # Create logger
        try:
            self.logger = RFLogger().create_logger("experiment")
            for msg in log_messages:
                self.logger.info(msg)
            # Log the version of rapidfireai that is running
            self.logger.info(f"Running RapidFire AI version {__version__}")
        except Exception as e:
            raise ExperimentException(f"Error creating logger: {e}, traceback: {traceback.format_exc()}") from e

        # Setup signal handlers for graceful shutdown
        try:
            self.experiment_utils.setup_signal_handlers(self.worker_processes)
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error setting up signal handlers: {e}")
            raise ExperimentException(
                f"Error setting up signal handlers: {e}, traceback: {traceback.format_exc()}"
            ) from e

    def _init_evals_mode(self) -> None:
        """Initialize evals-specific components."""
        # Import evals-specific modules
        import ray

        from rapidfireai.evals.db import RFDatabase
        from rapidfireai.evals.dispatcher import start_dispatcher_thread
        from rapidfireai.evals.scheduling.controller import Controller
        from rapidfireai.utils.colab import get_colab_auth_token
        from rapidfireai.utils.constants import DispatcherConfig
        from rapidfireai.evals.utils.constants import get_dispatcher_url
        from rapidfireai.evals.utils.experiment_utils import ExperimentUtils
        from rapidfireai.evals.utils.logger import RFLogger
        from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger
        from rapidfireai.evals.utils.notebook_ui import NotebookUI

        # Store ray reference for later use
        self._ray = ray

        # Disable tokenizers parallelism warning when using with Ray/multiprocessing
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Suppress verbose third-party library logging
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
        # Disable Ray and other verbose logging
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "0"
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

        # Initialize experiment utils
        self.experiment_utils = ExperimentUtils()

        # Create experiment using experiment_utils
        self.experiment_id, self.experiment_name, log_messages = self.experiment_utils.create_experiment(
            given_name=self.experiment_name,
            experiments_path=os.path.abspath(self.experiment_path),
        )

        # Initialize logging
        self.logging_manager = RFLogger(experiment_name=self.experiment_name, experiment_path=self.experiment_path)
        self.logger = self.logging_manager.get_logger("Experiment")

        # Log creation messages
        for msg in log_messages:
            self.logger.info(msg)

        # Initialize Ray with runtime environment for CUDA initialization
        # This fixes AWS-specific CUDA/cuBLAS initialization issues
        ray.init(
            logging_level=logging.ERROR,
            log_to_driver=False,
            configure_logging=True,
            ignore_reinit_error=True,
            include_dashboard=True,
            dashboard_host=RayConfig.HOST,
            dashboard_port=RayConfig.PORT,
            # Disable metrics export to prevent "Failed to establish connection" errors
            _metrics_export_port=None,
            runtime_env={
                "env_vars": {
                    # Force CUDA to initialize properly in Ray actors (AWS fix)
                    "CUDA_LAUNCH_BLOCKING": "0",
                    "CUDA_MODULE_LOADING": "LAZY",
                    "TF_CPP_MIN_LOG_LEVEL": "3",
                    "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
                }
            },
        )
        if ColabConfig.ON_COLAB:
            try:
                from google.colab.output import eval_js

                # Get the Colab proxy URL for the dispatcher port
                proxy_url = eval_js(f"google.colab.kernel.proxyPort({RayConfig.PORT})")
                print(f"🌐 Google Colab detected. Ray dashboard URL: {proxy_url}")
            except Exception as e:
                print(f"⚠️ Colab detected but failed to get proxy URL: {e}")

        # Create database reference
        self.db = RFDatabase()

        try:
            metric_loggers = RFMetricLogger.get_default_metric_loggers(experiment_name=self.experiment_name)
            self.metric_loggers = RFMetricLogger(metric_loggers, logger=self.logger)
            metric_experiment_id = self.metric_loggers.create_experiment(self.experiment_name)
            self.db.db.execute(
                "UPDATE experiments SET metric_experiment_id = ? WHERE experiment_id = ?",
                (metric_experiment_id, self.experiment_id), commit=True
            )
            self.logger.info(f"Initialized MetricLogger experiment: {metric_experiment_id}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize MetricLogger: {e}. MetricLogger logging will be disabled.")
            self.metric_loggers = None


        
        # Initialize the controller
        self.controller = Controller(
            experiment_name=self.experiment_name,
            experiment_path=self.experiment_path,
            metric_manager=self.metric_loggers,
        )

        # Start dispatcher in background thread for interactive control
        if ping_server(DispatcherConfig.HOST, DispatcherConfig.PORT, 2):
            self.logger.info(f"Using existing dispatcher/api server at {DispatcherConfig.HOST}:{DispatcherConfig.PORT}.")
            self.dispatcher_thread = None
            
        else:
            self.logger.info(f"Starting new dispatcher/api server at {DispatcherConfig.HOST}:{DispatcherConfig.PORT}.")
            self.dispatcher_thread = start_dispatcher_thread(host=DispatcherConfig.HOST, port=DispatcherConfig.PORT, logger=self.logger)

        # Initialize notebook UI controller with auth token for Colab
        self.notebook_ui = NotebookUI(dispatcher_url=get_dispatcher_url(), auth_token=get_colab_auth_token())

    def run_fit(
        self,
        param_config: Any,
        create_model_fn: Callable,
        train_dataset: Any,
        eval_dataset: Any,
        num_chunks: int,
        seed: int = 42,
    ) -> None:
        """
        Run the fit (training).

        Args:
            param_config: Parameter configuration for training
            create_model_fn: Function to create the model
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset
            num_chunks: Number of chunks to split data into
            seed: Random seed (default: 42)

        Raises:
            ValueError: If not in fit mode
        """
        if self.mode != "fit":
            raise ValueError("run_fit() is only available in 'fit' mode")

        from rapidfireai.fit.backend.controller import Controller

        ExperimentException = self._ExperimentException

        # Check if training is already running
        if self._training_thread is not None and self._training_thread.is_alive():
            print("⚠️  Training is already running in background. Please wait for it to complete.")
            return

        if ColabConfig.ON_COLAB:
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
                            '<p style="color: blue; font-weight: bold;">'
                            "🎉 Training completed! Check InteractiveController for final results."
                            "</p>"
                        )
                    )
                    self._training_thread = None

            self._training_thread = threading.Thread(target=_run_controller_background, daemon=True)
            self._training_thread.start()

            # Use IPython display for reliable output in Colab
            display(
                HTML(
                    '<div style="padding: 10px; background-color: #d4edda; '
                    'border: 1px solid #28a745; border-radius: 5px; color: #155724;">'
                    "<b>✓ Training started in background</b><br>"
                    "Use InteractiveController to monitor progress. "
                    "The notebook kernel will remain responsive while training runs.<br>"
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

    def run_evals(
        self,
        config_group: Any,
        dataset: Any,
        num_shards: int = 4,
        seed: int = 42,
        num_actors: int = None,
        gpus_per_actor: int = None,
        cpus_per_actor: int = None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Run multi-config inference experiment with fair round-robin scheduling.

        Args:
            config_group: Grid search or random search configuration group (RFGridSearch | RFRandomSearch)
            dataset: Dataset to process
            num_shards: Number of shards to split the dataset into (default: 4)
            seed: Random seed for reproducibility (default: 42)
            num_actors: Number of Ray actors to use (default: auto-detect based on GPUs)
            gpus_per_actor: Number of GPUs per actor (default: auto-detect from Ray cluster)
            cpus_per_actor: Number of CPUs per actor (default: auto-detect from Ray cluster)

        Returns:
            Dict mapping run_id to (aggregated_results, cumulative_metrics) tuple.
            Includes COMPLETED, STOPPED, and ONGOING runs (excludes DELETED and FAILED).

        Raises:
            ValueError: If not in evals mode
        """
        if self.mode != "evals":
            raise ValueError("run_evals() is only available in 'evals' mode")

        from rapidfireai.evals.utils.constants import ExperimentStatus

        # Auto-detect resources if not provided
        available_gpus = self._ray.cluster_resources().get("GPU", 0)
        available_cpus = self._ray.cluster_resources().get("CPU", 0)

        if gpus_per_actor is None:
            gpus_per_actor = available_gpus if available_gpus > 1 else available_gpus/2
        if cpus_per_actor is None:
            cpus_per_actor = available_cpus if available_cpus > 2 else available_cpus/2
        if num_actors is None:
            # Default to number of GPUs, or 1 if no GPUs available
            num_actors = int(gpus_per_actor) if gpus_per_actor > 0 else 1

        if gpus_per_actor == 0:
            self.logger.warning("No GPUs available. Be sure to use external APIs for inference.")

        self.logger.info(
            f"Running multi-config experiment with {num_shards} shard(s), "
            f"({gpus_per_actor} GPUs per actor, {cpus_per_actor} CPUs per actor, {num_actors} actors)"
        )

        # Reset states of any existing pipelines/contexts/tasks from previous runs
        # (in case run_evals() is called multiple times on the same experiment)
        try:
            self.db.reset_experiment_states()
            self.logger.info("Reset states of existing pipelines/contexts/tasks (marked as failed)")
        except Exception as e:
            self.logger.warning(f"Failed to reset experiment states: {e}")

        # Update experiment resources in database
        self.db.set_experiment_resources(self.experiment_id, num_actors, cpus_per_actor, gpus_per_actor)

        # Display interactive control panel in notebook
        # Give dispatcher a moment to start up
        time.sleep(0.5)
        try:
            self.notebook_ui.display()
        except Exception as e:
            self.logger.warning(f"Failed to display notebook UI: {e}")

        # Update experiment with num_shards
        self.db.set_experiment_num_shards(self.experiment_id, num_shards)

        # Delegate all complexity to Controller
        try:
            results = self.controller.run_multi_pipeline_inference(
                experiment_id=self.experiment_id,
                config_group=config_group,
                dataset=dataset,
                num_shards=num_shards,
                seed=seed,
                num_actors=num_actors,
                num_gpus=gpus_per_actor,
                num_cpus=cpus_per_actor,
            )
        except Exception as e:
            self.logger.exception("Error running multi-config experiment")
            # Mark experiment as failed
            self.db.set_experiment_status(self.experiment_id, ExperimentStatus.FAILED)
            self.db.set_experiment_error(self.experiment_id, str(e))
            raise



        return results

    def get_results(self) -> pd.DataFrame:
        """
        Get the training metrics for all runs in the experiment.

        Returns:
            DataFrame with training metrics

        Raises:
            ValueError: If not in fit mode
        """
        if self.mode != "fit":
            raise ValueError("get_results() is only available in 'fit' mode")

        ExperimentException = self._ExperimentException

        try:
            runs_info_df = self.experiment_utils.get_runs_info()

            # Check if there are any metric_run_ids before importing metrics
            has_metric_runs = (
                runs_info_df.get("metric_run_id") is not None and runs_info_df["metric_run_id"].notna().any()
            )

            if not has_metric_runs or RF_MLFLOW_ENABLED != "true":
                # No metric runs to fetch, return empty DataFrame
                return pd.DataFrame(columns=["run_id", "step"])

            # Lazy import - only import when we actually have metric runs to fetch
            from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger
            try:
                metric_loggers = RFMetricLogger.get_default_metric_loggers(experiment_name=self.experiment_name)
                self.metric_loggers = RFMetricLogger(metric_loggers, logger=self.logger)

            except Exception as e:
                self.logger.warning(f"Failed to initialize MetricLogger: {e}.")
                return pd.DataFrame(columns=["run_id", "step"])

            metrics_data = []

            for _, run_row in runs_info_df.iterrows():
                run_id = run_row["run_id"]
                metric_run_id = run_row.get("metric_run_id")

                if not metric_run_id:
                    continue

                run_metrics = self.metric_loggers.get_run_metrics(metric_run_id)

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

    def get_runs_info(self) -> pd.DataFrame:
        """
        Get the run info for all runs in the experiment.

        Returns:
            DataFrame with run information

        Raises:
            ValueError: If not in fit mode
        """
        if self.mode != "fit":
            raise ValueError("get_runs_info() is only available in 'fit' mode")

        ExperimentException = self._ExperimentException

        try:
            return self.experiment_utils.get_runs_info()
        except Exception as e:
            if hasattr(self, "logger"):
                self.logger.opt(exception=True).error(f"Error getting run info: {e}")
            raise ExperimentException(f"Error getting run info: {e}, traceback: {traceback.format_exc()}") from e

    def cancel_current(self) -> None:
        """
        Cancel the current task.

        Works in both fit and evals modes.
        """
        if self.mode == "fit":
            ExperimentException = self._ExperimentException
            try:
                self.experiment_utils.cancel_current(internal=False)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.opt(exception=True).error(f"Error canceling current task: {e}")
                raise ExperimentException(
                    f"Error canceling current task: {e}, traceback: {traceback.format_exc()}"
                ) from e
        else:
            # Eval mode
            self.experiment_utils.cancel_current(internal=False)

    def end(self) -> None:
        """
        End the experiment and clean up resources.

        Works in both fit and evals modes with mode-specific cleanup.
        """
        if self.mode == "fit":
            ExperimentException = self._ExperimentException

            try:
                self.experiment_utils.end_experiment(internal=False)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.opt(exception=True).error(f"Error ending experiment: {e}")
                raise ExperimentException(f"Error ending experiment: {e}, traceback: {traceback.format_exc()}") from e

            # Shutdown all child processes
            try:
                self.experiment_utils.shutdown_workers(self.worker_processes)
            except Exception as e:
                if hasattr(self, "logger"):
                    self.logger.opt(exception=True).error(f"Error shutting down RapidFire processes: {e}")
                raise ExperimentException(
                    f"Error shutting down RapidFire processes: {e}, traceback: {traceback.format_exc()}"
                ) from e
        else:
            # Eval mode
            # Use experiment_utils to end the experiment properly
            self.experiment_utils.end_experiment(internal=False)

            # Clean shutdown Ray
            self._ray.shutdown()
            self.logger.info("All actors shut down")
            self.logger.info("Dispatcher will automatically shut down (daemon thread)")

    def get_log_file_path(self, log_type: str | None = None) -> Path:
        """
        Get the log file path for the experiment.

        Args:
            log_type: Type of log to get (default: None)

        Returns:
            Path to the log file
        """
        if log_type is None or log_type.lower() in ["main", "experiment"]:
            return Path(RF_LOG_PATH) / self.experiment_name / RF_LOG_FILENAME
        elif log_type.lower() == "training":
            return Path(RF_LOG_PATH) / self.experiment_name / RF_TRAINING_LOG_FILENAME
        else:
            raise ValueError(f"Invalid log type: {log_type}")
