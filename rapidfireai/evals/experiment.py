import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import ray

from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.dispatcher import start_dispatcher_thread
from rapidfireai.evals.scheduling.controller import Controller
from rapidfireai.evals.utils.constants import DISPATCHER_HOST, DISPATCHER_PORT, ExperimentStatus, get_dispatcher_url
from rapidfireai.evals.utils.notebook_ui import NotebookUI
from rapidfireai.evals.utils.logger import RFLogger
from rapidfireai.evals.automl import RFGridSearch, RFRandomSearch, RFLangChainRagSpec, RFPromptManager


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        experiment_path: str = os.getenv("RF_EXPERIMENT_PATH", "./rapidfire_experiments"),
    ):
        """
        Initialize an experiment.

        Args:
            experiment_name: Name of the experiment
            experiment_path: Path to store experiment artifacts
        """
        # Disable tokenizers parallelism warning when using with Ray/multiprocessing
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Suppress verbose third-party library logging
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("RAY_LOG_TO_STDERR", "0")
        # Disable Ray and other verbose logging (now handled by Logging class)
        os.environ["RAY_DISABLE_IMPORT_WARNING"] = "1"
        os.environ["RAY_DEDUP_LOGS"] = "0"
        os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

        self.experiment_name = experiment_name
        self.experiment_path = experiment_path

        # Store results from run_evals
        self._results: dict[int, tuple[dict, dict]] | None = None

        # Initialize logging
        self.logging_manager = RFLogger(experiment_name=self.experiment_name, experiment_path=self.experiment_path)
        self.logger = self.logging_manager.get_logger("Experiment")

        # Initialize Ray
        ray.init(
            logging_level=logging.INFO,
            log_to_driver=True,
            configure_logging=True,
            ignore_reinit_error=True,
        )

        # Create experiment in database
        self.db = RFDatabase()

        # Clear all previous data for a fresh start
        self.db.clear_all_data()

        self.experiment_id = self.db.create_experiment(
            experiment_name=self.experiment_name,
            num_actors=0,  # Will be updated in run_evals
            status=ExperimentStatus.RUNNING,
        )
        self.logger.info(f"Created experiment {self.experiment_id}: {self.experiment_name}")

        # Start dispatcher in background thread for interactive control
        self.dispatcher_thread = start_dispatcher_thread(
            host=DISPATCHER_HOST,
            port=DISPATCHER_PORT,
            logger=self.logger
        )
        
        if self.dispatcher_thread is None:
            self.logger.error(
                f"Failed to start dispatcher on {DISPATCHER_HOST}:{DISPATCHER_PORT}. "
                f"Interactive control will not be available. "
                f"Please check if port {DISPATCHER_PORT} is available."
            )
        else:
            dispatcher_url = get_dispatcher_url()
            self.logger.info(f"Dispatcher started. Notebook UI will connect to: {dispatcher_url}")

        # Initialize notebook UI controller (will be displayed in run())
        # Auto-detects Colab vs Local environment and uses appropriate URL
        self.notebook_ui = NotebookUI(
            dispatcher_url=get_dispatcher_url()
        )

    def run_evals(
        self,
        config_group: RFGridSearch | RFRandomSearch,
        dataset: Any,
        num_shards: int = 4,
        seed: int = 42,
        num_actors: int = None,
        gpus_per_actor: int = None,
        cpus_per_actor: int = None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Run multi-config inference experiment with fair round-robin scheduling.

        Orchestrates multiple inference configurations processing shards in a time-sharing manner.
        Each configuration is scheduled fairly using generation-based round-robin scheduling.

        Args:
            config_group: Grid search or random search configuration group
            dataset: Dataset to process
            num_shards: Number of shards to split the dataset into (default: 4)
            seed: Random seed for reproducibility (default: 42)
            num_actors: Number of Ray actors to use (default: auto-detect based on GPUs)
            gpus_per_actor: Number of GPUs per actor (default: auto-detect from Ray cluster)
            cpus_per_actor: Number of CPUs per actor (default: auto-detect from Ray cluster)

        Returns:
            Dict mapping run_id to (aggregated_results, cumulative_metrics) tuple

        Note:
            For OpenAI API model configs, rate limits (rpm_limit, tpm_limit) and max_completion_tokens
            should be specified directly in RFOpenAIAPIModelConfig instances.
        """
        # Auto-detect resources if not provided
        available_gpus = ray.cluster_resources().get("GPU", 0)
        available_cpus = ray.cluster_resources().get("CPU", 0)

        if gpus_per_actor is None:
            gpus_per_actor = available_gpus
        if cpus_per_actor is None:
            cpus_per_actor = available_cpus
        if num_actors is None:
            # Default to number of GPUs, or 1 if no GPUs available
            num_actors = int(gpus_per_actor) if gpus_per_actor > 0 else 1

        if gpus_per_actor == 0:
            self.logger.warning("No GPUs available. Be sure to use external APIs for inference.")

        self.logger.info(
            f"Running multi-config experiment with {num_shards} shard(s), "
            f"({gpus_per_actor} GPUs per actor, {cpus_per_actor} CPUs per actor, {num_actors} actors)"
        )

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

        # Initialize the controller (created here with rate limiting params)
        controller = Controller(
            experiment_name=self.experiment_name,
            experiment_path=self.experiment_path,
        )

        # Delegate all complexity to Controller
        # Calculate total GPUs and CPUs (controller expects total, not per-actor)
        total_gpus = gpus_per_actor * num_actors
        total_cpus = cpus_per_actor * num_actors
        
        try:
            results = controller.run_multi_pipeline_inference(
                experiment_id=self.experiment_id,
                config_group=config_group,
                dataset=dataset,
                num_shards=num_shards,
                seed=seed,
                num_actors=num_actors,
                num_gpus=total_gpus,
                num_cpus=total_cpus,
            )
        except Exception as e:
            self.logger.exception("Error running multi-config experiment")
            # Mark experiment as failed
            self.db.set_experiment_status(self.experiment_id, ExperimentStatus.FAILED)
            self.db.set_experiment_error(self.experiment_id, str(e))
            raise

        self.logger.info("Multi-config experiment completed")

        # Mark experiment as completed
        self.db.set_experiment_status(self.experiment_id, ExperimentStatus.COMPLETED)

        # Store results for later retrieval
        self._results = results


    def get_results(self) -> pd.DataFrame:
        """
        Get the results from the most recent run_evals execution as a DataFrame.

        Returns:
            DataFrame with one row per run_id and columns:
            - run_id: The pipeline/run ID
            - dataset: JSON string containing all aggregated_results (prompts, generated_text,
              ground_truth, model_answer, label, etc.) as a list of dictionaries
            - Individual columns for each metric from cumulative_metrics (e.g., Accuracy, Throughput, etc.)

        Raises:
            RuntimeError: If run_evals has not been called yet or no results are available.
        """
        if self._results is None:
            raise RuntimeError(
                "No results available. Please run experiment.run_evals() first."
            )
        
        # Convert results to DataFrame with one row per run_id
        rows = []
        for run_id, (aggregated_results, cumulative_metrics) in self._results.items():
            if not aggregated_results:
                continue
            
            row = {"run_id": run_id}
            
            # Convert aggregated_results to a list of dictionaries (dataset format)
            # Each dictionary represents one sample with all its fields
            num_samples = len(next(iter(aggregated_results.values())))
            dataset = []
            
            for i in range(num_samples):
                sample = {}
                for key, values in aggregated_results.items():
                    if isinstance(values, list) and i < len(values):
                        sample[key] = values[i]
                    else:
                        sample[key] = values
                dataset.append(sample)
            
            # Store dataset as JSON string
            row["results_dataset"] = json.dumps(dataset)
            
            # Add cumulative metrics as individual columns with confidence intervals
            if cumulative_metrics:
                for metric_key, metric_value in cumulative_metrics.items():
                    # Handle nested metrics (e.g., if Accuracy is a dict with 'value' key)
                    if isinstance(metric_value, dict):
                        # Extract value
                        if "value" in metric_value:
                            row[metric_key] = metric_value["value"]
                            
                            # Add confidence interval columns if available
                            if "lower_bound" in metric_value and metric_value["lower_bound"] is not None:
                                row[f"{metric_key}_lower_bound"] = metric_value["lower_bound"]
                            if "upper_bound" in metric_value and metric_value["upper_bound"] is not None:
                                row[f"{metric_key}_upper_bound"] = metric_value["upper_bound"]
                            if "margin_of_error" in metric_value and metric_value["margin_of_error"] is not None:
                                row[f"{metric_key}_margin_of_error"] = metric_value["margin_of_error"]
                            # Also include confidence level and method if available
                            if "confidence_level" in metric_value:
                                row[f"{metric_key}_confidence_level"] = metric_value["confidence_level"]
                            if "ci_method" in metric_value:
                                row[f"{metric_key}_ci_method"] = metric_value["ci_method"]
                        else:
                            # No value key, store as JSON
                            row[metric_key] = json.dumps(metric_value)
                    else:
                        # Simple scalar value
                        row[metric_key] = metric_value
            
            rows.append(row)
        
        if not rows:
            return pd.DataFrame(columns=["run_id", "dataset"])
        
        return pd.DataFrame(rows)

    def end(self):
        """
        Clean shutdown of experiment resources.

        Shuts down Ray actors and terminates the experiment.
        Note: Dispatcher thread is a daemon and will automatically clean up.
        """
        # Mark experiment as completed if still running (safe with try-except)
        try:
            experiment = self.db.get_experiment(self.experiment_id)
            if experiment and experiment.get("status") == ExperimentStatus.RUNNING:
                self.db.set_experiment_status(self.experiment_id, ExperimentStatus.COMPLETED)
        except Exception as e:
            self.logger.warning(f"Failed to update experiment status during shutdown: {e}")

        # Clean shutdown
        ray.shutdown()
        self.logger.info("All actors shut down")
        self.logger.info("Dispatcher will automatically shut down (daemon thread)")
