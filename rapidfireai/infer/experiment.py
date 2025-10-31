import json
import logging
import os
import time
from collections.abc import Callable
from typing import Any

import ray

from rapidfireai.infer.db import RFDatabase
from rapidfireai.infer.dispatcher import start_dispatcher_thread
from rapidfireai.infer.scheduling.controller import Controller
from rapidfireai.infer.utils.constants import DISPATCHER_HOST, DISPATCHER_PORT, ExperimentStatus, get_dispatcher_url
from rapidfireai.infer.utils.notebook_ui import NotebookUI
from rapidfireai.infer.utils.logger import RFLogger


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        num_actors: int = None,
        experiment_path: str = os.getenv("RF_EXPERIMENT_PATH", "./rapidfire_experiments"),
        openai_rpm_limit: int | None = None,
        openai_tpm_limit: int | None = None,
        openai_max_completion_tokens: int = 150,
    ):
        """
        Initialize an experiment.

        Args:
            experiment_name: Name of the experiment
            num_actors: Number of Ray actors to use (default: auto-detect based on GPUs)
            experiment_path: Path to store experiment artifacts
            openai_rpm_limit: OpenAI API requests per minute limit (required if using OpenAI configs)
            openai_tpm_limit: OpenAI API tokens per minute limit (required if using OpenAI configs)
            openai_max_completion_tokens: Maximum completion tokens per OpenAI request (default: 150)
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
        self.num_actors = num_actors

        # OpenAI rate limiting (experiment-wide)
        self.openai_rpm_limit = openai_rpm_limit
        self.openai_tpm_limit = openai_tpm_limit
        self.openai_max_completion_tokens = openai_max_completion_tokens

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

        # Use all available resources on the machine
        available_gpus = ray.cluster_resources().get("GPU", 0)
        available_cpus = ray.cluster_resources().get("CPU", 0)

        self.num_gpus = available_gpus
        self.num_cpus = available_cpus

        if self.num_gpus == 0:
            self.logger.warning("No GPUs available. Be sure to use external APIs for inference.")

        self.logger.info(
            "Using all available resources: %s\nCPUs: %s\nGPUs: %s\nNumber of actors: %s",
            json.dumps(ray.cluster_resources(), indent=4),
            self.num_cpus,
            self.num_gpus,
            self.num_actors,
        )

        # Initialize the controller
        self.controller = Controller(
            num_actors=self.num_actors,
            num_gpus=self.num_gpus,
            num_cpus=self.num_cpus,
            experiment_name=self.experiment_name,
            experiment_path=self.experiment_path,
            openai_rpm_limit=self.openai_rpm_limit,
            openai_tpm_limit=self.openai_tpm_limit,
            openai_max_completion_tokens=self.openai_max_completion_tokens,
        )

        # Create experiment in database
        self.db = RFDatabase()

        # Clear all previous data for a fresh start
        self.db.clear_all_data()

        self.experiment_id = self.db.create_experiment(
            experiment_name=self.experiment_name,
            num_actors=self.num_actors,
            status=ExperimentStatus.RUNNING,
        )
        self.logger.info(f"Created experiment {self.experiment_id}: {self.experiment_name}")

        # Start dispatcher in background thread for interactive control
        self.dispatcher_thread = start_dispatcher_thread(
            host=DISPATCHER_HOST,
            port=DISPATCHER_PORT,
            logger=self.logger
        )

        # Initialize notebook UI controller (will be displayed in run())
        # Auto-detects Colab vs Local environment and uses appropriate URL
        self.notebook_ui = NotebookUI(
            dispatcher_url=get_dispatcher_url()
        )

    def run_evals(
        self,
        configs: list[tuple[str, Any]],
        dataset,
        batch_size: int,
        num_shards: int = 4,
        preprocess_fn: Callable = None,
        postprocess_fn: Callable = None,
        compute_metrics_fn: Callable = None,
        accumulate_metrics_fn: Callable = None,
        online_strategy_kwargs: dict[str, Any] = None,
    ) -> dict[int, tuple[dict, dict]]:
        """
        Run multi-config inference experiment with fair round-robin scheduling.

        Orchestrates multiple inference configurations processing shards in a time-sharing manner.
        Each configuration is scheduled fairly using generation-based round-robin scheduling.

        Args:
            configs: List of (config_name, model_config) tuples with context generators
            dataset: Dataset to process
            batch_size: Batch size for processing
            num_shards: Number of shards to split the dataset into (default: 4)
            preprocess_fn: Optional preprocessing function
            postprocess_fn: Optional postprocessing function
            compute_metrics_fn: Optional metrics computation function
            accumulate_metrics_fn: Optional metrics accumulation function
            online_strategy_kwargs: Optional online aggregation strategy parameters

        Returns:
            Dict mapping run_id to (aggregated_results, cumulative_metrics) tuple
        """
        self.logger.info(
            f"Running multi-config experiment with {len(configs)} config(s), "
            f"{num_shards} shard(s), ({self.num_gpus} GPUs, {self.num_cpus} CPUs)"
        )

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
                pipeline_configs=configs,
                dataset=dataset,
                batch_size=batch_size,
                num_shards=num_shards,
                preprocess_fn=preprocess_fn,
                postprocess_fn=postprocess_fn,
                compute_metrics_fn=compute_metrics_fn,
                accumulate_metrics_fn=accumulate_metrics_fn,
                online_strategy_kwargs=online_strategy_kwargs,
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

        return results

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
