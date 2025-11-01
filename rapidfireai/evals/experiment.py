import logging
import os
import time
from typing import Any

import ray

from rapidfireai.evals.automl import RFGridSearch, RFRandomSearch
from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.dispatcher import start_dispatcher_thread
from rapidfireai.evals.scheduling.controller import Controller
from rapidfireai.evals.utils.constants import DISPATCHER_HOST, DISPATCHER_PORT, ExperimentStatus, get_dispatcher_url
from rapidfireai.evals.utils.experiment_utils import ExperimentUtils
from rapidfireai.evals.utils.logger import RFLogger
from rapidfireai.evals.utils.notebook_ui import NotebookUI


class Experiment:
    def __init__(
        self,
        experiment_name: str,
        experiment_path: str = os.getenv("RF_EXPERIMENT_PATH", "./rapidfire_experiments"),
        openai_rpm_limit: int | None = None,
        openai_tpm_limit: int | None = None,
        openai_max_completion_tokens: int = 150,
    ):
        """
        Initialize an experiment.

        Args:
            experiment_name: Name of the experiment
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

        # OpenAI rate limiting (experiment-wide)
        self.openai_rpm_limit = openai_rpm_limit
        self.openai_tpm_limit = openai_tpm_limit
        self.openai_max_completion_tokens = openai_max_completion_tokens

        # Initialize experiment utils
        self.experiment_utils = ExperimentUtils()

        # Create experiment using experiment_utils
        self.experiment_id, self.experiment_name, log_messages = self.experiment_utils.create_experiment(
            given_name=self.experiment_name,
            experiments_path=os.path.abspath(experiment_path),
        )

        # Initialize logging
        self.logging_manager = RFLogger(experiment_name=self.experiment_name, experiment_path=self.experiment_path)
        self.logger = self.logging_manager.get_logger("Experiment")

        # Log creation messages
        for msg in log_messages:
            self.logger.info(msg)

        # Initialize Ray
        ray.init(
            logging_level=logging.INFO,
            log_to_driver=True,
            configure_logging=True,
            ignore_reinit_error=True,
        )

        # Create database reference (for run_evals usage)
        self.db = RFDatabase()

        # Initialize the controller
        self.controller = Controller(
            experiment_name=self.experiment_name,
            experiment_path=self.experiment_path,
            openai_rpm_limit=self.openai_rpm_limit,
            openai_tpm_limit=self.openai_tpm_limit,
            openai_max_completion_tokens=self.openai_max_completion_tokens,
        )

        # Start dispatcher in background thread for interactive control
        self.dispatcher_thread = start_dispatcher_thread(host=DISPATCHER_HOST, port=DISPATCHER_PORT, logger=self.logger)

        # Initialize notebook UI controller (will be displayed in run())
        # Auto-detects Colab vs Local environment and uses appropriate URL
        self.notebook_ui = NotebookUI(dispatcher_url=get_dispatcher_url())

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
            num_gpus: Number of GPUs to use (default: auto-detect from Ray cluster)
            num_cpus: Number of CPUs to use (default: auto-detect from Ray cluster)

        Returns:
            Dict mapping run_id to (aggregated_results, cumulative_metrics) tuple
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
        # Use experiment_utils to end the experiment properly
        self.experiment_utils.end_experiment(internal=False)

        # Clean shutdown Ray
        ray.shutdown()
        self.logger.info("All actors shut down")
        self.logger.info("Dispatcher will automatically shut down (daemon thread)")

    def cancel_current(self) -> None:
        """Cancel current running evaluation tasks."""
        self.experiment_utils.cancel_current(internal=False)
