"""
Ray Actor-based Worker for distributed training.

This module provides a Ray remote class that replaces the multiprocessing-based Worker.
Each WorkerActor runs on a separate GPU and handles training tasks assigned by the Controller.
"""

import gc
import os
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from logging import Logger
from multiprocessing import Manager
from typing import Any

import ray
import torch

from rapidfireai.db import RfDb
from rapidfireai.fit.backend.chunks import DatasetChunks
from rapidfireai.fit.ml.checkpoint_utils import (
    save_checkpoint_to_disk,
    save_checkpoint_to_shared_memory,
    save_model_to_shared_memory,
)
from rapidfireai.fit.ml.trainer import create_trainer_instance
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager, USE_SHARED_MEMORY
from rapidfireai.fit.utils.trainer_config import TrainerConfig
from rapidfireai.utils.constants import (
    MLFlowConfig,
    RunStatus,
    SHMObjectType,
    TaskStatus,
    WorkerTask,
)
from rapidfireai.utils.exceptions import WorkerException
from rapidfireai.utils.logging import RFLogger, TrainingLogger
from rapidfireai.metrics import RFMetricLogger
from rapidfireai.utils.serialize import decode_db_payload


@ray.remote
class WorkerActor:
    """
    Ray Actor-based Worker for handling training and validation.

    Each WorkerActor:
    - Runs on its own GPU (resource allocation handled by Ray)
    - Polls the database for scheduled tasks
    - Executes training chunks when assigned
    - Saves checkpoints to shared memory and disk
    """

    def __init__(
        self,
        worker_id: int,
        model_registry: dict[int, Any] = None,
        process_lock: Any = None,
    ):
        """
        Initialize the WorkerActor.

        Args:
            worker_id: Unique identifier for this worker
            model_registry: (Ignored in Ray mode) - workers use local storage
            process_lock: (Ignored in Ray mode) - workers use local locks
        """
        self.worker_id: int = worker_id
        self._shutdown_requested: bool = False

        # In Ray mode, each worker has its own local registry and lock
        # (multiprocessing.Manager() proxies don't work across Ray actors)
        # Workers communicate through the database, not shared memory
        self.shm_manager = SharedMemoryManager(
            name=f"worker-{worker_id}-shm",
            registry=None,  # Create local registry
            multiprocess_lock=None,  # Create local lock
        )
        self.model_registry, self.process_lock = self.shm_manager.get_shm_objects()

        # Create database connection
        self.db: RfDb = RfDb()

        # Get experiment info
        running_experiment = self.db.get_running_experiment()
        self.experiment_name: str = running_experiment["experiment_name"]
        self.experiment_id: int = running_experiment["experiment_id"]

        # Create loggers with experiment name
        self.logger: Logger = RFLogger(experiment_name=self.experiment_name).get_logger(f"worker_{worker_id}")
        self.training_logger: Logger = TrainingLogger(experiment_name=self.experiment_name).get_logger(f"worker_{worker_id}")
        self.logger.debug(f"WorkerActor {self.worker_id} initialized with PID {os.getpid()}")

        # Initialize data paths
        DataPath.initialize(self.experiment_name, self.db.get_experiments_path(self.experiment_id))

        # create metric logger
        default_metric_loggers = RFMetricLogger.get_default_metric_loggers(experiment_name=self.experiment_name)
        self.metric_logger = RFMetricLogger(default_metric_loggers, logger=self.logger)
        if self.metric_logger is None:
            raise WorkerException("MetricLogger is not initialized. Please check the metric logger configuration.")
        self.metric_logger.get_experiment(self.experiment_name)

        # Load datasets
        self.train_dataset, self.eval_dataset, self.num_chunks = self._load_datasets()
        self.len_train_dataset = len(self.train_dataset)

        self.logger.info(f"WorkerActor {self.worker_id} ready on GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')}")

    def _load_datasets(
        self,
    ) -> tuple[torch.utils.data.Dataset | None, torch.utils.data.Dataset | None, int]:
        """Load the train and eval datasets from disk."""
        try:
            with open(DataPath.dataset_path(), "rb") as f:
                datasets = decode_db_payload(f.read())
            self.logger.debug("Loaded datasets")
            return datasets["train"], datasets["eval"], datasets["num_chunks"]
        except Exception as e:
            raise WorkerException(f"Error loading datasets: {e}") from e

    def run_fit(
        self,
        run_id: int,
        chunk_id: int,
        create_model_fn: Callable,
    ) -> None:
        """
        Execute a training chunk for the specified run.

        Args:
            run_id: ID of the training run
            chunk_id: ID of the chunk to train on
            create_model_fn: Function to create/load the model
        """
        self.logger.debug(f"Received run_fit on worker for run {run_id} with chunk {chunk_id}")

        # Get run details
        run_details = self.db.get_run(run_id)
        config_leaf = run_details["config_leaf"]
        metric_run_id = run_details["metric_run_id"]


        effective_batch_size = config_leaf["training_args"].get("per_device_train_batch_size", 1) * config_leaf[
            "training_args"
        ].get("gradient_accumulation_steps", 1)

        # Fetch train dataset chunk
        train_dataset_chunker = DatasetChunks(
            self.len_train_dataset,
            self.num_chunks,
            batch_size=effective_batch_size,
            offset=run_details["chunk_offset"],
        )
        train_dataset_chunk = train_dataset_chunker.get_chunk(self.train_dataset, chunk_id)

        # Create trainer config
        trainer_config = TrainerConfig(
            worker_id=self.worker_id,
            run_id=run_id,
            metric_run_id=metric_run_id,
            config_leaf=config_leaf,
            total_steps=run_details["total_steps"],
            completed_steps=run_details["completed_steps"],
            create_model_fn=create_model_fn,
            train_dataset=train_dataset_chunk,
            eval_dataset=self.eval_dataset,
            warm_started_from=run_details["warm_started_from"],
            cloned_from=run_details["cloned_from"],
            num_epochs_completed=run_details["num_epochs_completed"],
        )
        completed_steps = self.db.get_completed_steps(run_id)

        # Add reward funcs to config_leaf if cloned from a GRPO run
        if trainer_config.cloned_from is not None and trainer_config.config_leaf.get("trainer_type") == "GRPO":
            parent_run_details = self.db.get_run(trainer_config.cloned_from)
            config_leaf["reward_funcs"] = parent_run_details["config_leaf"].get("reward_funcs")
            self.db.set_run_details(run_id, config_leaf=config_leaf)

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            trainer_instance, _ = create_trainer_instance(
                trainer_config, self.shm_manager, USE_SHARED_MEMORY, self.metric_logger, chunk_id
            )

        # If first time, save checkpoint to disk
        if completed_steps == 0 and not USE_SHARED_MEMORY:
            save_checkpoint_to_disk(trainer_instance, trainer_config, first=True)

        # Write logs to user logger
        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

        self.logger.debug(f"Beginning training for run {run_id} on chunk {chunk_id}")

        # Train the model
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            trainer_instance.train()

        # Write logs to user logger
        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

        # Update completed steps
        new_completed_steps = completed_steps + trainer_instance.state.global_step
        self.db.set_completed_steps(run_id, new_completed_steps)

        save_strategy = config_leaf.get("training_args", {}).get("save_strategy", "epoch")
        # Save checkpoints to shared memory
        if USE_SHARED_MEMORY:
            save_checkpoint_to_shared_memory(trainer_instance, trainer_config, self.shm_manager)
            if not trainer_config.config_leaf.get("peft_params"):
                save_model_to_shared_memory(
                    trainer_instance.model,
                    trainer_instance.tokenizer,
                    trainer_config,
                    self.shm_manager,
                    SHMObjectType.FULL_MODEL,
                    trainer_config.run_id,
                )
            self.logger.debug(f"Saved checkpoint to shared memory for run {run_id} on chunk {chunk_id}")
            if save_strategy == "chunk" or (save_strategy == "epoch" and chunk_id == self.num_chunks - 1):
                save_checkpoint_to_disk(
                    trainer_instance,
                    trainer_config,
                    completed_steps=new_completed_steps,
                )
                self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")
        else:  # Save checkpoint to disk when not using shared memory
            save_checkpoint_to_disk(trainer_instance, trainer_config, completed_steps=new_completed_steps)
            self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")

        if chunk_id == self.num_chunks - 1 and new_completed_steps >= trainer_config.total_steps:
            save_checkpoint_to_disk(trainer_instance, trainer_config, last=True)
            self.logger.debug(f"Saved final checkpoint for run {run_id} on chunk {chunk_id}")

        # Clean up all references to free GPU memory
        if hasattr(trainer_instance, "model"):
            del trainer_instance.model
        if hasattr(trainer_instance, "ref_model"):
            del trainer_instance.ref_model
        if hasattr(trainer_instance, "optimizer"):
            del trainer_instance.optimizer
        if hasattr(trainer_instance, "lr_scheduler"):
            del trainer_instance.lr_scheduler
        del trainer_instance

        # Run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.debug(f"Completed training for run {run_id} on chunk {chunk_id}")

    def serve_forever(self) -> None:
        """
        Main loop: poll the database for scheduled tasks and execute them.

        This method runs continuously until shutdown is requested (via request_shutdown())
        or a critical error occurs.
        """
        prev_task_id: int | None = None

        while not self._shutdown_requested:
            try:
                scheduled_task = self.db.get_worker_scheduled_task(self.worker_id)
                if not scheduled_task or scheduled_task["task_id"] == prev_task_id:
                    # No new tasks or same task as previous iteration
                    time.sleep(1)
                    continue

                # Get task details
                prev_task_id = scheduled_task["task_id"]
                task_type = scheduled_task["task_type"]
                run_id = scheduled_task["run_id"]
                chunk_id = scheduled_task["chunk_id"]
                create_model_fn = scheduled_task["config_options"]["create_model_fn"]
                self.logger.debug(f"Received task {task_type} for run {run_id}")

                if task_type == WorkerTask.TRAIN_VAL:
                    self.db.set_worker_task_status(self.worker_id, TaskStatus.IN_PROGRESS)

                    # Run train and validation function
                    try:
                        self.run_fit(run_id, chunk_id, create_model_fn)
                        self.db.set_worker_task_status(self.worker_id, TaskStatus.COMPLETED)
                    except Exception as e:
                        self.logger.exception(
                            f"Error while running run_fit for run {run_id} and chunk {chunk_id}: {e}"
                        )
                        self.db.set_run_details(
                            run_id,
                            status=RunStatus.FAILED,
                            error=str(e) + traceback.format_exc(),
                        )
                        self.db.set_worker_task_status(self.worker_id, TaskStatus.FAILED)
                else:
                    raise WorkerException(f"Invalid task type: {task_type}")
            except Exception as e:
                self.logger.exception(f"WorkerActor {self.worker_id} error: {e}")
                self.db.set_experiment_error(str(e) + "\n" + traceback.format_exc())
                break

        self._cleanup()

    def request_shutdown(self) -> bool:
        """
        Request graceful shutdown of this worker.

        The worker will complete its current chunk before shutting down.

        Returns:
            True if shutdown was requested successfully
        """
        self.logger.debug(f"WorkerActor {self.worker_id} shutdown requested")
        self._shutdown_requested = True
        return True

    def is_shutdown_requested(self) -> bool:
        """Check if shutdown has been requested."""
        return self._shutdown_requested

    def _cleanup(self) -> None:
        """Clean up resources on shutdown."""
        self.logger.debug(f"WorkerActor {self.worker_id} cleaning up")

        # Close database connection
        try:
            if hasattr(self, "db"):
                self.db.close()
        except Exception as e:
            self.logger.debug(f"Error closing database connection: {e}")

        self.logger.info(f"WorkerActor {self.worker_id} shutdown complete")

    def get_worker_id(self) -> int:
        """Return this worker's ID."""
        return self.worker_id

    def ping(self) -> bool:
        """Health check endpoint."""
        return True


def create_worker_actors(
    num_workers: int,
    gpus_per_worker: int,
    cpus_per_worker: int,
    model_registry: dict,
    process_lock: Any,
) -> list:
    """
    Create WorkerActor instances with proper GPU allocation.

    Args:
        num_workers: Number of workers to create
        gpus_per_worker: Number of GPUs per worker (typically 1)
        cpus_per_worker: Number of CPUs per worker
        model_registry: Shared dictionary for model storage
        process_lock: Lock for synchronizing shared memory access

    Returns:
        List of WorkerActor handles
    """
    workers = []

    for worker_id in range(num_workers):
        # Create actor with resource allocation
        worker = WorkerActor.options(
            num_gpus=gpus_per_worker,
            num_cpus=cpus_per_worker,
            name=f"fit_worker_{worker_id}",
        ).remote(
            worker_id=worker_id,
            model_registry=model_registry,
            process_lock=process_lock,
        )
        workers.append(worker)

    return workers

