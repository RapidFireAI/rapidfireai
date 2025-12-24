"""This module contains the Worker class which is responsible for handling the worker operations."""

import gc
import os
import time
import traceback
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from logging import Logger
from multiprocessing import Process
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event as EventType
from multiprocessing.synchronize import Lock
from typing import Any

import torch

from rapidfireai.fit.backend.chunks import DatasetChunks
from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.fit.ml.checkpoint_utils import (
    save_checkpoint_to_disk,
    save_checkpoint_to_shared_memory,
    save_model_to_shared_memory,
)
from rapidfireai.fit.ml.trainer import create_trainer_instance
from rapidfireai.fit.utils.constants import (
    USE_SHARED_MEMORY,
    RunStatus,
    SHMObjectType,
    TaskStatus,
    WorkerTask,
)
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.exceptions import WorkerException
from rapidfireai.fit.utils.logging import RFLogger, TrainingLogger
from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger
from rapidfireai.fit.utils.serialize import decode_db_payload
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager
from rapidfireai.fit.utils.trainer_config import TrainerConfig


class Worker:
    """Worker class that handles training and validation of runs"""

    def __init__(
        self,
        worker_id: int,
        model_registry: DictProxy,
        process_lock: Lock,
        shutdown_event: EventType,
    ):
        """Initialize the worker"""
        self.process: Process
        self.worker_id: int = worker_id
        self.shutdown_event: EventType = shutdown_event

        # Shared memory attributes (set by WorkerManager)
        self.model_registry: DictProxy[int, Any] = model_registry
        self.process_lock: Lock = process_lock

        # Shared memory manager will be created using global objects
        self.shm_manager = SharedMemoryManager(
            name=f"worker-{worker_id}-shm",
            registry=model_registry,
            multiprocess_lock=process_lock,
        )

        # create logger
        self.logger: Logger = RFLogger().create_logger(f"worker_{worker_id}")
        self.training_logger: Logger = TrainingLogger().create_logger(f"worker_{worker_id}")
        self.logger.debug(f"Worker {self.worker_id} initialized with PID {os.getpid()}")

        # create database object
        self.db: RfDb = RfDb()

        # get experiment name
        self.experiment_name: str = self.db.get_running_experiment()["experiment_name"]

        # initialize data paths
        DataPath.initialize(self.experiment_name, self.db.get_experiments_path(self.experiment_name))

        # create metric logger
        default_metric_loggers = RFMetricLogger.get_default_metric_loggers(experiment_name=self.experiment_name)
        self.metric_logger = RFMetricLogger(default_metric_loggers, logger=self.logger)
        if self.metric_logger is None:
            raise WorkerException("MetricLogger is not initialized. Please check the metric logger configuration.")
        self.metric_logger.get_experiment(self.experiment_name)

        # load datasets
        self.train_dataset, self.eval_dataset, self.num_chunks = self.load_datasets()
        self.len_train_dataset = len(self.train_dataset)

    def load_datasets(
        self,
    ) -> tuple[torch.utils.data.Dataset | None, torch.utils.data.Dataset | None, int]:
        """Load the train and eval datasets"""
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
        """Run fit"""
        self.logger.debug(f"Received run_fit on worker for run {run_id} with chunk {chunk_id}")

        # get run details
        run_details = self.db.get_run(run_id)
        config_leaf = run_details["config_leaf"]
        metric_run_id = run_details["metric_run_id"]


        # set seed
        # torch.manual_seed(run_details["seed"])
        # np.random.seed(run_details["seed"])
        # random.seed(run_details["seed"])
        effective_batch_size = config_leaf["training_args"].get("per_device_train_batch_size", 1) * config_leaf[
            "training_args"
        ].get("gradient_accumulation_steps", 1)

        # fetch train dataset chunk
        train_dataset_chunker = DatasetChunks(
            self.len_train_dataset,
            self.num_chunks,
            batch_size=effective_batch_size,
            offset=run_details["chunk_offset"],
        )
        train_dataset_chunk = train_dataset_chunker.get_chunk(self.train_dataset, chunk_id)
        # create worker config
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

        # add reward funcs to config_leaf if cloned from a GRPO run
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

        # if first time, save checkpoint to disk
        if completed_steps == 0 and not USE_SHARED_MEMORY:
            save_checkpoint_to_disk(trainer_instance, trainer_config, first=True)

        # write logs to user logger
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

        # write logs to user logger
        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

        # update completed steps
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
        else:  # save checkpoint to disk when not using shared memory
            save_checkpoint_to_disk(trainer_instance, trainer_config, completed_steps=new_completed_steps)
            self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")

        if chunk_id == self.num_chunks - 1 and new_completed_steps >= trainer_config.total_steps:
            save_checkpoint_to_disk(trainer_instance, trainer_config, last=True)
            self.logger.debug(f"Saved final checkpoint for run {run_id} on chunk {chunk_id}")

        # clean up all references to shared memory objects
        if hasattr(trainer_instance, "model"):
            del trainer_instance.model
        if hasattr(trainer_instance, "ref_model"):
            del trainer_instance.ref_model
        if hasattr(trainer_instance, "optimizer"):
            del trainer_instance.optimizer
        if hasattr(trainer_instance, "lr_scheduler"):
            del trainer_instance.lr_scheduler
        del trainer_instance

        # run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.debug(f"Completed training for run {run_id} on chunk {chunk_id}")

    def serve_forever(self) -> None:
        """This runs in the worker process"""

        prev_task_id: int | None = None
        while not (self.shutdown_event and self.shutdown_event.is_set()):
            try:
                scheduled_task = self.db.get_worker_scheduled_task(self.worker_id)
                if not scheduled_task or scheduled_task["task_id"] == prev_task_id:
                    # no new tasks or same task as previous iteration
                    time.sleep(1)
                    continue

                # get task details
                prev_task_id = scheduled_task["task_id"]
                task_type = scheduled_task["task_type"]
                run_id = scheduled_task["run_id"]
                chunk_id = scheduled_task["chunk_id"]
                create_model_fn = scheduled_task["config_options"]["create_model_fn"]
                self.logger.debug(f"Received task {task_type} for run {run_id}")

                if task_type == WorkerTask.TRAIN_VAL:
                    self.db.set_worker_task_status(self.worker_id, TaskStatus.IN_PROGRESS)

                    # run train and validation function
                    try:
                        self.run_fit(run_id, chunk_id, create_model_fn)
                        self.db.set_worker_task_status(self.worker_id, TaskStatus.COMPLETED)
                    except Exception as e:
                        self.logger.opt(exception=True).error(
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
                self.logger.opt(exception=True).error(f"Worker {self.worker_id} error: {e}")
                self.db.set_experiment_error(str(e) + "\n" + traceback.format_exc())
                break

        self.shutdown()

    def shutdown(self):
        """Called by WorkerManager to gracefully shutdown this worker"""
        self.logger.debug(f"Worker {self.worker_id} shutdown requested")
        if self.shutdown_event:
            self.shutdown_event.set()

        # Close database connection to prevent resource leaks
        try:
            if hasattr(self, "db"):
                self.db.close()
        except Exception as e:
            self.logger.debug(f"Error closing database connection: {e}")

    def is_alive(self):
        """Check if the worker process is alive"""
        return self.process and self.process.is_alive()
        return self.process and self.process.is_alive()
