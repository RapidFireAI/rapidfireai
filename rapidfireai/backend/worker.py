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

from rapidfireai.backend.chunks import DatasetChunks
from rapidfireai.db.rf_db import RfDb
from rapidfireai.ml.checkpoint_utils import save_checkpoint_to_disk, save_model_to_shared_memory
from rapidfireai.ml.trainer import create_trainer_instance
from rapidfireai.utils.constants import MLFLOW_URL, USE_SHARED_MEMORY, RunStatus, SHMObjectType, TaskStatus, WorkerTask
from rapidfireai.utils.datapaths import DataPath
from rapidfireai.utils.distributed_utils import barrier, cleanup_distributed, is_distributed_initialized
from rapidfireai.utils.exceptions import WorkerException
from rapidfireai.utils.logging import RFLogger, TrainingLogger
from rapidfireai.utils.mlflow_manager import MLflowManager
from rapidfireai.utils.serialize import decode_db_payload
from rapidfireai.utils.shm_manager import SharedMemoryManager, SHMObjectType
from rapidfireai.utils.trainer_config import TrainerConfig


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

        # create mlflow manager
        self.mlflow_manager: MLflowManager = MLflowManager(MLFLOW_URL)
        self.mlflow_manager.get_experiment(self.experiment_name)

        # initialize data paths
        DataPath.initialize(self.experiment_name, self.db.get_experiments_path(self.experiment_name))

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
        multi_worker_details: dict[str, Any],
        create_model_fn: Callable,
    ) -> None:
        """Run fit"""
        self.logger.debug(f"Received run_fit on worker for run {run_id} with chunk {chunk_id}")

        # get run details
        run_details = self.db.get_run(run_id)
        config_leaf = run_details["config_leaf"]
        mlflow_run_id = run_details["mlflow_run_id"]
        if config_leaf.get("training_args", {}).get("fsdp_config", {}):  # FIXME: just fsdp arg
            use_fsdp = True
        else:
            use_fsdp = False
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        # Initialize distributed training if FSDP is enabled for this run
        if use_fsdp:
            try:
                from rapidfireai.utils.distributed_utils import setup_distributed_environment

                # Get distributed configuration from run details
                master_addr = multi_worker_details["master_address"]
                master_port = multi_worker_details["master_port"]
                world_size = multi_worker_details["world_size"]
                rank = multi_worker_details["local_rank"]
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, multi_worker_details["worker_ids"]))
                setup_distributed_environment(
                    rank=rank, world_size=world_size, master_addr=master_addr, master_port=master_port
                )
                self.logger.debug(f"Worker {self.worker_id} initialized distributed training for run {run_id}")
            except Exception as e:
                self.logger.error(f"Failed to initialize distributed training for run {run_id}: {e}")
                raise

        # set seed
        # torch.manual_seed(run_details["seed"])
        # np.random.seed(run_details["seed"])
        # random.seed(run_details["seed"])

        # get effective batch size
        per_device_train_batch_size = config_leaf["training_args"].get("per_device_train_batch_size", 1)
        gradient_accumulation_steps = config_leaf["training_args"].get("gradient_accumulation_steps", 1)
        effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps

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
            mlflow_run_id=mlflow_run_id,
            config_leaf=config_leaf,
            total_steps=run_details["total_steps"],
            completed_steps=run_details["completed_steps"],
            local_rank=multi_worker_details["local_rank"],
            world_size=multi_worker_details["world_size"],
            world_worker_ids=multi_worker_details["worker_ids"],
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

        # create trainer instance and write logs to user logger
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            trainer_instance, base_model_name = create_trainer_instance(
                trainer_config, self.shm_manager, USE_SHARED_MEMORY, self.mlflow_manager, chunk_id, use_fsdp=use_fsdp
            )
        trainer_instance.model.hf_device_map = {"": trainer_config.local_rank}
        # trainer_instance.model = trainer_instance.model.to(f"cuda:{trainer_config.local_rank}")
        self.logger.debug(
            f"device checkkkkkkkk: {trainer_instance.model.hf_device_map.values()}, {trainer_instance.accelerator.device}"
        )
        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

        # if first time, save checkpoint to disk
        if completed_steps == 0 and not USE_SHARED_MEMORY:
            save_checkpoint_to_disk(trainer_instance, trainer_config, first=True)

        # update base model name in db for run
        trainer_config.config_leaf["model_name"] = trainer_instance.model.config._name_or_path
        self.db.set_run_details(run_id, config_leaf=trainer_config.config_leaf)

        # train the model and time it
        self.logger.debug(f"Beginning training for run {run_id} on chunk {chunk_id}")

        # Synchronize all workers before training starts
        if use_fsdp and is_distributed_initialized():
            barrier()
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        start_time = time.time()
        # for param in trainer_instance.model.parameters():
        #     if (param.dtype == torch.float32):
        #         param.data = param.data.to(torch.bfloat16)
        with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
            trainer_instance.train()
        end_time = time.time()
        self.logger.debug(f"accelerator device: {trainer_instance.accelerator.device} on worker {self.worker_id}")

        # Synchronize all workers after training completes
        if use_fsdp and is_distributed_initialized():
            barrier()

        # update estimated runtime in database for scheduler optimization
        if trainer_config.local_rank == 0:
            runtime = end_time - start_time
            self.db.set_estimated_runtime(run_id, runtime)

        # write logs to user logger
        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

        # update completed steps
        new_completed_steps = completed_steps + trainer_instance.state.global_step
        self.db.set_completed_steps(run_id, new_completed_steps)

        save_strategy = trainer_config.config_leaf.get("training_args", {}).get("save_strategy", "epoch")
        # total_params = sum(p.numel() for p in trainer_instance.model.parameters())
        # trainable_params = sum(p.numel() for p in trainer_instance.model.parameters() if p.requires_grad)
        # self.logger.debug(f"Total parameters: {total_params}, Trainable parameters: {trainable_params} for worker {self.worker_id}")

        # save checkpoints
        if USE_SHARED_MEMORY:
            if use_fsdp and is_distributed_initialized():
                barrier()
                self.logger.debug(f"Worker {self.worker_id} passed barrier after training")

            # save checkpoints to shared memory
            save_checkpoint_to_shared_memory(trainer_instance, trainer_config, self.shm_manager, use_fsdp=use_fsdp)
            if not config_leaf.get("peft_params"):
                save_model_to_shared_memory(
                    trainer_instance.model,
                    trainer_instance.tokenizer,
                    trainer_config,
                    self.shm_manager,
                    SHMObjectType.FULL_MODEL,
                    trainer_config.run_id,
                    use_fsdp=use_fsdp,
                )
            self.logger.debug(
                f"Saved checkpoint to shared memory for run {run_id} on chunk {chunk_id}",
                f"and worker {self.worker_id}",
            )

            if use_fsdp and is_distributed_initialized():
                barrier()

            if save_strategy == "chunk" or (save_strategy == "epoch" and chunk_id == self.num_chunks - 1):
                save_checkpoint_to_disk(
                    trainer_instance,
                    trainer_config,
                    completed_steps=new_completed_steps,
                )
                self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")
        else:
            # save checkpoint to disk when not using shared memory
            save_checkpoint_to_disk(trainer_instance, trainer_config, completed_steps=new_completed_steps)
            self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")

        # save final checkpoint
        if chunk_id == self.num_chunks - 1 and new_completed_steps >= trainer_config.total_steps:
            save_checkpoint_to_disk(trainer_instance, trainer_config, last=True)
            self.logger.debug(f"Saved final checkpoint for run {run_id} on chunk {chunk_id}")

        if use_fsdp and is_distributed_initialized():
            barrier()
        if hasattr(trainer_instance.model, "_fix_weakref"):
            trainer_instance.model._fix_weakref()

        # clean up all references to shared memory objects
        if hasattr(trainer_instance, "model"):
            # if hasattr(trainer_instance.model, "cpu"):
            #     trainer_instance.model.cpu()
            del trainer_instance.model
        if hasattr(trainer_instance, "ref_model"):
            # if hasattr(trainer_instance.ref_model, "cpu"):
            #     trainer_instance.ref_model.cpu()
            del trainer_instance.ref_model
        if hasattr(trainer_instance, "optimizer"):
            trainer_instance.optimizer.zero_grad(set_to_none=True)
            if hasattr(trainer_instance.optimizer, "state"):
                trainer_instance.optimizer.state.clear()
            del trainer_instance.optimizer
        if hasattr(trainer_instance, "lr_scheduler"):
            del trainer_instance.lr_scheduler
        del trainer_instance

        # run garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if use_fsdp:
                torch.cuda.synchronize()

        if use_fsdp and is_distributed_initialized():
            barrier()
        self.logger.debug(f"Completed training for run {run_id} on chunk {chunk_id}")

    def serve_forever(self) -> None:
        """The main loop for the worker"""

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
                multi_worker_details = scheduled_task["multi_worker_details"]
                create_model_fn = scheduled_task["config_options"]["create_model_fn"]
                self.logger.debug(f"Received task {task_type} for run {run_id}")

                if task_type == WorkerTask.TRAIN_VAL:
                    self.db.set_worker_task_status(self.worker_id, TaskStatus.IN_PROGRESS)

                    # run train and validation function
                    try:
                        self.run_fit(run_id, chunk_id, multi_worker_details, create_model_fn)
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
                if is_distributed_initialized():
                    cleanup_distributed()
                    self.logger.debug(f"Worker {self.worker_id} distributed training cleaned up")
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

        # Clean up distributed training if enabled
        if is_distributed_initialized():
            try:
                cleanup_distributed()
                self.logger.debug(f"Worker {self.worker_id} distributed training cleaned up")
            except Exception as e:
                self.logger.debug(f"Error during distributed cleanup: {e}")

        # Close database connection to prevent resource leaks
        try:
            if hasattr(self, "db"):
                self.db.close()
        except Exception as e:
            self.logger.debug(f"Error closing database connection: {e}")

    def is_alive(self):
        """Check if the worker process is alive"""
        return self.process and self.process.is_alive()
