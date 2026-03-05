"""
Ray Actor-based Worker for distributed training.

This module provides a Ray remote class that replaces the multiprocessing-based Worker.
Each WorkerActor runs on a separate GPU and handles training tasks assigned by the Controller.
"""

import gc
import os
import sys
import time
import traceback
from collections.abc import Callable
from io import StringIO
from logging import Logger
from typing import Any

import ray
import torch

from rapidfireai.db import RfDb
from rapidfireai.fit.backend.chunks import DatasetChunks
from rapidfireai.fit.ml.checkpoint_utils import (
    flush_cuda_cache,
    purge_model_kv_caches,
    release_model_gpu_memory,
    save_checkpoint_to_disk,
    save_checkpoint_to_shared_memory,
    save_model_to_shared_memory,
)
from rapidfireai.fit.ml.trainer import create_trainer_instance
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager, USE_SHARED_MEMORY
from rapidfireai.fit.utils.trainer_config import TrainerConfig
from rapidfireai.utils.constants import (
    MLflowConfig,
    RunStatus,
    SHMObjectType,
    TaskStatus,
    WorkerTask,
)
from rapidfireai.utils.distributed_utils import (
    barrier,
    cleanup_distributed,
    is_distributed_initialized,
    setup_distributed_environment,
    find_free_port,
)
from rapidfireai.utils.exceptions import WorkerException
from rapidfireai.utils.logging import RFLogger, TrainingLogger
from rapidfireai.metrics import RFMetricLogger
from rapidfireai.utils.serialize import decode_db_payload


def _scan_and_release_leaked_cuda_tensors(skip_quantized: bool = False):
    """Zero all gc-tracked CUDA tensors unreachable by model iteration."""
    gc.collect()
    gc.collect()

    bnb_tensor_ids: set[int] = set()
    if skip_quantized:
        try:
            from bitsandbytes.functional import QuantState

            for obj in gc.get_objects():
                if not isinstance(obj, QuantState):
                    continue
                for attr_name in ("absmax", "code", "offset"):
                    t = getattr(obj, attr_name, None)
                    if torch.is_tensor(t):
                        bnb_tensor_ids.add(id(t))
                nested = getattr(obj, "state2", None)
                if isinstance(nested, QuantState):
                    for attr_name in ("absmax", "code", "offset"):
                        t = getattr(nested, attr_name, None)
                        if torch.is_tensor(t):
                            bnb_tensor_ids.add(id(t))
        except ImportError:
            pass

    _empty = torch.empty(0, device="cpu")
    for obj in gc.get_objects():
        if not torch.is_tensor(obj):
            continue
        try:
            if obj.is_cuda and id(obj) not in bnb_tensor_ids:
                obj.data = _empty
        except Exception:
            pass


class TeeOutput:
    """Captures output to a buffer while providing fileno() for vLLM compatibility."""

    def __init__(self, original_stream, buffer):
        self.original_stream = original_stream
        self.buffer = buffer

    def write(self, text):
        self.buffer.write(text)

    def flush(self):
        self.buffer.flush()

    def fileno(self):
        return self.original_stream.fileno()

    def isatty(self):
        return self.original_stream.isatty()

    def getvalue(self):
        return self.buffer.getvalue()

    def __getattr__(self, name):
        return getattr(self.original_stream, name)


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
        registry_actor,
    ):
        """
        Initialize the WorkerActor.

        Args:
            worker_id: Unique identifier for this worker
            registry_actor: Ray actor handle for the shared RegistryActor
        """
        self.worker_id: int = worker_id
        self._shutdown_requested: bool = False

        # Create shared memory manager using the registry actor
        # The registry actor is shared across all workers for coordination
        self.shm_manager = SharedMemoryManager(
            name=f"worker-{worker_id}-shm",
            registry_actor=registry_actor,
        )

        self.db: RfDb = RfDb()

        running_experiment = self.db.get_running_experiment()
        self.experiment_name: str = running_experiment["experiment_name"]
        self.experiment_id: int = running_experiment["experiment_id"]

        self.logger: Logger = RFLogger(experiment_name=self.experiment_name).get_logger(f"worker_{worker_id}")
        self.training_logger: Logger = TrainingLogger(experiment_name=self.experiment_name).get_logger(f"worker_{worker_id}")
        self.logger.debug(f"WorkerActor {self.worker_id} initialized with PID {os.getpid()}")

        DataPath.initialize(self.experiment_name, self.db.get_experiments_path(self.experiment_id))

        default_metric_loggers = RFMetricLogger.get_default_metric_loggers(experiment_name=self.experiment_name)
        self.metric_logger = RFMetricLogger(default_metric_loggers, logger=self.logger)
        if self.metric_logger is None:
            raise WorkerException("MetricLogger is not initialized. Please check the metric logger configuration.")
        self.metric_logger.get_experiment(self.experiment_name)

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
        multi_worker_details: dict[str, Any],
        create_model_fn: Callable,
    ) -> None:
        """
        Execute a training chunk for the specified run.

        Args:
            run_id: ID of the training run
            chunk_id: ID of the chunk to train on
            multi_worker_details: FSDP distributed training configuration
            create_model_fn: Function to create/load the model
        """
        self.logger.debug(f"Received run_fit on worker for run {run_id} with chunk {chunk_id}")

        run_details = self.db.get_run(run_id)
        config_leaf = run_details["config_leaf"]
        metric_run_id = run_details["metric_run_id"]

        use_fsdp = (
            "training_args" in config_leaf
            and "fsdp_config" in config_leaf["training_args"]
        )

        if config_leaf.get("trainer_type", "SFT") == "GRPO":
            try:
                from vllm.distributed import parallel_state

                if hasattr(parallel_state, "destroy_model_parallel"):
                    try:
                        parallel_state.destroy_model_parallel()
                    except Exception:
                        pass
                if hasattr(parallel_state, "destroy_distributed_environment"):
                    try:
                        parallel_state.destroy_distributed_environment()
                    except Exception:
                        pass
            except (ImportError, AttributeError):
                pass

        if use_fsdp:
            try:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                master_addr = multi_worker_details["master_address"]
                master_port = multi_worker_details["master_port"]
                world_size = multi_worker_details["world_size"]
                rank = multi_worker_details["local_rank"]
                self.logger.debug(
                    f"Worker {self.worker_id} initializing distributed training for run {run_id} "
                    f"with master_addr {master_addr}, master_port {master_port}, world_size {world_size}, rank {rank}"
                )
                setup_distributed_environment(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
                    local_rank=0,
                )
                self.logger.debug(
                    f"Worker {self.worker_id} initialized distributed training for run {run_id}"
                )
            except Exception as e:
                self.logger.error(
                    f"Failed to initialize distributed training for run {run_id}: {e}"
                )
                raise
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.worker_id)
            if config_leaf.get("trainer_type", "SFT") == "GRPO":
                master_port = find_free_port()
                os.environ["MASTER_PORT"] = str(master_port)

        per_device_train_batch_size = config_leaf["training_args"].get("per_device_train_batch_size", 1)
        gradient_accumulation_steps = config_leaf["training_args"].get("gradient_accumulation_steps", 1)
        effective_batch_size = (
            per_device_train_batch_size
            * gradient_accumulation_steps
            * multi_worker_details.get("world_size", 1)
        )

        train_dataset_chunker = DatasetChunks(
            self.len_train_dataset,
            self.num_chunks,
            batch_size=effective_batch_size,
            offset=run_details["chunk_offset"],
        )
        train_dataset_chunk = train_dataset_chunker.get_chunk(self.train_dataset, chunk_id)

        # NOTE: local_rank here is the FSDP group rank (0..N-1), NOT the CUDA device
        # index. Each Ray actor sees only its assigned GPU as cuda:0, so the device
        # index is always 0. This rank is used for conditional logic (e.g., only rank 0
        # saves checkpoints) throughout checkpoint_utils.py and this file.
        trainer_config = TrainerConfig(
            worker_id=self.worker_id,
            run_id=run_id,
            metric_run_id=metric_run_id,
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

        if trainer_config.cloned_from is not None and trainer_config.config_leaf.get("trainer_type") == "GRPO":
            parent_run_details = self.db.get_run(trainer_config.cloned_from)
            config_leaf["reward_funcs"] = parent_run_details["config_leaf"].get("reward_funcs")
            self.db.set_run_details(run_id, config_leaf=config_leaf)

        if use_fsdp and is_distributed_initialized():
            barrier()

        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        devnull_fd = None
        try:
            with open(os.devnull, "w") as devnull:
                devnull_fd = devnull.fileno()
                if hasattr(sys.stdout, "fileno"):
                    os.dup2(devnull_fd, sys.stdout.fileno())
                if hasattr(sys.stderr, "fileno"):
                    os.dup2(devnull_fd, sys.stderr.fileno())
        except Exception as e:
            self.logger.debug(f"Could not redirect stdout/stderr to /dev/null: {e}")

        tee_stdout = TeeOutput(original_stdout, stdout_buffer)
        tee_stderr = TeeOutput(original_stderr, stderr_buffer)

        try:
            sys.stdout = tee_stdout
            sys.stderr = tee_stderr

            trainer_instance, base_model_name = create_trainer_instance(
                trainer_config,
                self.shm_manager,
                USE_SHARED_MEMORY,
                self.metric_logger,
                chunk_id,
                use_fsdp=use_fsdp,
                db=self.db,
            )
            is_quantized = bool(
                config_leaf.get("model_kwargs", {}).get("quantization_config")
                or getattr(trainer_instance.model.config, "quantization_config", None) is not None
            )

            completed_steps = self.db.get_completed_steps(run_id)
            if completed_steps == 0 and not USE_SHARED_MEMORY:
                save_checkpoint_to_disk(
                    trainer_instance, trainer_config, first=True, use_fsdp=use_fsdp
                )

            trainer_config.config_leaf["model_name"] = trainer_instance.model.config._name_or_path
            self.db.set_run_details(run_id, config_leaf=trainer_config.config_leaf)

            self.logger.debug(f"Beginning training for run {run_id} on chunk {chunk_id}")

            if use_fsdp and is_distributed_initialized():
                barrier()

            start_time = time.time()
            trainer_instance.train()
            end_time = time.time()

            if use_fsdp and is_distributed_initialized():
                barrier()

            new_completed_steps = completed_steps + trainer_instance.state.global_step
            self.db.set_completed_steps(run_id, new_completed_steps)

            if trainer_config.local_rank == 0:
                new_runtime_per_batch = (
                    end_time - start_time
                ) / train_dataset_chunker.get_chunk_size(chunk_id)
                running_average_runtime = (
                    run_details["estimated_runtime"] * completed_steps
                    + new_runtime_per_batch
                ) / new_completed_steps
                self.db.set_estimated_runtime(run_id, running_average_runtime)

            if USE_SHARED_MEMORY:
                if use_fsdp and is_distributed_initialized():
                    barrier()
                    self.logger.debug(f"Worker {self.worker_id} passed barrier after training")

                if use_fsdp and torch.cuda.is_available():
                    try:
                        if hasattr(trainer_instance, "optimizer") and trainer_instance.optimizer is not None:
                            trainer_instance.optimizer.zero_grad(set_to_none=True)
                            for param_group in trainer_instance.optimizer.param_groups:
                                for p in param_group["params"]:
                                    if p in trainer_instance.optimizer.state:
                                        for key in list(trainer_instance.optimizer.state[p].keys()):
                                            t = trainer_instance.optimizer.state[p][key]
                                            if torch.is_tensor(t) and t.is_cuda:
                                                trainer_instance.optimizer.state[p][key] = t.to("cpu")
                    except Exception:
                        pass

                    purge_model_kv_caches(trainer_instance.model)

                    try:
                        if hasattr(trainer_instance, "_memory_tracker"):
                            trainer_instance._memory_tracker = None
                    except Exception:
                        pass

                flush_cuda_cache()

                is_run_finished = (
                    chunk_id == self.num_chunks - 1
                    and new_completed_steps >= trainer_config.total_steps
                )

                if is_run_finished:
                    save_checkpoint_to_disk(
                        trainer_instance, trainer_config, last=True, use_fsdp=use_fsdp,
                    )
                    self.logger.debug(f"Saved final checkpoint to disk for run {run_id} on chunk {chunk_id}")
                else:
                    save_checkpoint_to_shared_memory(
                        trainer_instance, trainer_config, self.shm_manager, use_fsdp=use_fsdp,
                    )
                    if not config_leaf.get("peft_params") and not use_fsdp:
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
                        f"Saved checkpoint to shared memory for run {run_id} on chunk {chunk_id}"
                        f" and worker {self.worker_id}",
                    )

                    if use_fsdp and is_distributed_initialized():
                        barrier()

                    save_strategy = trainer_config.config_leaf.get("training_args", {}).get("save_strategy", "epoch")
                    if save_strategy == "chunk":
                        save_checkpoint_to_disk(
                            trainer_instance, trainer_config,
                            completed_steps=new_completed_steps, use_fsdp=use_fsdp,
                        )
                        self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")
            else:
                save_checkpoint_to_disk(
                    trainer_instance, trainer_config,
                    completed_steps=new_completed_steps, use_fsdp=use_fsdp,
                )
                self.logger.debug(f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}")

            if (
                not USE_SHARED_MEMORY
                and chunk_id == self.num_chunks - 1
                and new_completed_steps >= trainer_config.total_steps
            ):
                save_checkpoint_to_disk(
                    trainer_instance, trainer_config, last=True, use_fsdp=use_fsdp
                )
                self.logger.debug(f"Saved final checkpoint for run {run_id} on chunk {chunk_id}")

            if use_fsdp and is_distributed_initialized():
                barrier()

        except Exception as e:
            self.logger.error(f"Error during training for run {run_id}: {e}")
            raise
        finally:
            try:
                if "trainer_instance" in locals() and hasattr(trainer_instance, "llm"):
                    try:
                        if hasattr(trainer_instance.llm, "shutdown"):
                            trainer_instance.llm.shutdown()
                        elif hasattr(trainer_instance.llm, "llm_engine"):
                            if hasattr(trainer_instance.llm.llm_engine, "engine_core"):
                                if hasattr(trainer_instance.llm.llm_engine.engine_core, "shutdown"):
                                    trainer_instance.llm.llm_engine.engine_core.shutdown()
                        try:
                            from vllm.distributed import parallel_state

                            if hasattr(parallel_state, "destroy_model_parallel"):
                                parallel_state.destroy_model_parallel()
                            if hasattr(parallel_state, "destroy_distributed_environment"):
                                parallel_state.destroy_distributed_environment()
                        except (ImportError, AttributeError):
                            pass
                        del trainer_instance.llm
                    except Exception as cleanup_e:
                        self.logger.debug(f"Error cleaning up vLLM engine: {cleanup_e}")
                        if hasattr(trainer_instance, "llm"):
                            try:
                                del trainer_instance.llm
                            except Exception:
                                pass

                if "trainer_instance" in locals():
                    _is_quantized_fsdp = (
                        use_fsdp and "is_quantized" in dir() and is_quantized
                    )

                    try:
                        if hasattr(trainer_instance, "callback_handler"):
                            trainer_instance.callback_handler.model = None
                            trainer_instance.callback_handler.optimizer = None
                            trainer_instance.callback_handler.lr_scheduler = None
                    except Exception:
                        pass

                    try:
                        if hasattr(trainer_instance, "optimizer") and trainer_instance.optimizer is not None:
                            trainer_instance.optimizer.zero_grad(set_to_none=True)
                            if hasattr(trainer_instance.optimizer, "state"):
                                trainer_instance.optimizer.state.clear()
                            if hasattr(trainer_instance.optimizer, "param_groups"):
                                for pg in trainer_instance.optimizer.param_groups:
                                    pg["params"] = []
                    except Exception:
                        pass

                    if use_fsdp and not _is_quantized_fsdp:
                        for attr in ("model", "model_wrapped", "ref_model"):
                            try:
                                release_model_gpu_memory(getattr(trainer_instance, attr, None))
                            except Exception:
                                pass

                    for attr in ("optimizer", "lr_scheduler", "model_wrapped",
                                 "model", "ref_model"):
                        try:
                            if hasattr(trainer_instance, attr):
                                delattr(trainer_instance, attr)
                        except Exception:
                            pass

                    try:
                        if hasattr(trainer_instance, "accelerator"):
                            if hasattr(trainer_instance.accelerator, "free_memory"):
                                trainer_instance.accelerator.free_memory()
                            del trainer_instance.accelerator
                    except Exception:
                        pass

                    for attr in ("tokenizer", "state", "callback_handler",
                                 "_memory_tracker", "data_collator"):
                        try:
                            if hasattr(trainer_instance, attr):
                                delattr(trainer_instance, attr)
                        except Exception:
                            pass

                    try:
                        del trainer_instance
                    except Exception:
                        pass
                else:
                    _is_quantized_fsdp = False

                flush_cuda_cache()

                if torch.cuda.is_available():
                    if use_fsdp and not _is_quantized_fsdp:
                        _scan_and_release_leaked_cuda_tensors(
                            skip_quantized=("is_quantized" in dir() and is_quantized)
                        )
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                if use_fsdp and is_distributed_initialized():
                    barrier()
            except Exception as cleanup_error:
                self.logger.error(f"Error during cleanup for run {run_id}: {cleanup_error}")
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

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
                    time.sleep(1)
                    continue

                prev_task_id = scheduled_task["task_id"]
                task_type = scheduled_task["task_type"]
                run_id = scheduled_task["run_id"]
                chunk_id = scheduled_task["chunk_id"]
                multi_worker_details = scheduled_task["multi_worker_details"]
                create_model_fn = scheduled_task["config_options"]["create_model_fn"]
                self.logger.debug(f"Received task {task_type} for run {run_id}")

                if task_type == WorkerTask.TRAIN_VAL:
                    self.db.set_worker_task_status(self.worker_id, TaskStatus.IN_PROGRESS)

                    try:
                        self.run_fit(run_id, chunk_id, multi_worker_details, create_model_fn)
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
                if is_distributed_initialized():
                    cleanup_distributed()
                    self.logger.debug(f"Worker {self.worker_id} distributed training cleaned up")
            except Exception as e:
                self.logger.exception(f"WorkerActor {self.worker_id} error: {e}")
                self.db.set_experiment_error(self.experiment_id, str(e) + "\n" + traceback.format_exc())
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

        if is_distributed_initialized():
            try:
                cleanup_distributed()
                self.logger.debug(f"WorkerActor {self.worker_id} distributed training cleaned up")
            except Exception as e:
                self.logger.debug(f"Error during distributed cleanup: {e}")

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
    registry_actor,
) -> list:
    """
    Create WorkerActor instances with proper GPU allocation.

    Args:
        num_workers: Number of workers to create
        gpus_per_worker: Number of GPUs per worker (typically 1)
        cpus_per_worker: Number of CPUs per worker
        registry_actor: Ray actor handle for the shared RegistryActor

    Returns:
        List of WorkerActor handles
    """
    workers = []

    for worker_id in range(num_workers):
        worker = WorkerActor.options(
            num_gpus=gpus_per_worker,
            num_cpus=cpus_per_worker,
            name=f"fit_worker_{worker_id}",
        ).remote(
            worker_id=worker_id,
            registry_actor=registry_actor,
        )
        workers.append(worker)

    return workers
