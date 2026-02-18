"""This module contains the Worker class which is responsible for handling the worker operations."""

import gc
import os
import sys
import time
import traceback
from collections.abc import Callable
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
from rapidfireai.utils.distributed_utils import (
    barrier,
    cleanup_distributed,
    is_distributed_initialized,
    setup_distributed_environment,
    find_free_port,
)
from rapidfireai.fit.utils.exceptions import WorkerException
from rapidfireai.fit.utils.logging import RFLogger, TrainingLogger
from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger
from rapidfireai.fit.utils.serialize import decode_db_payload
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager
from rapidfireai.fit.utils.trainer_config import TrainerConfig


def _release_model_gpu_memory(model, logger=None):
    """Release GPU tensor storage from a model to free CUDA memory.
    """
    if model is None:
        return
    _empty = torch.empty(0, device="cpu")
    freed_bytes = 0
    handles_found = 0
    params_found = 0

    # ---- 1. FSDP flat-parameter handles (hold extra refs to GPU tensors) ----
    try:
        for module in model.modules():
            handle = getattr(module, "_handle", None)
            if handle is not None:
                handles_found += 1
                fp = getattr(handle, "flat_param", None)
                if fp is not None and hasattr(fp, "data") and fp.data.is_cuda:
                    freed_bytes += fp.data.nelement() * fp.data.element_size()
                    fp.data = _empty
                    fp.grad = None
    except Exception as e:
        if logger:
            logger.debug(f"[GPU cleanup] Error in FSDP handle scan: {e}")

    # ---- 2. All model parameters ----
    try:
        for param in model.parameters():
            params_found += 1
            if param.data.is_cuda:
                freed_bytes += param.data.nelement() * param.data.element_size()
                param.data = _empty
            if param.grad is not None:
                param.grad = None
    except Exception as e:
        if logger:
            logger.debug(f"[GPU cleanup] Error in param scan: {e}")

    # ---- 3. All model buffers ----
    try:
        for buf in model.buffers():
            if buf.is_cuda:
                freed_bytes += buf.nelement() * buf.element_size()
                buf.data = _empty
    except Exception as e:
        if logger:
            logger.debug(f"[GPU cleanup] Error in buffer scan: {e}")

    if logger:
        logger.debug(
            f"[GPU cleanup] model scan: {handles_found} FSDP handles, "
            f"{params_found} params, released {freed_bytes / (1024**3):.2f} GiB"
        )


def _scan_and_release_leaked_cuda_tensors(logger=None):
    """Nuclear fallback: scan ALL gc-tracked objects for CUDA tensors and zero them.

    This catches tensors held by FSDP C++ internals, callback handlers,
    accelerator caches, or any other reference that model-level iteration
    cannot reach.  Safe to call between training runs when no CUDA tensors
    should be alive.
    """
    gc.collect()
    gc.collect()  # second pass catches circular refs

    freed_bytes = 0
    tensor_count = 0
    _empty = torch.empty(0, device="cpu")
    leaked_summary = {}

    for obj in gc.get_objects():
        if not torch.is_tensor(obj):
            continue
        try:
            if obj.is_cuda:
                size = obj.nelement() * obj.element_size()
                freed_bytes += size
                tensor_count += 1
                key = (tuple(obj.shape), str(obj.dtype))
                if key not in leaked_summary:
                    leaked_summary[key] = {"count": 0, "total_bytes": 0}
                leaked_summary[key]["count"] += 1
                leaked_summary[key]["total_bytes"] += size
                obj.data = _empty
        except Exception:
            pass

    if logger:
        logger.debug(
            f"[GPU cleanup] gc scan: zeroed {tensor_count} leaked CUDA tensors, "
            f"{freed_bytes / (1024**3):.2f} GiB"
        )
        if leaked_summary:
            top = sorted(
                leaked_summary.items(),
                key=lambda x: x[1]["total_bytes"],
                reverse=True,
            )[:10]
            for (shape, dtype), info in top:
                logger.debug(
                    f"  shape={shape}, dtype={dtype}: "
                    f"{info['count']}x, {info['total_bytes'] / (1024**2):.1f} MiB"
                )

    return freed_bytes


class TeeOutput:
    """A file-like object that captures output to a buffer while providing fileno() for vLLM compatibility.
    Output is NOT written to the original stream to prevent it from appearing in the notebook.
    """

    def __init__(self, original_stream, buffer):
        self.original_stream = original_stream
        self.buffer = buffer

    def write(self, text):
        """Write only to the buffer, not to the original stream (to suppress notebook output)."""
        self.buffer.write(text)
        # Don't write to original_stream to prevent output in notebook

    def flush(self):
        """Flush the buffer only."""
        self.buffer.flush()
        # Don't flush original_stream to prevent output in notebook

    def fileno(self):
        """Return the original stream's file descriptor (required by vLLM)."""
        return self.original_stream.fileno()

    def isatty(self):
        """Check if the original stream is a TTY."""
        return self.original_stream.isatty()

    def getvalue(self):
        """Get the captured buffer content."""
        return self.buffer.getvalue()

    def __getattr__(self, name):
        """Delegate other attributes to the original stream."""
        return getattr(self.original_stream, name)


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
        self.training_logger: Logger = TrainingLogger().create_logger(
            f"worker_{worker_id}"
        )
        self.logger.debug(f"Worker {self.worker_id} initialized with PID {os.getpid()}")

        # create database object
        self.db: RfDb = RfDb()

        # get experiment name
        self.experiment_name: str = self.db.get_running_experiment()["experiment_name"]

        # initialize data paths
        DataPath.initialize(
            self.experiment_name, self.db.get_experiments_path(self.experiment_name)
        )

        # create metric logger
        default_metric_loggers = RFMetricLogger.get_default_metric_loggers(
            experiment_name=self.experiment_name
        )
        self.metric_logger = RFMetricLogger(default_metric_loggers, logger=self.logger)
        if self.metric_logger is None:
            raise WorkerException(
                "MetricLogger is not initialized. Please check the metric logger configuration."
            )
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
        multi_worker_details: dict[str, Any],
        create_model_fn: Callable,
    ) -> None:
        """Run fit"""
        self.logger.debug(
            f"Received run_fit on worker for run {run_id} with chunk {chunk_id}"
        )

        # get run details
        run_details = self.db.get_run(run_id)
        config_leaf = run_details["config_leaf"]
        metric_run_id = run_details["metric_run_id"]

        # check if FSDP is enabled
        use_fsdp = (
            "training_args" in config_leaf
            and "fsdp_config" in config_leaf["training_args"]
        )
        # Clean up any stale vLLM parallel state before initializing distributed environment
        # This is critical when training multiple models sequentially with GRPO
        if config_leaf.get("trainer_type", "SFT") == "GRPO":
            try:
                from vllm.distributed import parallel_state

                if hasattr(parallel_state, "destroy_model_parallel"):
                    try:
                        parallel_state.destroy_model_parallel()
                    except Exception:
                        pass  # Ignore if not initialized
                if hasattr(parallel_state, "destroy_distributed_environment"):
                    try:
                        parallel_state.destroy_distributed_environment()
                    except Exception:
                        pass  # Ignore if not initialized
            except (ImportError, AttributeError):
                pass  # vLLM parallel state cleanup not available

        # Initialize distributed training if FSDP is enabled for this run
        if use_fsdp:
            try:
                os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                # Get distributed configuration from run details
                master_addr = multi_worker_details["master_address"]
                master_port = multi_worker_details["master_port"]
                world_size = multi_worker_details["world_size"]
                rank = multi_worker_details["local_rank"]
                self.logger.debug(
                    f"Worker {self.worker_id} initializing distributed training for run {run_id} with master_addr {master_addr}, master_port {master_port}, world_size {world_size}, rank {rank}"
                )
                os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                    map(str, multi_worker_details["worker_ids"])
                )
                setup_distributed_environment(
                    rank=rank,
                    world_size=world_size,
                    master_addr=master_addr,
                    master_port=master_port,
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

        # get effective batch size
        per_device_train_batch_size = config_leaf["training_args"].get(
            "per_device_train_batch_size", 1
        )
        gradient_accumulation_steps = config_leaf["training_args"].get(
            "gradient_accumulation_steps", 1
        )
        effective_batch_size = (
            per_device_train_batch_size
            * gradient_accumulation_steps
            * multi_worker_details.get("world_size", 1)
        )

        # fetch train dataset chunk
        train_dataset_chunker = DatasetChunks(
            self.len_train_dataset,
            self.num_chunks,
            batch_size=effective_batch_size,
            offset=run_details["chunk_offset"],
        )
        train_dataset_chunk = train_dataset_chunker.get_chunk(
            self.train_dataset, chunk_id
        )

        # create worker config
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
        # add reward funcs to config_leaf if cloned from a GRPO run
        if (
            trainer_config.cloned_from is not None
            and trainer_config.config_leaf.get("trainer_type") == "GRPO"
        ):
            parent_run_details = self.db.get_run(trainer_config.cloned_from)
            config_leaf["reward_funcs"] = parent_run_details["config_leaf"].get(
                "reward_funcs"
            )
            self.db.set_run_details(run_id, config_leaf=config_leaf)

        if use_fsdp and is_distributed_initialized():
            barrier()

        # create trainer instance and write logs to user logger
        # Redirect stdout/stderr to /dev/null at OS level to suppress all output (Python and C-level)
        # Then use TeeOutput to capture Python-level output for logging while maintaining fileno() for vLLM
        stdout_buffer = StringIO()
        stderr_buffer = StringIO()
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        # Redirect to /dev/null to suppress all output from appearing in notebook
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

        # Create TeeOutput wrappers that capture to buffer but don't write to original stream
        # fileno() will return the /dev/null fd which is fine for vLLM
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
            )

            # if first time, save checkpoint to disk
            completed_steps = self.db.get_completed_steps(run_id)
            if completed_steps == 0 and not USE_SHARED_MEMORY:
                save_checkpoint_to_disk(
                    trainer_instance, trainer_config, first=True, use_fsdp=use_fsdp
                )

            # update base model name in db for run
            trainer_config.config_leaf["model_name"] = (
                trainer_instance.model.config._name_or_path
            )
            self.db.set_run_details(run_id, config_leaf=trainer_config.config_leaf)

            # train the model and time it
            if torch.cuda.is_available():
                mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                mem_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
                self.logger.debug(
                    f"Beginning training for run {run_id} on chunk {chunk_id} | "
                    f"GPU mem: {mem_alloc:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved, "
                    f"{mem_total - mem_alloc:.2f} GiB free (of {mem_total:.2f} GiB total)"
                )
            else:
                self.logger.debug(
                    f"Beginning training for run {run_id} on chunk {chunk_id}"
                )

            # Synchronize all workers before training starts
            if use_fsdp and is_distributed_initialized():
                barrier()

            start_time = time.time()
            trainer_instance.train()
            end_time = time.time()
            # Synchronize all workers after training completes
            if use_fsdp and is_distributed_initialized():
                barrier()

            # update completed steps
            new_completed_steps = completed_steps + trainer_instance.state.global_step
            self.db.set_completed_steps(run_id, new_completed_steps)

            # update running average runtime in database for scheduler optimization
            if trainer_config.local_rank == 0:
                new_runtime_per_batch = (
                    end_time - start_time
                ) / train_dataset_chunker.get_chunk_size(chunk_id)
                running_average_runtime = (
                    run_details["estimated_runtime"] * completed_steps
                    + new_runtime_per_batch
                ) / new_completed_steps
                self.db.set_estimated_runtime(run_id, running_average_runtime)

            # save checkpoints
            if USE_SHARED_MEMORY:
                if use_fsdp and is_distributed_initialized():
                    barrier()
                    self.logger.debug(
                        f"Worker {self.worker_id} passed barrier after training"
                    )

                # ---- Aggressive GPU cleanup before checkpoint save ----
                # The checkpoint save uses FSDP FULL_STATE_DICT all-gather
                # which needs contiguous GPU memory.  After eval/generation
                # passes, CUDA allocator fragmentation and leaked NCCL
                # buffers can reduce actual usable memory even when
                # memory_allocated() looks fine.  Clean everything we can.
                if torch.cuda.is_available():
                    # 1. Zero optimizer gradients (set_to_none frees storage)
                    try:
                        if hasattr(trainer_instance, "optimizer") and trainer_instance.optimizer is not None:
                            trainer_instance.optimizer.zero_grad(set_to_none=True)
                    except Exception:
                        pass

                    # 2. Clear optimizer state tensors (exp_avg, exp_avg_sq)
                    #    that might still live on GPU despite FSDP CPU offload.
                    #    We're about to re-collect the optimizer state via
                    #    FSDP.optim_state_dict() in the checkpoint save, so
                    #    these cached copies are expendable.
                    try:
                        if hasattr(trainer_instance, "optimizer") and trainer_instance.optimizer is not None:
                            for param_group in trainer_instance.optimizer.param_groups:
                                for p in param_group["params"]:
                                    if p in trainer_instance.optimizer.state:
                                        state = trainer_instance.optimizer.state[p]
                                        for key in list(state.keys()):
                                            if torch.is_tensor(state[key]) and state[key].is_cuda:
                                                state[key] = state[key].to("cpu")
                    except Exception:
                        pass

                    # 3. Purge model-level caches left by eval generation
                    #    (KV caches, past_key_values, etc.)
                    try:
                        model = trainer_instance.model
                        for attr in ("past_key_values", "_past", "key_cache",
                                     "value_cache"):
                            if hasattr(model, attr):
                                setattr(model, attr, None)
                        for module in model.modules():
                            for attr in ("past_key_values", "_past", "key_cache",
                                         "value_cache", "_seen_tokens"):
                                if hasattr(module, attr):
                                    try:
                                        setattr(module, attr, None)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # 4. Release trainer caches that aren't needed for
                    #    checkpoint saving (callback handler, data collator
                    #    cached tensors, memory tracker).
                    try:
                        if hasattr(trainer_instance, "_memory_tracker"):
                            trainer_instance._memory_tracker = None
                    except Exception:
                        pass

                    # 5. Two-pass gc to break circular refs, then sync +
                    #    empty_cache so CUDA returns all freed blocks.
                    gc.collect()
                    gc.collect()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    # 6. Reset allocator stats for cleaner decision-making.
                    try:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                    except Exception:
                        pass

                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    mem_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
                    self.logger.debug(
                        f"Pre-checkpoint memory freed | "
                        f"GPU mem: {mem_alloc:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved, "
                        f"{mem_total - mem_alloc:.2f} GiB free (of {mem_total:.2f} GiB total)"
                    )

                is_run_finished = (
                    chunk_id == self.num_chunks - 1
                    and new_completed_steps >= trainer_config.total_steps
                )

                if is_run_finished:
                    # Run finished all steps — save the final checkpoint
                    # directly to disk and skip shared memory.
                    # save_checkpoint_to_shared_memory zeroes flat_params for
                    # the optimizer all-gather, which makes any subsequent
                    # FSDP state_dict calls fail.  By going straight to disk
                    # we avoid that corruption.
                    save_checkpoint_to_disk(
                        trainer_instance,
                        trainer_config,
                        last=True,
                        use_fsdp=use_fsdp,
                    )
                    self.logger.debug(
                        f"Saved final checkpoint to disk for run {run_id} on chunk {chunk_id}"
                    )
                else:
                    # Normal (non-final) chunk: save to shared memory
                    save_checkpoint_to_shared_memory(
                        trainer_instance,
                        trainer_config,
                        self.shm_manager,
                        use_fsdp=use_fsdp,
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
                        f"Saved checkpoint to shared memory for run {run_id} on chunk {chunk_id}",
                        f"and worker {self.worker_id}",
                    )

                    if use_fsdp and is_distributed_initialized():
                        barrier()

                    # save checkpoint to disk based on save strategy
                    save_strategy = trainer_config.config_leaf.get("training_args", {}).get(
                        "save_strategy", "epoch"
                    )
                    if save_strategy == "chunk":
                        save_checkpoint_to_disk(
                            trainer_instance,
                            trainer_config,
                            completed_steps=new_completed_steps,
                            use_fsdp=use_fsdp,
                        )
                        self.logger.debug(
                            f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}"
                        )
            else:
                # save checkpoint to disk when not using shared memory
                save_checkpoint_to_disk(
                    trainer_instance,
                    trainer_config,
                    completed_steps=new_completed_steps,
                    use_fsdp=use_fsdp,
                )
                self.logger.debug(
                    f"Saved checkpoint to disk for run {run_id} on chunk {chunk_id}"
                )

            # save final checkpoint (only for non-shared-memory path;
            # the USE_SHARED_MEMORY path handles this above)
            if (
                not USE_SHARED_MEMORY
                and chunk_id == self.num_chunks - 1
                and new_completed_steps >= trainer_config.total_steps
            ):
                save_checkpoint_to_disk(
                    trainer_instance, trainer_config, last=True, use_fsdp=use_fsdp
                )
                self.logger.debug(
                    f"Saved final checkpoint for run {run_id} on chunk {chunk_id}"
                )

            if use_fsdp and is_distributed_initialized():
                barrier()

        except Exception as e:
            self.logger.error(f"Error during training for run {run_id}: {e}")
            raise
        finally:
            # CRITICAL: Cleanup must happen even when training fails to prevent memory leaks
            try:
                # Clean up vLLM engine before deleting trainer to avoid stale process group references
                # This is critical for GRPO with FSDP when training multiple models sequentially
                if "trainer_instance" in locals() and hasattr(trainer_instance, "llm"):
                    try:
                        # Shutdown vLLM engine to clean up its process groups
                        if hasattr(trainer_instance.llm, "shutdown"):
                            trainer_instance.llm.shutdown()
                        elif hasattr(trainer_instance.llm, "llm_engine"):
                            # Try to clean up the engine core if shutdown method doesn't exist
                            if hasattr(trainer_instance.llm.llm_engine, "engine_core"):
                                if hasattr(
                                    trainer_instance.llm.llm_engine.engine_core, "shutdown"
                                ):
                                    trainer_instance.llm.llm_engine.engine_core.shutdown()

                        # Clean up vLLM's parallel state if it exists
                        try:
                            from vllm.distributed import parallel_state

                            if hasattr(parallel_state, "destroy_model_parallel"):
                                parallel_state.destroy_model_parallel()
                            if hasattr(parallel_state, "destroy_distributed_environment"):
                                parallel_state.destroy_distributed_environment()
                        except (ImportError, AttributeError):
                            pass  # vLLM parallel state cleanup not available

                        # Delete the llm instance explicitly
                        del trainer_instance.llm
                    except Exception as cleanup_e:
                        self.logger.debug(f"Error cleaning up vLLM engine: {cleanup_e}")
                        # Force delete even if shutdown fails
                        if hasattr(trainer_instance, "llm"):
                            try:
                                del trainer_instance.llm
                            except Exception:
                                pass

                # ---- Systematic GPU memory cleanup ----
                if "trainer_instance" in locals():
                    # Step 1: Neutralize all internal references that point TO the model.
                    # The HF Trainer's callback_handler, optimizer, and accelerator
                    # all hold separate refs to the model that keep it alive.
                    try:
                        if hasattr(trainer_instance, "callback_handler"):
                            trainer_instance.callback_handler.model = None
                            trainer_instance.callback_handler.optimizer = None
                            trainer_instance.callback_handler.lr_scheduler = None
                    except Exception:
                        pass

                    # Step 2: Clear optimizer state AND param_groups (param_groups
                    # hold direct refs to model parameter tensors).
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

                    # Step 3: Release GPU tensor storage via model-level iteration.
                    for attr in ("model", "model_wrapped", "ref_model"):
                        try:
                            _release_model_gpu_memory(
                                getattr(trainer_instance, attr, None),
                                logger=self.logger,
                            )
                        except Exception:
                            pass

                    # Step 4: Delete all trainer attributes in dependency order.
                    try:
                        if hasattr(trainer_instance, "optimizer"):
                            del trainer_instance.optimizer
                    except Exception:
                        pass
                    try:
                        if hasattr(trainer_instance, "lr_scheduler"):
                            del trainer_instance.lr_scheduler
                    except Exception:
                        pass
                    try:
                        if hasattr(trainer_instance, "model_wrapped"):
                            del trainer_instance.model_wrapped
                    except Exception:
                        pass
                    try:
                        if hasattr(trainer_instance, "model"):
                            del trainer_instance.model
                    except Exception:
                        pass
                    try:
                        if hasattr(trainer_instance, "ref_model"):
                            del trainer_instance.ref_model
                    except Exception:
                        pass

                    # Step 5: Free accelerator (holds wrapped model + optimizer refs).
                    try:
                        if hasattr(trainer_instance, "accelerator"):
                            if hasattr(trainer_instance.accelerator, "free_memory"):
                                trainer_instance.accelerator.free_memory()
                            del trainer_instance.accelerator
                    except Exception:
                        pass

                    # Step 6: Clear remaining trainer attrs and kill the object.
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

                # Step 7: Two-pass gc.collect() (second pass catches circular refs).
                gc.collect()
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

                    # Step 8: Nuclear fallback — scan ALL gc-tracked objects for
                    # leaked CUDA tensors that model iteration couldn't reach
                    # (FSDP C++ handles, lingering autograd graph nodes, etc.).
                    _scan_and_release_leaked_cuda_tensors(logger=self.logger)

                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

                    try:
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                    except Exception:
                        pass

                if use_fsdp and is_distributed_initialized():
                    barrier()
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    mem_reserved = torch.cuda.memory_reserved() / (1024**3)
                    mem_total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024**3)
                    self.logger.debug(
                        f"Cleanup completed for run {run_id} on chunk {chunk_id} | "
                        f"GPU mem: {mem_alloc:.2f} GiB allocated, {mem_reserved:.2f} GiB reserved, "
                        f"{mem_total - mem_alloc:.2f} GiB free (of {mem_total:.2f} GiB total)"
                    )
                else:
                    self.logger.debug(
                        f"Cleanup completed for run {run_id} on chunk {chunk_id}"
                    )
            except Exception as cleanup_error:
                self.logger.error(f"Error during cleanup for run {run_id}: {cleanup_error}")
            finally:
                # Restore original stdout/stderr
                sys.stdout = original_stdout
                sys.stderr = original_stderr

        # Write captured output to training logger
        if stdout_buffer.getvalue():
            self.training_logger.info(stdout_buffer.getvalue())
        if stderr_buffer.getvalue():
            self.training_logger.error(stderr_buffer.getvalue())

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
                    self.db.set_worker_task_status(
                        self.worker_id, TaskStatus.IN_PROGRESS
                    )

                    # run train and validation function
                    try:
                        self.run_fit(
                            run_id, chunk_id, multi_worker_details, create_model_fn
                        )
                        self.db.set_worker_task_status(
                            self.worker_id, TaskStatus.COMPLETED
                        )
                    except Exception as e:
                        self.logger.opt(exception=True).error(
                            f"Error while running run_fit for run {run_id} and chunk {chunk_id}: {e}"
                        )
                        self.db.set_run_details(
                            run_id,
                            status=RunStatus.FAILED,
                            error=str(e) + traceback.format_exc(),
                        )
                        self.db.set_worker_task_status(
                            self.worker_id, TaskStatus.FAILED
                        )
                else:
                    raise WorkerException(f"Invalid task type: {task_type}")
                if is_distributed_initialized():
                    cleanup_distributed()
                    self.logger.debug(
                        f"Worker {self.worker_id} distributed training cleaned up"
                    )
            except Exception as e:
                self.logger.opt(exception=True).error(
                    f"Worker {self.worker_id} error: {e}"
                )
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
                self.logger.debug(
                    f"Worker {self.worker_id} distributed training cleaned up"
                )
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
