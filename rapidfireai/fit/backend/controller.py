"""This module contains the Controller class which is responsible for orchestrating the RapidFire lifecycle."""

import math
import random
import time
from collections.abc import Callable
from logging import Logger
from pprint import pformat
from typing import Any

import ray
from torch.utils.data import Dataset

from rapidfireai.automl import AutoMLAlgorithm, get_flattened_config_leaf, get_runs
from rapidfireai.db import RfDb
from rapidfireai.fit.backend.chunks import DatasetChunks
from rapidfireai.fit.backend.scheduler import Scheduler
from rapidfireai.fit.backend.worker_actor import create_worker_actors
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager
from rapidfireai.utils.constants import (
    ControllerTask,
    ExperimentTask,
    ICOperation,
    ICStatus,
    MLFlowConfig,
    RunEndedBy,
    RunSource,
    RunStatus,
    TaskStatus,
    WorkerTask,
)
from rapidfireai.utils.exceptions import ControllerException
from rapidfireai.utils.logging import RFLogger
from rapidfireai.metrics import RFMetricLogger
from rapidfireai.utils.os_utils import mkdir_p
from rapidfireai.utils.serialize import encode_payload


class Controller:
    """This module contains the ML Controller class which is responsible for orchestrating the RapidFire lifecycle."""

    def __init__(
        self,
        experiment_id: int,
        experiment_name: str,
        num_workers: int = 1,
        gpus_per_worker: int = 1,
        cpus_per_worker: int = 1,
    ) -> None:
        """
        Initialize the controller.

        Args:
            experiment_id: ID of the experiment
            experiment_name: Name of the experiment
            num_workers: Number of worker actors to create (typically = num GPUs)
            gpus_per_worker: Number of GPUs per worker (typically 1)
            cpus_per_worker: Number of CPUs per worker
        """
        self.experiment_id: int = experiment_id
        self.experiment_name: str = experiment_name
        self.num_workers: int = num_workers
        self.gpus_per_worker: int = gpus_per_worker
        self.cpus_per_worker: int = cpus_per_worker

        # Create database object
        self.db: RfDb = RfDb()

        # Initialize data paths
        DataPath.initialize(experiment_name, self.db.get_experiments_path(experiment_id))

        # Create controller logger
        logging = RFLogger(experiment_name=experiment_name)
        self.logger: Logger = logging.get_logger("controller")
        self.user_logger: Logger = logging.get_logger("user")
        self.ic_logger: Logger = logging.get_logger("interactive-control")

        # Validate GPU availability
        if self.num_workers == 0 or (self.gpus_per_worker == 0 and self.num_workers > 0):
            self.logger.warning("No GPUs available. Training will run on CPU (not recommended).")
        else:
            self.logger.debug(f"Configured for {self.num_workers} workers with {self.gpus_per_worker} GPU(s) each.")

        # Initialize shared memory manager (used for model/checkpoint sharing between workers)
        # SharedMemoryManager creates a RegistryActor internally for coordination
        self.shm_manager: SharedMemoryManager = SharedMemoryManager(name="controller-shm")

        # Worker actors (created in run_fit)
        self.worker_actors: list = []

        default_metric_loggers = RFMetricLogger.get_default_metric_loggers(experiment_name=self.experiment_name)
        self.metric_logger = RFMetricLogger(
            default_metric_loggers,
            logger=self.logger,
        )
        if self.metric_logger is None:
            raise ControllerException("MetricLogger is not initialized. Please check the metric logger configuration.")
        self.metric_logger.get_experiment(self.experiment_name)
        self.logger.debug("Controller initialized")

    def _create_models(
        self,
        param_config: AutoMLAlgorithm | dict[str, Any],
        source: RunSource,
        seed: int,
        len_train_dataset: int,
        num_chunks: int,
        clone_modify_info: dict[str, Any] | None = None,
    ) -> list[int]:
        """Create the models."""

        # get config_leaf from param_config for each run
        config_leafs = get_runs(param_config, seed)

        # create runs
        runs = {}
        for config_leaf in config_leafs:
            flattened_config = get_flattened_config_leaf(config_leaf)
            total_steps = self._get_total_step(config_leaf, len_train_dataset, num_chunks)

            # get clone modify info
            warm_started_from = clone_modify_info.get("warm_started_from") if clone_modify_info else None
            cloned_from = clone_modify_info.get("cloned_from") if clone_modify_info else None
            chunk_offset = clone_modify_info.get("chunk_offset", 0) if clone_modify_info else 0

            run_id = self.db.create_run(
                config_leaf=config_leaf,
                status=RunStatus.NEW,
                completed_steps=0,
                total_steps=total_steps,
                error="",
                source=source,
                ended_by=None,
                chunk_offset=chunk_offset,
                warm_started_from=warm_started_from,
                cloned_from=cloned_from,
            )
            runs[run_id] = flattened_config

            # create directories for each run
            try:
                base_run_path = DataPath.base_run_path(run_id)
                work_dir_path = DataPath.work_dir_path(base_run_path)
                initial_checkpoint_path = DataPath.initial_checkpoint_path(base_run_path)
                final_checkpoint_path = DataPath.final_checkpoint_path(base_run_path)
                intermediate_checkpoint_path = DataPath.intermediate_checkpoint_path(base_run_path)

                mkdir_p(work_dir_path, notify=False)
                mkdir_p(initial_checkpoint_path, notify=False)
                mkdir_p(final_checkpoint_path, notify=False)
                mkdir_p(intermediate_checkpoint_path, notify=False)
            except (PermissionError, OSError) as e:
                raise ControllerException(f"Failed to create required Run DataPath directories: {e}") from e

            # create new tracking run
            metric_run_id = None
            try:
                # create new tracking run and get the metric_run_id
                metric_run_id = self.metric_logger.create_run(str(run_id))
                # populate tracking backend with model config info
                for key, value in flattened_config.items():
                    self.metric_logger.log_param(metric_run_id, key, value)
                if warm_started_from:
                    self.metric_logger.log_param(metric_run_id, "warm-start", str(warm_started_from))
                if cloned_from:
                    self.metric_logger.log_param(metric_run_id, "parent-run", str(cloned_from))
                self.logger.debug(f"Populated MetricLogger with model config info for run {run_id}.")
                self.db.set_run_details(
                    run_id=run_id,
                    metric_run_id=metric_run_id,
                    flattened_config=flattened_config,
                )
            except Exception as e:
                # Catch any metric logger exceptions (MLflow, TensorBoard, etc.)
                msg = f"Error creating new tracking run for run {run_id} - {e}."
                print(msg)
                if metric_run_id:
                    try:
                        self.metric_logger.end_run(metric_run_id)
                    except Exception:
                        pass
                self.logger.error(msg, exc_info=True)

        total_runs = len(runs)
        self.logger.info(f"Created {total_runs} runs - \n{pformat(runs, indent=4, width=120)}")
        self.logger.debug(f"Got {total_runs} runs for {source.value}.")

        # set experiment task to run_fit
        self.db.set_experiment_current_task(ExperimentTask.RUN_FIT)
        self.logger.debug("Completed creating models.")

        return list(runs.keys())

    def _clear_run_from_shm(self, run_id: int) -> None:
        """Clear the run from shared memory."""

        # check if there are any other runs with the same base model
        base_model_name = self.db.get_run(run_id)["config_leaf"]["model_name"]
        relevant_runs = self.db.get_runs_by_status([RunStatus.ONGOING, RunStatus.NEW, RunStatus.STOPPED])

        # get shared object types to delete - if no other runs are using it
        delete_shared_objects = True
        for r_run_id, r_run_details in relevant_runs.items():
            if r_run_details["config_leaf"]["model_name"] == base_model_name and r_run_id != run_id:
                delete_shared_objects = False
                break

        # delete model object from shared memory
        self.shm_manager.delete_model_object(run_id, base_model_name if delete_shared_objects else None)

    def _process_interactive_control(
        self,
        run_states: dict[str, Any],
        clone_modify_tasks: list[dict[str, Any]],
        len_train_dataset: int,
        seed: int,
        num_chunks: int,
    ) -> None:
        """Process interactive control tasks."""

        # process non-clone_modify tasks
        for run_id, run_state in run_states.items():
            if not run_state["task_id"]:
                continue

            if run_state["status"] == RunStatus.STOPPED:
                # process stopped tasks
                # mark run as stopped
                self.db.set_run_details(
                    run_id=run_id,
                    status=RunStatus.STOPPED,
                    ended_by=RunEndedBy.INTERACTIVE_CONTROL,
                )
                self.db.update_ic_operation_status(run_state["ic_id"], ICStatus.COMPLETED)
                self.ic_logger.info(f"Stopping run {run_id} by Interactive Control")
            elif run_state["status"] == RunStatus.DELETED:
                # process deleted tasks
                # clear run from shm
                # TODO: commented out to prevent clone of deleted runs issue (see Issue # 22)
                # self._clear_run_from_shm(run_id)

                # delete run from MetricLogger
                metric_run_id = self.db.get_run(run_id)["metric_run_id"]
                self.metric_logger.delete_run(metric_run_id)
                # mark run as deleted
                self.db.set_run_details(
                    run_id=run_id,
                    status=RunStatus.DELETED,
                    ended_by=RunEndedBy.INTERACTIVE_CONTROL,
                )
                self.db.update_ic_operation_status(run_state["ic_id"], ICStatus.COMPLETED)
                self.ic_logger.info(f"Deleting run {run_id} by Interactive Control")
            elif run_state["status"] == RunStatus.ONGOING:
                # process ongoing tasks
                self.db.set_run_details(
                    run_id=run_id,
                    status=RunStatus.ONGOING,
                    ended_by="",
                )
                self.db.update_ic_operation_status(run_state["ic_id"], ICStatus.COMPLETED)
                self.ic_logger.info(f"Resuming run {run_id} by Interactive Control")
            elif run_state["status"] == RunStatus.COMPLETED:
                # process completed tasks
                self.logger.warning(f"Run {run_id} is already completed. Skipping Interactive Control task.")
                self.db.update_ic_operation_status(run_state["ic_id"], ICStatus.SKIPPED)
            else:
                raise ValueError(f"Unsupported run status {run_state['status']}")

        # process clone_modify tasks from the collected list
        for task in clone_modify_tasks:
            parent_run_id = task["target_id"]
            operation = task["operation"]
            config_leaf = task["config_data"]

            # add additional_kwargs to config_leaf if it exists in the parent run
            parent_run_details = self.db.get_run(parent_run_id)
            if "additional_kwargs" in parent_run_details["config_leaf"]:
                config_leaf["additional_kwargs"] = parent_run_details["config_leaf"]["additional_kwargs"]

            # create model for the new run
            try:
                if operation == ICOperation.CLONE.value:
                    clone_modify_info = {
                        "cloned_from": parent_run_id,
                    }
                    run_ids = self._create_models(
                        config_leaf,
                        RunSource.INTERACTIVE_CONTROL,
                        seed,
                        len_train_dataset,
                        num_chunks=num_chunks,
                        clone_modify_info=clone_modify_info,
                    )
                elif operation == ICOperation.CLONE_WARM.value:
                    # calculate clone chunk offset
                    effective_batch_size = parent_run_details["config_leaf"]["training_args"].get(
                        "per_device_train_batch_size", 1
                    ) * parent_run_details["config_leaf"]["training_args"].get("gradient_accumulation_steps", 1)
                    chunker = DatasetChunks(
                        len_train_dataset,
                        num_chunks,
                        batch_size=effective_batch_size,
                        offset=parent_run_details["chunk_offset"],
                    )
                    # Convert count to chunk_id by subtracting 1, with edge case handling for 0 chunks visited
                    num_chunks_visited = parent_run_details["num_chunks_visited_curr_epoch"]
                    if num_chunks_visited == 0:
                        # No chunks visited yet - warm-clone behaves like cold clone (start from beginning)
                        clone_chunk_offset = 0
                    else:
                        last_completed_chunk_id = num_chunks_visited - 1
                        clone_chunk_offset = chunker.get_clone_offset(last_completed_chunk_id)
                    clone_modify_info = {
                        "cloned_from": parent_run_id,
                        "warm_started_from": parent_run_id,
                        "chunk_offset": clone_chunk_offset,
                    }
                    run_ids = self._create_models(
                        config_leaf,
                        RunSource.INTERACTIVE_CONTROL,
                        seed,
                        len_train_dataset,
                        num_chunks,
                        clone_modify_info,
                    )
                else:
                    raise ValueError(f"Unsupported IC operation {operation}")

                # mark task as completed
                self.db.update_ic_operation_status(task["ic_id"], ICStatus.COMPLETED)
                self.ic_logger.info(
                    f"Cloned run {parent_run_id} by Interactive Control with {operation} into runs - {run_ids}"
                )
            except Exception as e:
                self.db.update_ic_operation_status(task["ic_id"], ICStatus.FAILED)
                self.ic_logger.error(f"Error creating model for run {parent_run_id}: {e}")
                raise ControllerException(f"Error creating model for run {parent_run_id}: {e}") from e

    def _process_interm_ic_ops_states(
        self,
        currently_scheduled_runs: list[int],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Process the interactive control."""
        # get IC Ops scheduled tasks
        ic_scheduled_tasks = self.db.get_pending_ic_operations()

        # track states for each task(run) and collect clone_modify tasks separately
        run_states = {}
        clone_modify_tasks = []
        for task in ic_scheduled_tasks:
            run_id = task["target_id"]  # target_id is the run_id for run-targeted operations

            # skip if run is currently scheduled (we process IC ops only at chunk boundaries)
            if run_id in currently_scheduled_runs:
                # self.logger.debug(f"Skipping IC op for run {run_id} as it is currently scheduled")
                continue

            is_clone_modify_task = task["operation"] in (
                ICOperation.CLONE.value,
                ICOperation.CLONE_WARM.value,
            )

            if is_clone_modify_task:
                # clone_modify tasks
                # get latest run state
                run_status = run_states[run_id]["status"] if run_id in run_states else self.db.get_run(run_id)["status"]

                # track clone_modify tasks only for non-deleted runs
                if run_status != RunStatus.DELETED:
                    clone_modify_tasks.append(task)
                    self.ic_logger.info(f"Added {task['operation']} task for run {run_id}.")
                else:
                    self.db.update_ic_operation_status(task["ic_id"], ICStatus.SKIPPED)
                    self.ic_logger.warning(f"Skipping {task['operation']} task for deleted run {run_id}.")
            else:
                # Non clone_modify tasks
                if run_id not in run_states:
                    run_states[run_id] = {
                        "ic_id": None,
                        "task": None,
                        "status": self.db.get_run(run_id)["status"],
                    }

                # update run states based on existing status and task
                current_status = run_states[run_id]["status"]
                if current_status == RunStatus.COMPLETED and task["operation"] in [
                    ICOperation.RESUME.value,
                    ICOperation.STOP.value,
                ]:
                    # ignore RESUME/STOP tasks for completed runs
                    self.ic_logger.warning(f"Ignoring RESUME/STOP task for run {run_id} as it is already completed")
                    self.db.update_ic_operation_status(task["ic_id"], ICStatus.SKIPPED)
                elif current_status == RunStatus.FAILED and task["operation"] != ICOperation.DELETE.value:
                    # ignore all tasks except DELETE for failed runs
                    self.ic_logger.warning(f"Ignoring task {task['operation']} for failed run {run_id}")
                    self.db.update_ic_operation_status(task["ic_id"], ICStatus.SKIPPED)
                elif current_status == RunStatus.DELETED:
                    # ignore all tasks for deleted runs
                    self.ic_logger.warning(f"Ignoring task {task['operation']} for deleted run {run_id}")
                    self.db.update_ic_operation_status(task["ic_id"], ICStatus.SKIPPED)
                else:
                    # valid ic_op for this run
                    # mark prev task as completed
                    if run_states[run_id]["ic_id"] is not None:
                        self.db.update_ic_operation_status(run_states[run_id]["ic_id"], ICStatus.COMPLETED)

                    # add new task to run states
                    if task["operation"] == ICOperation.STOP.value:
                        updated_status = RunStatus.STOPPED
                        info_msg = f"Received STOP task for run {run_id}"
                    elif task["operation"] == ICOperation.DELETE.value:
                        updated_status = RunStatus.DELETED
                        info_msg = f"Received DELETE task for run {run_id}"
                    elif task["operation"] == ICOperation.RESUME.value:
                        updated_status = RunStatus.ONGOING
                        info_msg = f"Received RESUME task for run {run_id}"
                    else:
                        self.db.update_ic_operation_status(task["ic_id"], ICStatus.FAILED)
                        raise ValueError(f"Unsupported task {task['operation']}")
                    run_states[run_id].update(
                        {
                            "ic_id": task["ic_id"],
                            "task": task["operation"],
                            "status": (updated_status if updated_status else current_status),
                        }
                    )
                    self.ic_logger.info(info_msg)

        return run_states, clone_modify_tasks

    def _get_total_step(self, config_leaf: dict[str, Any], len_train_dataset: int, num_chunks: int) -> int:
        """Get the total number of steps for a run."""
        num_train_epochs = config_leaf["training_args"].get("num_train_epochs", 1)

        total_steps = 0
        # max_steps overrides num_train_epochs
        if config_leaf["training_args"].get("max_steps", None):
            # ceil to nearest chunk multiple
            total_steps = config_leaf["training_args"]["max_steps"]
        elif num_train_epochs:
            per_device_train_batch_size = config_leaf["training_args"].get("per_device_train_batch_size", 1)
            gradient_accumulation_steps = config_leaf["training_args"].get("gradient_accumulation_steps", 1)
            len_dataloader = math.ceil(len_train_dataset / per_device_train_batch_size)
            num_update_steps_per_epoch = max(
                len_dataloader // gradient_accumulation_steps + int(len_dataloader % gradient_accumulation_steps > 0),
                1,
            )
            total_steps = math.ceil(num_train_epochs * num_update_steps_per_epoch)

            if config_leaf.get("trainer_type", "SFT") == "GRPO":
                num_generations = config_leaf["training_args"].get("num_generations", 8)
                total_steps = (num_generations * len_train_dataset * num_train_epochs) // (
                    gradient_accumulation_steps * per_device_train_batch_size
                )
        return total_steps

    def run_fit(
        self,
        param_config: Any,
        create_model_fn: Callable,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        num_chunks: int,
        seed: int = 42,
    ) -> None:
        """Run the fit."""

        # set experiment task to create models
        self.db.set_experiment_current_task(ExperimentTask.CREATE_MODELS)
        self.logger.debug(f"Set experiment task to {ExperimentTask.CREATE_MODELS.value}.")

        # save train and eval dataset objects to a file for workers to load
        try:
            datasets = {
                "train": train_dataset,
                "eval": eval_dataset if eval_dataset else None,
                "num_chunks": num_chunks,
            }
            with open(DataPath.dataset_path(), "w", encoding="utf-8") as f:
                f.write(encode_payload(datasets))
            self.logger.debug(f"Saved datasets to {DataPath.dataset_path()}")
        except Exception as e:
            raise ControllerException(f"Error saving datasets: {e}") from e

        # set seed
        random.seed(seed)
        self.logger.info(f"Set seed to {seed}")

        # create models
        try:
            len_train_dataset = len(train_dataset)
            self._create_models(
                param_config,
                RunSource.INITIAL,
                seed,
                len_train_dataset,
                num_chunks=num_chunks,
            )
            self.logger.debug("Created models.")
        except Exception as e:
            raise ControllerException(f"Error creating models: {e}") from e

        # set experiment task to create models
        self.db.set_experiment_current_task(ExperimentTask.RUN_FIT)
        self.logger.debug(f"Set experiment task to {ExperimentTask.RUN_FIT.value}.")

        # Create Ray worker actors
        try:
            # Pass registry actor handle - workers use this for shared state coordination
            self.worker_actors = create_worker_actors(
                num_workers=self.num_workers,
                gpus_per_worker=self.gpus_per_worker,
                cpus_per_worker=self.cpus_per_worker,
                registry_actor=self.shm_manager.get_registry_actor(),
            )
            # Start all workers' serve_forever loops (fire-and-forget, runs in background)
            for worker in self.worker_actors:
                worker.serve_forever.remote()
            self.logger.info(f"Created {self.num_workers} Ray worker actors")
            self.logger.debug(f"Created {self.num_workers} Ray worker actors.")
        except Exception as e:
            raise ControllerException(f"Error creating Ray worker actors: {e}") from e

        # create scheduler
        run_ids = list(
            self.db.get_runs_by_status(
                [
                    RunStatus.NEW,
                ]
            ).keys()
        )
        scheduler = Scheduler(run_ids, self.num_workers, num_chunks)

        # run fit
        self.logger.info("Starting Training and Validation")
        try:
            all_done = False
            prev_worker_tasks = {}  # Track previous iteration's worker tasks

            while not all_done:
                # check for errors
                exp_error = self.db.get_experiment_error(self.experiment_id)
                if exp_error:
                    print(f"Error in experiment: {exp_error}")
                    self.logger.error(f"Error in experiment: {exp_error}")
                    break

                # get current state (pre IC ops states)
                all_worker_tasks = self.db.get_all_worker_tasks()
                all_run_details = self.db.get_all_runs()

                # Filter and separate fresh completed and failed tasks in a single loop
                completed_tasks = {}
                failed_tasks = []
                for worker_id, worker_task in all_worker_tasks.items():
                    prev_task = prev_worker_tasks.get(worker_id, {})
                    current_task_tuple = (worker_task["task_id"], worker_task["status"])
                    prev_task_tuple = (
                        prev_task.get("task_id"),
                        prev_task.get("status"),
                    )

                    # skip if task is the same as previous iteration (no change in status) or run is not active
                    if current_task_tuple == prev_task_tuple or worker_task["run_id"] not in scheduler.run_ids:
                        continue

                    if worker_task["status"] == TaskStatus.COMPLETED:
                        completed_tasks[worker_id] = worker_task
                    elif worker_task["status"] == TaskStatus.FAILED:
                        failed_tasks.append(worker_task)

                # Process completed tasks first (before scheduling new ones)
                for worker_id, worker_task in completed_tasks.items():
                    run_id = worker_task["run_id"]
                    chunk_id = worker_task["chunk_id"]
                    run_details = all_run_details[run_id]
                    self.logger.debug(f"Completed task: run {run_id}, chunk {chunk_id} on worker {worker_id}")
                    self.logger.info(
                        f"Run {run_id} completed steps - {run_details['completed_steps']}/{run_details['total_steps']}"
                    )

                    # Update scheduler state
                    scheduler.set_completed_task(worker_id)

                    # Update database state and local state using scheduler's state as source of truth
                    new_chunks_visited = scheduler.run_visited_num_chunks[run_id]
                    if new_chunks_visited == num_chunks:
                        num_epochs_completed = run_details["num_epochs_completed"] + 1
                    else:
                        num_epochs_completed = run_details["num_epochs_completed"]
                    self.db.set_run_details(
                        run_id=run_id,
                        num_chunks_visited_curr_epoch=new_chunks_visited,
                        num_epochs_completed=num_epochs_completed,
                    )

                    # Update progress
                    progress_percentage = (
                        (run_details["completed_steps"] / run_details["total_steps"] * 100)
                        if run_details["total_steps"] > 0
                        else 0
                    )
                    self.db.set_controller_progress(run_id, progress_percentage)

                    # Check if run has completed all epochs
                    # completed_steps can go beyond total_steps since we stop only at a chunk boundary
                    if run_details["completed_steps"] >= run_details["total_steps"]:
                        scheduler.remove_run(run_id)
                        self.db.set_run_details(
                            run_id=run_id,
                            status=RunStatus.COMPLETED,
                            ended_by=RunEndedBy.EPOCH_COMPLETED,
                        )
                        self.logger.info(
                            f"Run {run_id} has completed all its epochs - "
                            f"steps {run_details['completed_steps']}/{run_details['total_steps']}"
                        )
                    # Check if run has completed only current epoch (hasn't reached total_steps yet)
                    elif (
                        new_chunks_visited == num_chunks and run_details["completed_steps"] < run_details["total_steps"]
                    ):
                        scheduler.reset_run(run_id)
                        self.db.set_run_details(run_id=run_id, num_chunks_visited_curr_epoch=0)
                        self.logger.info(f"Run {run_id} has completed epoch ({new_chunks_visited}/{num_chunks} chunks)")

                # Check for failed runs and update scheduler, local state, shm
                for worker_task in failed_tasks:
                    run_id = worker_task["run_id"]
                    run_error = all_run_details[run_id]["error"]
                    if run_id in scheduler.run_ids:
                        scheduler.remove_run(run_id)
                        self._clear_run_from_shm(run_id)
                        err_msg = f"Run {run_id} has failed: {run_error}"
                        print(err_msg)
                        self.logger.error(err_msg)
                    self.logger.debug(f"Removed run {run_id} from scheduler")

                # Process interactive control tasks (this fetches latest run states internally)
                try:
                    currently_scheduled_runs = list(scheduler.worker_running_current_run.values())
                    run_states, clone_modify_tasks = self._process_interm_ic_ops_states(currently_scheduled_runs)
                    self._process_interactive_control(
                        run_states,
                        clone_modify_tasks,
                        len_train_dataset,
                        seed,
                        num_chunks,
                    )
                except Exception as e:
                    raise ControllerException(f"Error processing interactive control tasks: {e}") from e

                # fetch latest run states again (post IC ops states)
                all_run_details = self.db.get_all_runs()

                # Update scheduler with active and inactive runs from IC Ops changes
                for run_id, run_details in all_run_details.items():
                    # add active runs to scheduler
                    if run_details["status"] in (RunStatus.ONGOING, RunStatus.NEW) and run_id not in scheduler.run_ids:
                        chunks_visited = all_run_details[run_id]["num_chunks_visited_curr_epoch"]
                        scheduler.add_run(run_id, chunks_visited)
                        self.logger.debug(f"Added run {run_id} to scheduler with {chunks_visited} chunks visited")
                    # remove inactive runs from scheduler
                    elif (
                        run_details["status"] in (RunStatus.STOPPED, RunStatus.DELETED) and run_id in scheduler.run_ids
                    ):
                        scheduler.remove_run(run_id)
                        self.logger.debug(f"Removed run {run_id} from scheduler")

                # Get schedule from scheduler
                schedule = scheduler.schedule()
                run_id = schedule["run_id"]
                worker_id = schedule["worker_id"]
                chunk_id = schedule["chunk_id"]

                # Check termination condition
                if run_id is None and worker_id is None and chunk_id is None:
                    self.logger.info("Scheduler indicates all runs have completed all chunks")
                    all_done = True
                    break

                # Check if no schedule possible
                if run_id == -1 and worker_id == -1 and chunk_id == -1:
                    # self.logger.debug("No schedule possible - all workers busy or no available runs")
                    time.sleep(1)
                    continue

                # Execute Schedule
                # Create worker task
                # self.logger.debug(f"Scheduler schedule: {schedule}")
                self.db.set_run_details(run_id=run_id, status=RunStatus.ONGOING)
                self.db.create_worker_task(
                    worker_id,
                    WorkerTask.TRAIN_VAL,
                    TaskStatus.SCHEDULED,
                    run_id,
                    chunk_id,
                    config_options={"create_model_fn": create_model_fn},
                )
                self.logger.debug(f"Scheduled run {run_id} on worker {worker_id} for chunk {chunk_id}")

                # Small delay
                time.sleep(1)

                # Update prev_worker_tasks for next iteration (only track task_id and status)
                prev_worker_tasks = {
                    worker_id: {
                        "task_id": worker_task["task_id"],
                        "status": worker_task["status"],
                    }
                    for worker_id, worker_task in all_worker_tasks.items()
                }

            # set experiment task to idle
            self.db.set_experiment_current_task(ExperimentTask.IDLE)
            self.logger.debug(f"Set experiment task to {ExperimentTask.IDLE.value}.")

        except Exception as e:
            self._shutdown_workers()
            raise ControllerException(f"Error during run_fit: {e}") from e

        # Shutdown workers gracefully
        self._shutdown_workers()

    def _shutdown_workers(self, timeout: float = 60.0) -> None:
        """
        Gracefully shutdown all worker actors.

        Uses hybrid approach:
        1. Request graceful shutdown (wait for chunk completion)
        2. ray.kill() as fallback after timeout

        Args:
            timeout: Maximum seconds to wait for graceful shutdown
        """
        if not self.worker_actors:
            return

        self.logger.info("Shutting down worker actors...")

        # Request graceful shutdown for all workers
        shutdown_futures = []
        for worker in self.worker_actors:
            try:
                shutdown_futures.append(worker.request_shutdown.remote())
            except Exception as e:
                self.logger.warning(f"Error requesting shutdown: {e}")

        # Wait for graceful shutdown with timeout
        try:
            ray.get(shutdown_futures, timeout=timeout)
            self.logger.info("All workers shut down gracefully")
        except ray.exceptions.GetTimeoutError:
            self.logger.warning(f"Timeout waiting for graceful shutdown after {timeout}s. Force killing workers...")
            for worker in self.worker_actors:
                try:
                    ray.kill(worker)
                except Exception as e:
                    self.logger.debug(f"Error killing worker: {e}")
        except Exception as e:
            self.logger.warning(f"Error during graceful shutdown: {e}. Force killing workers...")
            import contextlib

            for worker in self.worker_actors:
                with contextlib.suppress(Exception):
                    ray.kill(worker)

        self.worker_actors = []
        self.logger.info("Worker shutdown complete")
