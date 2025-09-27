"""This module contains the Controller class which is responsible for orchestrating the RapidFire lifecycle."""

import contextlib
import math
import random
import time
from collections.abc import Callable
from logging import Logger
from pathlib import Path
from pprint import pformat
from typing import Any

import mlflow
import torch
from torch.utils.data import Dataset

from rapidfireai.automl import AutoMLAlgorithm
from rapidfireai.backend.chunks import DatasetChunks
from rapidfireai.backend.scheduler import Scheduler
from rapidfireai.db.rf_db import RfDb
from rapidfireai.utils.automl_utils import get_flattened_config_leaf, get_runs
from rapidfireai.utils.constants import (
    MLFLOW_URL,
    ControllerTask,
    ExperimentTask,
    RunEndedBy,
    RunSource,
    RunStatus,
    TaskStatus,
    WorkerTask,
)
from rapidfireai.utils.datapaths import DataPath
from rapidfireai.utils.distributed_utils import find_free_port
from rapidfireai.utils.exceptions import ControllerException, NoGPUsFoundException
from rapidfireai.utils.logging import RFLogger
from rapidfireai.utils.mlflow_manager import MLflowManager
from rapidfireai.utils.serialize import encode_payload
from rapidfireai.utils.shm_manager import SharedMemoryManager
from rapidfireai.utils.worker_manager import WorkerManager


class Controller:
    """This module contains the ML Controller class which is responsible for orchestrating the RapidFire lifecycle."""

    def __init__(self, experiment_id: int, experiment_name: str) -> None:
        """Initialize the controller."""
        import torch.multiprocessing as mp

        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn", force=True)

        self.experiment_id: int = experiment_id
        self.experiment_name: str = experiment_name

        # create database object
        self.db: RfDb = RfDb()

        # create controller logger
        logging = RFLogger()
        self.logger: Logger = logging.create_logger("controller")
        self.user_logger: Logger = logging.create_logger("user")
        self.ic_logger: Logger = logging.create_logger("interactive-control")

        # get number of GPUs
        self.num_workers: int = torch.cuda.device_count()
        if self.num_workers == 0:
            raise NoGPUsFoundException("No GPUs found while initializing controller.")
        self.logger.debug(f"Found {self.num_workers} workers/GPUs.")

        # set default required workers
        self.default_req_workers: int = 1

        # initialize shared manager and registry, create shared memory manager instance
        self.shm_manager: SharedMemoryManager = SharedMemoryManager(name="controller-shm")
        registry, process_lock = self.shm_manager.get_shm_objects()

        # create worker manager
        self.worker_manager: WorkerManager = WorkerManager(self.num_workers, registry, process_lock)

        # create mlflow manager
        self.mlflow_manager: MLflowManager = MLflowManager(MLFLOW_URL)
        self.mlflow_manager.get_experiment(self.experiment_name)

        self.logger.debug("Controller initialized")

    def _create_model_entries(
        self,
        param_config: AutoMLAlgorithm | dict[str, Any],
        source: RunSource,
        seed: int,
        len_train_dataset: int,
        num_chunks: int,
        clone_modify_info: dict[str, Any] | None = None,
    ) -> list[int]:
        """Creates all model-related entries - database values, directories and MLFlow run."""

        # get config_leaf from param_config for each run
        config_leafs = get_runs(param_config, seed)

        # create runs
        runs = {}
        for config_leaf in config_leafs:
            # get flattened config and total steps
            flattened_config = get_flattened_config_leaf(config_leaf)
            total_steps = self._get_total_step(config_leaf, len_train_dataset, num_chunks)

            # get clone modify info
            if clone_modify_info:
                warm_started = clone_modify_info.get("warm_started", False)
                cloned_from = clone_modify_info.get("cloned_from", None)
                chunk_offset = clone_modify_info.get("chunk_offset", 0)
            else:
                warm_started = False
                cloned_from = None
                chunk_offset = 0

            # determine estimated runtime and required workers
            if warm_started:
                # use parent run's estimated runtime and required workers
                parent_run_details = self.db.get_run(cloned_from)
                estimated_runtime = parent_run_details["estimated_runtime"]
                required_workers = parent_run_details["required_workers"]
            else:
                # add an initial random estimated runtime
                estimated_runtime = random.uniform(1.0, 10.0)
                required_workers = config_leaf.get("num_gpus", self.default_req_workers)

            # create run
            run_id = self.db.create_run(
                config_leaf=config_leaf,
                status=RunStatus.NEW,
                completed_steps=0,
                total_steps=total_steps,
                error="",
                source=source,
                ended_by=None,
                chunk_offset=chunk_offset,
                warm_started=warm_started,
                cloned_from=cloned_from,
                estimated_runtime=estimated_runtime,
                required_workers=required_workers,
            )
            runs[run_id] = flattened_config

            # create directories for each run
            try:
                base_run_path = DataPath.base_run_path(run_id)
                work_dir_path = DataPath.work_dir_path(base_run_path)
                initial_checkpoint_path = DataPath.initial_checkpoint_path(base_run_path)
                final_checkpoint_path = DataPath.final_checkpoint_path(base_run_path)
                intermediate_checkpoint_path = DataPath.intermediate_checkpoint_path(base_run_path)

                Path.mkdir(work_dir_path, parents=True, exist_ok=True)
                Path.mkdir(initial_checkpoint_path, parents=True, exist_ok=True)
                Path.mkdir(final_checkpoint_path, parents=True, exist_ok=True)
                Path.mkdir(intermediate_checkpoint_path, parents=True, exist_ok=True)
            except (PermissionError, OSError) as e:
                raise ControllerException(f"Failed to create required Run DataPath directories: {e}") from e

            # create new MlFlow run
            try:
                # create new MlFlow run and get the mlflow_run_id
                mlflow_run_id = self.mlflow_manager.create_run(str(run_id))

                # populate MLFlow with model config info
                for key, value in flattened_config.items():
                    self.mlflow_manager.log_param(mlflow_run_id, key, value)
                if warm_started:
                    self.mlflow_manager.log_param(mlflow_run_id, "warm-start", str(warm_started))
                if cloned_from:
                    self.mlflow_manager.log_param(mlflow_run_id, "parent-run", str(cloned_from))
                self.logger.debug(f"Populated MLFlow with model config info for run {run_id}.")
                self.db.set_run_details(
                    run_id=run_id,
                    mlflow_run_id=mlflow_run_id,
                    flattened_config=flattened_config,
                )
            except mlflow.exceptions.MlflowException as e:
                msg = f"Error creating new MLFlow run for run {run_id} - {e}."
                print(msg)
                self.mlflow_manager.end_run(mlflow_run_id)
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

    def _execute_interactive_control(
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
                self.db.set_ic_ops_task_status(run_state["task_id"], TaskStatus.COMPLETED)
                self.ic_logger.info(f"Stopping run {run_id} by Interactive Control")
            elif run_state["status"] == RunStatus.DELETED:
                # process deleted tasks
                # clear run from shm
                # TODO: commented out to prevent clone of deleted runs issue (see Issue # 22)
                # self._clear_run_from_shm(run_id)

                # delete run from MLFlow
                mlflow_run_id = self.db.get_run(run_id)["mlflow_run_id"]
                self.mlflow_manager.delete_run(mlflow_run_id)
                # mark run as deleted
                self.db.set_run_details(
                    run_id=run_id,
                    status=RunStatus.DELETED,
                    ended_by=RunEndedBy.INTERACTIVE_CONTROL,
                )
                self.db.set_ic_ops_task_status(run_state["task_id"], TaskStatus.COMPLETED)
                self.ic_logger.info(f"Deleting run {run_id} by Interactive Control")
            elif run_state["status"] == RunStatus.ONGOING:
                # process ongoing tasks
                self.db.set_run_details(
                    run_id=run_id,
                    status=RunStatus.ONGOING,
                    ended_by="",
                )
                self.db.set_ic_ops_task_status(run_state["task_id"], TaskStatus.COMPLETED)
                self.ic_logger.info(f"Resuming run {run_id} by Interactive Control")
            elif run_state["status"] == RunStatus.COMPLETED:
                # process completed tasks
                self.logger.warning(f"Run {run_id} is already completed. Skipping Interactive Control task.")
                self.db.set_ic_ops_task_status(run_state["task_id"], TaskStatus.SKIPPED)
            else:
                raise ValueError(f"Unsupported run status {run_state['status']}")

        # process clone_modify tasks from the collected list
        for task in clone_modify_tasks:
            parent_run_id, ic_op, config_leaf = (
                task["run_id"],
                task["ic_op"],
                task["config_leaf"],
            )

            # add additional_kwargs to config_leaf if it exists in the parent run
            parent_run_details = self.db.get_run(parent_run_id)
            if "additional_kwargs" in parent_run_details["config_leaf"]:
                config_leaf["additional_kwargs"] = parent_run_details["config_leaf"]["additional_kwargs"]

            # create model for the new run
            try:
                if ic_op == ControllerTask.IC_CLONE_MODIFY:
                    clone_modify_info = {
                        "cloned_from": parent_run_id,
                        "warm_started": False,
                        "chunk_offset": 0,
                    }
                    run_ids = self._create_model_entries(
                        config_leaf,
                        RunSource.INTERACTIVE_CONTROL,
                        seed,
                        len_train_dataset,
                        num_chunks=num_chunks,
                        clone_modify_info=clone_modify_info,
                    )
                elif ic_op == ControllerTask.IC_CLONE_MODIFY_WARM:
                    # calculate clone chunk offset
                    per_device_train_batch_size = parent_run_details["config_leaf"]["training_args"].get(
                        "per_device_train_batch_size", 1
                    )
                    gradient_accumulation_steps = parent_run_details["config_leaf"]["training_args"].get(
                        "gradient_accumulation_steps", 1
                    )
                    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
                    chunker = DatasetChunks(
                        len_train_dataset,
                        num_chunks,
                        batch_size=effective_batch_size,
                        offset=parent_run_details["chunk_offset"],
                    )
                    clone_chunk_offset = chunker.get_clone_offset(parent_run_details["num_chunks_visited_curr_epoch"])
                    clone_modify_info = {
                        "cloned_from": parent_run_id,
                        "warm_started": True,
                        "chunk_offset": clone_chunk_offset,
                    }
                    run_ids = self._create_model_entries(
                        config_leaf,
                        RunSource.INTERACTIVE_CONTROL,
                        seed,
                        len_train_dataset,
                        num_chunks,
                        clone_modify_info,
                    )
                else:
                    raise ValueError(f"Unsupported IC operation {ic_op}")

                # mark task as completed
                self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.COMPLETED)
                self.ic_logger.info(
                    f"Cloned run {parent_run_id} by Interactive Control with {ic_op.value} into runs - {run_ids}"
                )
            except Exception as e:
                self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.FAILED)
                self.ic_logger.error(f"Error creating model for run {parent_run_id}: {e}")
                raise ControllerException(f"Error creating model for run {parent_run_id}: {e}") from e

    def _process_interm_ic_ops_states(
        self,
        currently_scheduled_runs: list[int],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Process the interactive control."""
        # get IC Ops scheduled tasks
        ic_scheduled_tasks = self.db.get_scheduled_ic_ops_tasks()

        # track states for each task(run) and collect clone_modify tasks separately
        run_states = {}
        clone_modify_tasks = []
        for task in ic_scheduled_tasks:
            run_id = task["run_id"]

            # skip if run is currently scheduled (we process IC ops only at chunk boundaries)
            if run_id in currently_scheduled_runs:
                # self.logger.debug(f"Skipping IC op for run {run_id} as it is currently scheduled")
                continue

            is_clone_modify_task = task["ic_op"] in (
                ControllerTask.IC_CLONE_MODIFY,
                ControllerTask.IC_CLONE_MODIFY_WARM,
            )

            if is_clone_modify_task:
                # clone_modify tasks
                # get latest run state
                run_status = run_states[run_id]["status"] if run_id in run_states else self.db.get_run(run_id)["status"]

                # track clone_modify tasks only for non-deleted runs
                if run_status != RunStatus.DELETED:
                    clone_modify_tasks.append(task)
                    self.ic_logger.info(f"Added {task['ic_op']} task for run {run_id}.")
                else:
                    self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.SKIPPED)
                    self.ic_logger.warning(f"Skipping {task['ic_op']} task for deleted run {run_id}.")
            else:
                # Non clone_modify tasks
                if run_id not in run_states:
                    run_states[run_id] = {
                        "task_id": None,
                        "task": None,
                        "status": self.db.get_run(run_id)["status"],
                    }

                # update run states based on existing status and task
                current_status = run_states[run_id]["status"]
                if current_status == RunStatus.COMPLETED and task["ic_op"] in [
                    ControllerTask.IC_RESUME,
                    ControllerTask.IC_STOP,
                ]:
                    # ignore RESUME/STOP tasks for completed runs
                    self.ic_logger.warning(f"Ignoring RESUME/STOP task for run {run_id} as it is already completed")
                    self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.SKIPPED)
                elif current_status == RunStatus.FAILED and task["ic_op"] != ControllerTask.IC_DELETE:
                    # ignore all tasks except DELETE for failed runs
                    self.ic_logger.warning(f"Ignoring task {task['ic_op'].value} for failed run {run_id}")
                    self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.SKIPPED)
                elif current_status == RunStatus.DELETED:
                    # ignore all tasks for deleted runs
                    self.ic_logger.warning(f"Ignoring task {task['ic_op'].value} for deleted run {run_id}")
                    self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.SKIPPED)
                else:
                    # valid ic_op for this run
                    # mark prev task as completed
                    if run_states[run_id]["task_id"] is not None:
                        self.db.set_ic_ops_task_status(run_states[run_id]["task_id"], TaskStatus.COMPLETED)

                    # add new task to run states
                    if task["ic_op"] == ControllerTask.IC_STOP:
                        updated_status = RunStatus.STOPPED
                        info_msg = f"Received STOP task for run {run_id}"
                    elif task["ic_op"] == ControllerTask.IC_DELETE:
                        updated_status = RunStatus.DELETED
                        info_msg = f"Received DELETE task for run {run_id}"
                    elif task["ic_op"] == ControllerTask.IC_RESUME:
                        updated_status = RunStatus.ONGOING
                        info_msg = f"Received RESUME task for run {run_id}"
                    else:
                        self.db.set_ic_ops_task_status(task["task_id"], TaskStatus.FAILED)
                        raise ValueError(f"Unsupported task {task['ic_op']}")
                    run_states[run_id].update(
                        {
                            "task_id": task["task_id"],
                            "task": task["ic_op"],
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
        num_gpus: int = 1,
        monte_carlo_simulations: int = 1000,
    ) -> None:
        """Run the fit."""

        # set default required workers
        self.default_req_workers = num_gpus

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
            self._create_model_entries(param_config, RunSource.INITIAL, seed, len_train_dataset, num_chunks=num_chunks)
            self.logger.debug("Created models.")
        except Exception as e:
            raise ControllerException(f"Error creating models: {e}") from e

        # set experiment task to create models
        self.db.set_experiment_current_task(ExperimentTask.RUN_FIT)
        self.logger.debug(f"Set experiment task to {ExperimentTask.RUN_FIT.value}.")

        # create workers
        try:
            self.worker_manager.create_workers()
            print("Created workers")
            self.logger.debug(f"Created {self.num_workers} workers.")
        except Exception as e:
            raise ControllerException(f"Error creating workers: {e}") from e

        # create scheduler
        runs_info = []
        run_ids = list(self.db.get_runs_by_status([RunStatus.NEW]).keys())
        for run_id in run_ids:
            run_details = self.db.get_run(run_id)
            runs_info.append(
                {
                    "run_id": run_id,
                    "req_workers": run_details["required_workers"],
                    "estimated_runtime": run_details["estimated_runtime"],
                }
            )

        scheduler = Scheduler(runs_info, self.num_workers, num_chunks, monte_carlo_simulations)

        # run fit
        self.logger.info("Starting Training and Validation")
        try:
            all_done = False
            prev_worker_tasks = {}  # Track previous iteration's worker tasks
            active_runs = {}  # Track runs that are currently training and their workers: {run_id: worker_ids_tuple}

            while not all_done:
                # check for errors
                exp_error = self.db.get_experiment_error()
                if exp_error:
                    print(f"Error in experiment: {exp_error}")
                    self.logger.error(f"Error in experiment: {exp_error}")
                    break

                # get current state (pre IC ops states)
                all_worker_tasks = self.db.get_all_worker_tasks()
                all_run_details = self.db.get_all_runs()

                # Process completed and failed tasks by run
                completed_runs = set()
                failed_runs = set()

                for worker_id, worker_task in all_worker_tasks.items():
                    prev_task = prev_worker_tasks.get(worker_id, {})
                    prev_task_tuple = (prev_task.get("task_id"), prev_task.get("status"))
                    current_task_tuple = (worker_task["task_id"], worker_task["status"])

                    # skip if task is the same as previous iteration (no change in status) or run is not active
                    if current_task_tuple == prev_task_tuple or worker_task["run_id"] not in scheduler.run_ids:
                        continue

                    run_id = worker_task["run_id"]
                    if worker_task["status"] == TaskStatus.COMPLETED and run_id in active_runs:
                        completed_runs.add(run_id)
                    elif worker_task["status"] == TaskStatus.FAILED and run_id in active_runs:
                        failed_runs.add(run_id)

                # Process completed tasks first (before scheduling new ones)
                for run_id in completed_runs:
                    if run_id not in scheduler.run_ids:
                        continue

                    chunk_id = worker_task["chunk_id"]
                    run_details = all_run_details[run_id]
                    self.logger.debug(
                        f"Completed task: run {run_id}, chunk {chunk_id} on workers {active_runs[run_id]}"
                    )
                    self.logger.info(
                        f"Run {run_id} completed steps - {run_details['completed_steps']}/{run_details['total_steps']}"
                    )

                    # Update scheduler state
                    scheduler.set_completed_task(run_id)

                    # Update database state and local state using scheduler's state as source of truth
                    new_num_chunks_visited = scheduler.state.run_visited_num_chunks[run_id]
                    if new_num_chunks_visited == num_chunks:
                        num_epochs_completed = run_details["num_epochs_completed"] + 1
                    else:
                        num_epochs_completed = run_details["num_epochs_completed"]
                    self.db.set_run_details(
                        run_id=run_id,
                        num_chunks_visited_curr_epoch=new_num_chunks_visited,
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
                        new_num_chunks_visited == num_chunks
                        and run_details["completed_steps"] < run_details["total_steps"]
                    ):
                        scheduler.reset_run(run_id)
                        self.db.set_run_details(run_id=run_id, num_chunks_visited_curr_epoch=0)
                        self.logger.info(
                            f"Run {run_id} has completed epoch ({new_num_chunks_visited}/{num_chunks} chunks)"
                        )

                        # Remove from active runs
                        active_runs.pop(run_id, None)

                # Check for failed runs and update scheduler, local state, shm
                for run_id in failed_runs:
                    if run_id in scheduler.run_ids:
                        run_error = all_run_details[run_id]["error"]
                        scheduler.remove_run(run_id)
                        self._clear_run_from_shm(run_id)

                        active_runs.pop(run_id, None)

                        err_msg = f"Run {run_id} has failed: {run_error}"
                        print(err_msg)
                        self.logger.error(err_msg)
                        self.logger.debug(f"Removed run {run_id} from scheduler")

                # Process and execute interactive control tasks (this fetches latest run states internally)
                try:
                    currently_scheduled_runs = list(active_runs.keys())
                    run_states, clone_modify_tasks = self._process_interm_ic_ops_states(currently_scheduled_runs)
                    self._execute_interactive_control(
                        run_states, clone_modify_tasks, len_train_dataset, seed, num_chunks
                    )
                except Exception as e:
                    raise ControllerException(f"Error executing interactive control tasks: {e}") from e

                # fetch latest run states again (post IC ops states)
                all_run_details = self.db.get_all_runs()

                # Update scheduler with active and inactive runs from IC Ops changes
                for run_id, run_details in all_run_details.items():
                    if run_details["status"] in (RunStatus.ONGOING, RunStatus.NEW) and run_id not in scheduler.run_ids:
                        # add new runs to scheduler
                        run_info = {
                            "run_id": run_id,
                            "req_workers": run_details["required_workers"],
                            "estimated_runtime": run_details["estimated_runtime"],
                            "start_chunk_id": run_details["start_chunk_id"],
                        }
                        chunks_visited = all_run_details[run_id]["num_chunks_visited_curr_epoch"]
                        scheduler.add_run(run_info, chunks_visited)
                        self.logger.debug(f"Added run {run_id} to scheduler with {chunks_visited} chunks visited")
                    elif (
                        run_details["status"] in (RunStatus.STOPPED, RunStatus.DELETED) and run_id in scheduler.run_ids
                    ):
                        # remove inactive runs from scheduler
                        scheduler.remove_run(run_id)
                        active_runs.pop(run_id, None)
                        self.logger.debug(f"Removed run {run_id} from scheduler")

                # Get best-first action schedule from scheduler
                schedule = scheduler.schedule()
                run_id = schedule["run_id"]
                worker_ids = schedule["worker_ids"]
                chunk_id = schedule["chunk_id"]

                # Check termination condition
                if run_id is None:
                    self.logger.info("Scheduler indicates all runs have completed all chunks")
                    all_done = True
                    break

                # Check if no schedule possible
                if run_id == -1:
                    # self.logger.debug("No schedule possible - all workers busy or no available runs")
                    time.sleep(1)
                    continue

                # Execute Schedule
                # self.logger.debug(f"Scheduler schedule: {schedule}")
                # update run status to ongoing
                self.db.set_run_details(run_id=run_id, status=RunStatus.ONGOING)

                # track this run in active_runs
                active_runs[run_id] = worker_ids

                # create a worker task for each worker
                multi_worker_details = {
                    "world_size": len(worker_ids),
                    "worker_ids": worker_ids,
                    "master_address": "localhost",
                    "master_port": find_free_port(),
                }
                for worker_id in worker_ids:
                    multi_worker_details["local_rank"] = worker_ids.index(worker_id)
                    self.db.create_worker_task(
                        worker_id,
                        WorkerTask.TRAIN_VAL,
                        TaskStatus.SCHEDULED,
                        run_id,
                        chunk_id,
                        multi_worker_details=multi_worker_details,
                        config_options={"create_model_fn": create_model_fn},
                    )
                self.logger.debug(f"Scheduled run {run_id} on workers {worker_ids} for chunk {chunk_id}")

                # Update prev_worker_tasks for next iteration (only track task_id and status)
                prev_worker_tasks = {
                    worker_id: {
                        "task_id": worker_task["task_id"],
                        "status": worker_task["status"],
                    }
                    for worker_id, worker_task in all_worker_tasks.items()
                }

                # Small delay
                time.sleep(1)

            # set experiment task to idle
            self.db.set_experiment_current_task(ExperimentTask.IDLE)
            self.logger.debug(f"Set experiment task to {ExperimentTask.IDLE.value}.")

        except Exception as e:
            raise ControllerException(f"Error during run_fit: {e}") from e

        # shutdown workers
        self.worker_manager.shutdown()
