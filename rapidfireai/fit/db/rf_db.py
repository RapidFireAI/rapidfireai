"""This module contains the RfDb class which is responsible for handling the database operations."""

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from rapidfireai.fit.db.db_interface import DatabaseInterface
from rapidfireai.fit.utils.constants import (
    ControllerTask,
    ExperimentStatus,
    ExperimentTask,
    RunEndedBy,
    RunSource,
    RunStatus,
    TaskStatus,
    WorkerTask,
)
from rapidfireai.fit.utils.exceptions import DBException
from rapidfireai.fit.utils.serialize import decode_db_payload, encode_payload

# TODO: add custom exceptions like - RunNotFoundError, ExperimentNotFoundError, ICOpsOnKilledRuns, etc


class RfDb:
    """Class to handle the database operations"""

    def __init__(self):
        self.db: DatabaseInterface = DatabaseInterface()

    def create_tables(self):
        """Create the tables in the database"""
        try:
            # Get the directory where this file is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            tables_file = os.path.join(current_dir, "tables.sql")

            # First check if tables exist
            table_check = self.db.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='experiments'"
            ).fetchone()

            if table_check is None:
                # Tables don't exist, create them
                with open(tables_file, encoding="utf-8") as f:
                    sql_content = f.read()
                _ = self.db.conn.executescript(sql_content)
            else:
                try:
                    cursor = self.db.conn.execute("PRAGMA table_info(runs)")
                    columns = [column[1] for column in cursor.fetchall()]
                    if "metric_run_id" not in columns:
                        self.db.conn.execute("ALTER TABLE runs ADD COLUMN metric_run_id TEXT")
                        self.db.conn.commit()
                except sqlite3.Error:
                    pass
                try:
                    cursor = self.db.conn.execute("PRAGMA table_info(experiments)")
                    columns = [column[1] for column in cursor.fetchall()]
                    if "metric_experiment_id" not in columns:
                        self.db.conn.execute("ALTER TABLE experiments ADD COLUMN metric_experiment_id TEXT")
                        self.db.conn.commit()
                except sqlite3.Error:
                    pass
        except FileNotFoundError as e:
            raise DBException(f"tables.sql file not found at {tables_file}") from e
        except sqlite3.Error as e:
            raise DBException(f"Failed to create tables: {e}") from e
        except Exception as e:
            raise DBException(f"Unexpected error creating tables: {e}") from e

    def close(self):
        """Close the database connection"""
        self.db.close()

    def reset_all_tables(self, experiments_table: bool = False) -> None:
        """Truncate tables when an experiment is ended"""
        # ordering based on foreign key constraints
        tables = [
            "controller_progress",
            "worker_progress",
            "worker_task",
            "interactive_control",
            "runs",
        ]

        if experiments_table:
            tables.append("experiments")

        for table in tables:
            query = f"DELETE FROM {table};"
            self.db.execute(query, commit=True)

        # Reset auto-increment indices to start from 1
        for table in tables:
            query = "DELETE FROM sqlite_sequence WHERE name = ?;"
            self.db.execute(query, (table,), commit=True)

    def reset_experiment_states(self) -> None:
        """Reset the experiment states when a running task is cancelled"""

        # mark all scheduled and in-progress worker tasks as failed
        query = """
            UPDATE worker_task
            SET status = ?
            WHERE status = ? OR status = ?
        """
        self.db.execute(
            query, (TaskStatus.FAILED.value, TaskStatus.IN_PROGRESS.value, TaskStatus.SCHEDULED.value), commit=True
        )

        # mark ongoing and new Runs as failed
        all_runs = self.get_runs_by_status([RunStatus.ONGOING, RunStatus.NEW]).keys()
        for run_id in all_runs:
            self.set_run_status(run_id, RunStatus.FAILED)

        # reset all interactive control tasks
        all_ic_ops_tasks = self.get_scheduled_ic_ops_tasks()
        for task in all_ic_ops_tasks:
            self.set_ic_ops_task_status(task["task_id"], TaskStatus.SKIPPED)

        # reset all progress tables
        for table in ["controller_progress", "worker_progress"]:
            query = f"DELETE FROM {table};"
            self.db.execute(query, commit=True)

    # Experiments Table
    def create_experiment(
        self,
        experiment_name: str,
        metric_experiment_id: str | None,
        config_options: dict[str, Any],
    ) -> int:
        """Create a new experiment"""
        query = """
            INSERT INTO experiments (experiment_name, metric_experiment_id, config_options,
            status, current_task, error)
            VALUES (?, ?, ?, ?, ?, ?)
            RETURNING experiment_id
        """

        result = self.db.execute(
            query,
            (
                experiment_name,
                metric_experiment_id,
                encode_payload(config_options),
                ExperimentStatus.RUNNING.value,
                ExperimentTask.IDLE.value,
                "",
            ),
            commit=True,
            fetch=True,
        )
        if result:
            # run db optimizer every time an experiment is created
            self.db.optimize_periodically()

            # return experiment_id
            return result[0][0]
        raise DBException("Failed to create experiment")

    def get_running_experiment(self) -> dict[str, Any]:
        """Get an experiment's details by its ID"""
        query = """
            SELECT experiment_id, experiment_name, status, error, metric_experiment_id, config_options
            FROM experiments
            WHERE status = ?
            ORDER BY experiment_id DESC
            LIMIT 1
        """
        experiment_details = self.db.execute(query, (ExperimentStatus.RUNNING.value,), fetch=True)

        if experiment_details:
            experiment_details = experiment_details[0]
            experiment_details = {
                "experiment_id": experiment_details[0],
                "experiment_name": experiment_details[1],
                "status": experiment_details[2],  # Return string value, not enum (for JSON serialization)
                "error": experiment_details[3],
                "metric_experiment_id": experiment_details[4],
                "config_options": decode_db_payload(experiment_details[5]),
            }
            return experiment_details
        raise DBException("No running experiment found")

    def get_experiment_status(self) -> ExperimentStatus | None:
        """Get the status of an experiment"""
        query = """
            SELECT status
            FROM experiments
            ORDER BY experiment_id DESC
            LIMIT 1
        """
        status = self.db.execute(query, fetch=True)

        if status:
            return ExperimentStatus(status[0][0])
        raise DBException("No experiment status found")

    def set_experiment_status(self, experiment_id: int, status: ExperimentStatus) -> None:
        """Set the status of an experiment"""
        query = """
            UPDATE experiments
            SET status = ?
            WHERE experiment_id = ?
        """
        self.db.execute(query, (status.value, experiment_id), commit=True)

    def set_experiment_error(self, error: str) -> None:
        """Set the error message of an experiment"""
        query = """
            UPDATE experiments
            SET error = ?
            WHERE status = ?
        """
        self.db.execute(query, (error, ExperimentStatus.RUNNING.value), commit=True)

    def get_experiment_error(self) -> str:
        """Get the error message of an experiment"""
        query = """
            SELECT error
            FROM experiments
            ORDER BY experiment_id DESC
            LIMIT 1
        """
        error = self.db.execute(query, fetch=True)

        if error:
            return error[0][0]
        else:
            return ""

    def set_experiment_current_task(self, task: ExperimentTask) -> None:
        """Set the current task of an experiment"""
        query = """
            UPDATE experiments
            SET current_task = ?
            WHERE status = ?
        """
        self.db.execute(query, (task.value, ExperimentStatus.RUNNING.value), commit=True)

    def get_experiment_current_task(self) -> ExperimentTask:
        """Get the current task of an experiment"""
        query = """
            SELECT current_task
            FROM experiments
            WHERE status = ?
            ORDER BY experiment_id DESC
            LIMIT 1
        """
        task = self.db.execute(query, (ExperimentStatus.RUNNING.value,), fetch=True)

        if task:
            return ExperimentTask(task[0][0])
        raise DBException("No running experiment found")

    def get_all_experiment_names(self) -> list[str]:
        """Get all experiment names"""
        query = """
            SELECT experiment_name
            FROM experiments
        """
        experiment_names = self.db.execute(query, fetch=True)

        if experiment_names:
            return [experiment[0] for experiment in experiment_names]
        else:
            return []

    def get_experiments_path(self, experiment_name: str) -> Path:
        """Get the experiments path for a given experiment name"""
        query = """
            SELECT config_options
            FROM experiments
            WHERE experiment_name = ?
        """
        config_options = self.db.execute(query, (experiment_name,), fetch=True)
        if config_options:
            config_dict = decode_db_payload(config_options[0][0])
            return Path(config_dict["experiments_path"])
        raise DBException("Experiments path not found for running experiment")

    # Runs Table
    def create_run(
        self,
        config_leaf: dict[str, Any],
        status: RunStatus,
        metric_run_id: str | None = None,
        flattened_config: dict[str, Any] | None = None,
        completed_steps: int = 0,
        total_steps: int = 0,
        num_chunks_visited_curr_epoch: int = 0,
        num_epochs_completed: int = 0,
        chunk_offset: int = 0,
        error: str = "",
        source: RunSource | None = None,
        ended_by: RunEndedBy | None = None,
        warm_started_from: int | None = None,
        cloned_from: int | None = None,
    ) -> int:
        """Create a new run"""
        query = """
            INSERT INTO runs (status, metric_run_id, flattened_config, config_leaf,
            completed_steps, total_steps, num_chunks_visited_curr_epoch,
            num_epochs_completed, chunk_offset, error, source, ended_by, warm_started_from, cloned_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        self.db.execute(
            query,
            (
                status.value,
                metric_run_id,
                json.dumps(flattened_config) if flattened_config else "{}",
                encode_payload(config_leaf) if config_leaf else "{}",
                completed_steps,
                total_steps,
                num_chunks_visited_curr_epoch,
                num_epochs_completed,
                chunk_offset,
                error,
                source.value if source else "",
                ended_by.value if ended_by else "",
                warm_started_from,
                cloned_from,
            ),
            commit=True,
        )
        result = self.db.execute("SELECT last_insert_rowid()", fetch=True)
        if result:
            return result[0][0]
        raise DBException("Failed to create run")

    def set_run_details(
        self,
        run_id: int,
        status: RunStatus | None = None,
        metric_run_id: str | None = None,
        flattened_config: dict[str, Any] | None = None,
        config_leaf: dict[str, Any] | None = None,
        completed_steps: int | None = None,
        total_steps: int | None = None,
        num_chunks_visited_curr_epoch: int | None = None,
        num_epochs_completed: int | None = None,
        chunk_offset: int | None = None,
        error: str | None = None,
        source: RunSource | None = None,
        ended_by: RunEndedBy | None = None,
        warm_started_from: int | None = None,
        cloned_from: int | None = None,
    ) -> None:
        """Set the details of an existing run"""
        # Initialize a dictionary to hold the column-value pairs
        columns = {
            "status": status.value if status else None,
            "metric_run_id": metric_run_id,
            "flattened_config": json.dumps(flattened_config) if flattened_config else None,
            "config_leaf": encode_payload(config_leaf) if config_leaf else None,
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "num_chunks_visited_curr_epoch": num_chunks_visited_curr_epoch,
            "num_epochs_completed": num_epochs_completed,
            "chunk_offset": chunk_offset,
            "error": error,
            "source": source.value if source else None,
            "ended_by": ended_by.value if ended_by else None,
            "warm_started_from": warm_started_from,
            "cloned_from": cloned_from,
        }

        # Filter out None values
        columns = {k: v for k, v in columns.items() if v is not None}

        # If no columns to update, return early
        if not columns:
            return

        # Construct the query parts
        query_parts = [f"{col} = ?" for col in columns]
        values = list(columns.values())

        # Ensure run_id is always included
        values.append(run_id)

        # Construct the final query
        query = f"UPDATE runs SET {', '.join(query_parts)} WHERE run_id = ?"

        # Execute the query
        self.db.execute(query, tuple(values), commit=True)

    def get_run(self, run_id: int) -> dict[str, Any]:
        """Get a run's details"""
        query = """
            SELECT status, metric_run_id, flattened_config, config_leaf, completed_steps, total_steps,
            num_chunks_visited_curr_epoch, num_epochs_completed, chunk_offset, error, source, ended_by,
            warm_started_from, cloned_from
            FROM runs
            WHERE run_id = ?
        """
        run_details = self.db.execute(query, (run_id,), fetch=True)

        if run_details:
            run_details = run_details[0]
            formatted_details = {
                "status": RunStatus(run_details[0]),
                "metric_run_id": run_details[1],
                "flattened_config": json.loads(run_details[2]),
                "config_leaf": decode_db_payload(run_details[3]) if run_details[3] and run_details[3] != "{}" else {},
                "completed_steps": run_details[4],
                "total_steps": run_details[5],
                "num_chunks_visited_curr_epoch": run_details[6],
                "num_epochs_completed": run_details[7],
                "chunk_offset": run_details[8],
                "error": run_details[9],
                "source": RunSource(run_details[10]) if run_details[10] else None,
                "ended_by": RunEndedBy(run_details[11]) if run_details[11] else None,
                "warm_started_from": run_details[12],
                "cloned_from": run_details[13],
            }
            return formatted_details
        raise DBException("No run found")

    def get_runs_by_status(self, statuses: list[RunStatus]) -> dict[int, dict[str, Any]]:
        """Get all runs by statuses"""
        if not statuses:
            return {}

        # Create placeholders for SQL IN clause
        placeholders = ",".join(["?"] * len(statuses))
        query = f"""
            SELECT run_id, status, metric_run_id, flattened_config, config_leaf, completed_steps, total_steps,
            num_chunks_visited_curr_epoch, num_epochs_completed, chunk_offset, error, source, ended_by,
            warm_started_from, cloned_from
            FROM runs
            WHERE status IN ({placeholders})
        """
        # Extract status values for the query parameters
        status_values = [status.value for status in statuses]
        run_details = self.db.execute(query, status_values, fetch=True)
        formatted_details: dict[int, dict[str, Any]] = {}
        if run_details:
            for run in run_details:
                formatted_details[run[0]] = {
                    "status": RunStatus(run[1]),
                    "metric_run_id": run[2],
                    "flattened_config": json.loads(run[3]),
                    "config_leaf": decode_db_payload(run[4]) if run[4] and run[4] != "{}" else {},
                    "completed_steps": run[5],
                    "total_steps": run[6],
                    "num_chunks_visited_curr_epoch": run[7],
                    "num_epochs_completed": run[8],
                    "chunk_offset": run[9],
                    "error": run[10],
                    "source": RunSource(run[11]) if run[11] else None,
                    "ended_by": RunEndedBy(run[12]) if run[12] else None,
                    "warm_started_from": run[13],
                    "cloned_from": run[14],
                }
        return formatted_details

    def get_all_runs(self) -> dict[int, dict[str, Any]]:
        """Get all runs for UI display (ignore all complex fields)"""
        query = """
            SELECT run_id, status, metric_run_id, flattened_config, config_leaf, completed_steps, total_steps,
            num_chunks_visited_curr_epoch, num_epochs_completed, chunk_offset, error, source, ended_by,
            warm_started_from, cloned_from
            FROM runs
        """
        run_details = self.db.execute(query, fetch=True)

        formatted_details: dict[int, dict[str, Any]] = {}
        if run_details:
            for run in run_details:
                formatted_details[run[0]] = {
                    "status": RunStatus(run[1]),
                    "metric_run_id": run[2],
                    "flattened_config": json.loads(run[3]),
                    "config_leaf": decode_db_payload(run[4]) if run[4] and run[4] != "{}" else {},
                    "completed_steps": run[5],
                    "total_steps": run[6],
                    "num_chunks_visited_curr_epoch": run[7],
                    "num_epochs_completed": run[8],
                    "chunk_offset": run[9],
                    "error": run[10],
                    "source": RunSource(run[11]) if run[11] else None,
                    "ended_by": RunEndedBy(run[12]) if run[12] else None,
                    "warm_started_from": run[13],
                    "cloned_from": run[14],
                }
        return formatted_details

    def set_run_status(self, run_id: int, status: RunStatus) -> None:
        """Set the status of a run"""
        query = """
            UPDATE runs
            SET status = ?
            WHERE run_id = ?
        """
        self.db.execute(query, (status.value, run_id), commit=True)

    def set_completed_steps(self, run_id: int, completed_steps: int) -> None:
        """Set the current completed steps for a run"""
        query = """
            UPDATE runs
            SET completed_steps = ?
            WHERE run_id = ?
        """
        self.db.execute(query, (completed_steps, run_id), commit=True)

    def get_completed_steps(self, run_id: int) -> int:
        """Get the current completed steps for a run"""
        query = """
            SELECT completed_steps
            FROM runs
            WHERE run_id = ?
        """
        completed_steps = self.db.execute(query, (run_id,), fetch=True)
        if completed_steps:
            return completed_steps[0][0]
        raise DBException("No completed steps found")

    # Interactive Control Table
    def create_ic_ops_task(self, run_id: int, ic_op: ControllerTask, config_leaf: dict[str, Any]) -> int:
        """Create a new interactive control task"""
        query = """
            INSERT INTO interactive_control (run_id, ic_op, config_leaf, status)
            VALUES (?, ?, ?, ?)
            RETURNING task_id
        """
        config_leaf_str = encode_payload(config_leaf) if config_leaf else "{}"
        result = self.db.execute(
            query,
            (run_id, ic_op.value, config_leaf_str, TaskStatus.SCHEDULED.value),
            commit=True,
            fetch=True,
        )
        if result:
            return result[0][0]
        raise DBException("Failed to create interactive control task")

    def get_scheduled_ic_ops_tasks(self) -> list[dict[str, Any]]:
        """Get all scheduled interactive control operations"""
        query = """
            SELECT task_id, run_id, ic_op, config_leaf
            FROM interactive_control
            WHERE status = ?
        """
        tasks = self.db.execute(query, (TaskStatus.SCHEDULED.value,), fetch=True)
        if not tasks:
            return []
        return [
            {
                "task_id": task[0],
                "run_id": task[1],
                "ic_op": ControllerTask(task[2]),
                "config_leaf": decode_db_payload(task[3]),
            }
            for task in tasks
        ]

    def set_ic_ops_task_status(self, task_id: int, status: TaskStatus) -> None:
        """Set the status of an interactive control operation"""
        query = """
            UPDATE interactive_control
            SET status = ?
            WHERE task_id = ?
        """
        self.db.execute(query, (status.value, task_id), commit=True)

    # Worker Task Table
    def create_worker_task(
        self,
        worker_id: int,
        task_type: WorkerTask,
        status: TaskStatus,
        run_id: int,
        chunk_id: int = -1,
        config_options: dict[str, Any] | None = None,
    ) -> int:
        """Create a worker task"""

        query = """
            INSERT INTO worker_task (worker_id, task_type, status, run_id, chunk_id, config_options)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        config_options_str = encode_payload(config_options) if config_options else "{}"
        self.db.execute(
            query,
            (
                worker_id,
                task_type.value,
                status.value,
                run_id,
                chunk_id,
                config_options_str,
            ),
            commit=True,
        )
        result = self.db.execute("SELECT last_insert_rowid()", fetch=True)
        if result:
            return result[0][0]
        raise DBException("Failed to create worker task")

    def get_all_worker_tasks(self) -> dict[int, dict[str, Any]]:
        """Get the latest task of each worker"""
        query = """
            SELECT worker_id, task_id, task_type, status, run_id, chunk_id, config_options
            FROM worker_task wt1
            WHERE task_id = (
                SELECT MAX(task_id)
                FROM worker_task wt2
                WHERE wt2.worker_id = wt1.worker_id
            )
        """
        task_details = self.db.execute(query, fetch=True)

        formatted_details: dict[int, dict[str, Any]] = {}
        if task_details:
            for task in task_details:
                formatted_details[task[0]] = {
                    "task_id": task[1],
                    "task_type": WorkerTask(task[2]),
                    "status": TaskStatus(task[3]),
                    "run_id": task[4],
                    "chunk_id": task[5],
                    "config_options": decode_db_payload(task[6]) if task[6] and task[6] != "{}" else {},
                }
        return formatted_details

    def get_worker_scheduled_task(self, worker_id: int) -> dict[str, Any]:
        """Get the latest scheduled task for a worker"""
        query = """
            SELECT task_id, task_type, run_id, chunk_id, config_options
            FROM worker_task
            WHERE worker_id = ? AND status = ?
            ORDER BY task_id DESC
            LIMIT 1
        """
        task_details = self.db.execute(query, (worker_id, TaskStatus.SCHEDULED.value), fetch=True)

        if task_details:
            task_details = task_details[0]
            formatted_details = {
                "task_id": task_details[0],
                "task_type": WorkerTask(task_details[1]),
                "run_id": task_details[2],
                "chunk_id": task_details[3],
                "config_options": decode_db_payload(task_details[4])
                if task_details[4] and task_details[4] != "{}"
                else {},
            }
            return formatted_details
        return {}

    def set_worker_task_status(self, worker_id: int, status: TaskStatus) -> None:
        """Set the status of the latest task of a worker"""
        query = """
            UPDATE worker_task
            SET status = ?
            WHERE task_id = (
                SELECT task_id
                FROM worker_task
                WHERE worker_id = ?
                ORDER BY task_id DESC
                LIMIT 1
            )
        """
        self.db.execute(query, (status.value, worker_id), commit=True)

    # Train Controller Progress Table
    def set_controller_progress(self, run_id: int, progress: float) -> None:
        """Set the Train progress for a Train Controller"""
        query = """
            INSERT INTO controller_progress (run_id, progress)
            VALUES (?, ?)
            ON CONFLICT (run_id)
            DO UPDATE SET
                progress = EXCLUDED.progress;
        """
        progress_rounded = round(progress, 2)
        self.db.execute(query, (run_id, progress_rounded), commit=True)

    def get_controller_progress(self, run_id: int) -> float:
        """Get the train progress for a Controller"""
        query = """
            SELECT progress
            FROM controller_progress
            WHERE run_id = ?
        """
        progress_details = self.db.execute(query, (run_id,), fetch=True)

        if progress_details:
            return progress_details[0][0]
        return 0.0

    # Train Worker Progress Table
    def set_worker_progress(self, run_id: int, subchunk_progress: float) -> None:
        """Set the progress of a Worker for training"""
        query = """
            INSERT INTO worker_progress (run_id, subchunk_progress)
            VALUES (?, ?)
            ON CONFLICT (run_id)
            DO UPDATE SET
                subchunk_progress = EXCLUDED.subchunk_progress;
        """
        progress_rounded = round(subchunk_progress, 2)
        self.db.execute(query, (run_id, progress_rounded), commit=True)

    def get_worker_progress(self, run_id: int) -> float:
        """Get the progress of a Worker for training"""
        query = """
            SELECT subchunk_progress
            FROM worker_progress
            WHERE run_id = ?
        """
        progress_details = self.db.execute(query, (run_id,), fetch=True)

        if progress_details:
            return progress_details[0][0]
        return 0.0
