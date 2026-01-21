"""
RapidFire AI Database Manager

Provides high-level interface for CRUD operations on the experiment database.
Handles experiments, runs, pipelines, contexts, and interactive control operations.
"""

import json
import os
from typing import Any

from rapidfireai.db.db_interface import DatabaseInterface
from rapidfireai.utils.constants import (
    ContextStatus,
    ControllerTask,
    ExperimentStatus,
    ExperimentTask,
    ICOperation,
    ICStatus,
    PipelineStatus,
    RunEndedBy,
    RunSource,
    RunStatus,
    TaskStatus,
    WorkerTask,
)


def encode_payload(payload: object) -> str:
    """Encode the payload for the database using dill."""
    import base64
    import dill
    return base64.b64encode(dill.dumps(payload)).decode("utf-8")


def decode_db_payload(payload: str) -> object:
    """Decode the payload from the database using dill."""
    import base64
    import dill
    return dill.loads(base64.b64decode(payload))


class RfDb:
    """
    Database manager for RapidFire AI experiments.
    
    Provides CRUD operations for experiments, runs (training), pipelines (inference),
    contexts (RAG), and interactive control operations.
    """

    def __init__(self):
        """Initialize the database manager and create tables if needed."""
        self.db = DatabaseInterface()
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize database schema from tables.sql file."""
        schema_path = os.path.join(os.path.dirname(__file__), "tables.sql")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                schema_sql = f.read()
                self.db.conn.executescript(schema_sql)
                self.db.conn.commit()

        # Run migrations for any schema changes
        self._run_migrations()

    def _run_migrations(self):
        """Run any necessary schema migrations."""
        # Migration: Add experiments_path to experiments table if it doesn't exist
        try:
            cursor = self.db.conn.execute("PRAGMA table_info(experiments)")
            columns = [row[1] for row in cursor.fetchall()]
            
            if "experiments_path" not in columns:
                self.db.conn.execute("ALTER TABLE experiments ADD COLUMN experiments_path TEXT DEFAULT ''")
                self.db.conn.commit()
            
            if "metric_experiment_id" not in columns:
                self.db.conn.execute("ALTER TABLE experiments ADD COLUMN metric_experiment_id TEXT")
                self.db.conn.commit()
                
            if "config" not in columns:
                self.db.conn.execute("ALTER TABLE experiments ADD COLUMN config TEXT DEFAULT '{}'")
                self.db.conn.commit()
        except Exception:
            pass

        # Migration: Add metric_run_id to runs table
        try:
            cursor = self.db.conn.execute("PRAGMA table_info(runs)")
            columns = [row[1] for row in cursor.fetchall()]
            if "metric_run_id" not in columns:
                self.db.conn.execute("ALTER TABLE runs ADD COLUMN metric_run_id TEXT")
                self.db.conn.commit()
        except Exception:
            pass

        # Migration: Add metric_run_id to pipelines table
        try:
            cursor = self.db.conn.execute("PRAGMA table_info(pipelines)")
            columns = [row[1] for row in cursor.fetchall()]
            if "metric_run_id" not in columns:
                self.db.conn.execute("ALTER TABLE pipelines ADD COLUMN metric_run_id TEXT")
                self.db.conn.commit()
        except Exception:
            pass

    def create_tables(self):
        """
        Create database tables (public method for external callers).
        Re-runs _initialize_schema() which uses CREATE TABLE IF NOT EXISTS.
        """
        self._initialize_schema()

    def close(self):
        """Close the database connection."""
        self.db.close()

    # ============================================================================
    # EXPERIMENTS TABLE METHODS (Shared)
    # ============================================================================

    def create_experiment(
        self,
        experiment_name: str,
        experiments_path: str,
        metric_experiment_id: str | None = None,
        status: ExperimentStatus = ExperimentStatus.RUNNING,
        config: dict[str, Any] | None = None,
    ) -> int:
        """
        Create a new experiment record.

        Args:
            experiment_name: Name of the experiment
            experiments_path: Path to experiment artifacts directory
            metric_experiment_id: Optional MLflow/TensorBoard experiment ID
            status: Initial status (default: ExperimentStatus.RUNNING)
            config: Optional configuration dict (stored as JSON)

        Returns:
            experiment_id of the created experiment
        """
        config_json = json.dumps(config) if config else "{}"
        
        query = """
        INSERT INTO experiments (
            experiment_name, experiments_path, metric_experiment_id,
            status, current_task, config, error
        ) VALUES (?, ?, ?, ?, ?, ?, '')
        """
        self.db.execute(
            query,
            (
                experiment_name,
                experiments_path,
                metric_experiment_id,
                status.value,
                ExperimentTask.IDLE.value,
                config_json,
            ),
            commit=True,
        )
        
        # Optimize periodically
        self.db.optimize_periodically()
        
        return self.db.cursor.lastrowid

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        """
        Get experiment details by ID.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment dictionary with all fields, or None if not found
        """
        query = """
        SELECT experiment_id, experiment_name, experiments_path, metric_experiment_id,
               status, error, created_at, current_task, config
        FROM experiments
        WHERE experiment_id = ?
        """
        result = self.db.execute(query, (experiment_id,), fetch=True)
        if result and len(result) > 0:
            row = result[0]
            return {
                "experiment_id": row[0],
                "experiment_name": row[1],
                "experiments_path": row[2],
                "metric_experiment_id": row[3],
                "status": ExperimentStatus(row[4]),
                "error": row[5],
                "created_at": row[6],
                "current_task": ExperimentTask(row[7]) if row[7] else ExperimentTask.IDLE,
                "config": json.loads(row[8]) if row[8] else {},
            }
        return None

    def get_running_experiment(self) -> dict[str, Any]:
        """
        Get the currently running experiment (most recent if multiple).

        Returns:
            Dictionary with experiment fields

        Raises:
            Exception: If no running experiment found
        """
        query = """
        SELECT experiment_id, experiment_name, experiments_path, metric_experiment_id,
               status, error, created_at, current_task, config
        FROM experiments
        WHERE status = ?
        ORDER BY experiment_id DESC
        LIMIT 1
        """
        result = self.db.execute(query, (ExperimentStatus.RUNNING.value,), fetch=True)
        if result and len(result) > 0:
            row = result[0]
            return {
                "experiment_id": row[0],
                "experiment_name": row[1],
                "experiments_path": row[2],
                "metric_experiment_id": row[3],
                "status": ExperimentStatus(row[4]),
                "error": row[5],
                "created_at": row[6],
                "current_task": ExperimentTask(row[7]) if row[7] else ExperimentTask.IDLE,
                "config": json.loads(row[8]) if row[8] else {},
            }
        raise Exception("No running experiment found")

    def get_experiment_status(self) -> ExperimentStatus | None:
        """Get the status of the most recent experiment."""
        query = """
        SELECT status
        FROM experiments
        ORDER BY experiment_id DESC
        LIMIT 1
        """
        result = self.db.execute(query, fetch=True)
        if result:
            return ExperimentStatus(result[0][0])
        return None

    def set_experiment_status(self, experiment_id: int, status: ExperimentStatus) -> None:
        """Set the status of an experiment."""
        query = "UPDATE experiments SET status = ? WHERE experiment_id = ?"
        self.db.execute(query, (status.value, experiment_id), commit=True)

    def set_experiment_error(self, experiment_id: int, error: str) -> None:
        """Set the error message for an experiment."""
        query = "UPDATE experiments SET error = ? WHERE experiment_id = ?"
        self.db.execute(query, (error, experiment_id), commit=True)

    def get_experiment_error(self, experiment_id: int) -> str:
        """Get the error message for an experiment."""
        query = "SELECT error FROM experiments WHERE experiment_id = ?"
        result = self.db.execute(query, (experiment_id,), fetch=True)
        return result[0][0] if result else ""

    def get_all_experiment_names(self) -> list[str]:
        """Get all experiment names."""
        query = "SELECT experiment_name FROM experiments"
        result = self.db.execute(query, fetch=True)
        return [row[0] for row in result] if result else []

    def get_experiments_path(self, experiment_id: int) -> str:
        """Get the experiments path for a given experiment."""
        query = "SELECT experiments_path FROM experiments WHERE experiment_id = ?"
        result = self.db.execute(query, (experiment_id,), fetch=True)
        if result:
            return result[0][0]
        raise Exception("Experiments path not found")

    # --- Experiment Config Methods ---

    def get_experiment_config(self, experiment_id: int) -> dict[str, Any]:
        """Get the config dict for an experiment."""
        query = "SELECT config FROM experiments WHERE experiment_id = ?"
        result = self.db.execute(query, (experiment_id,), fetch=True)
        return json.loads(result[0][0]) if result and result[0][0] else {}

    def update_experiment_config(self, experiment_id: int, **kwargs) -> None:
        """Update specific keys in the experiment config."""
        config = self.get_experiment_config(experiment_id)
        config.update(kwargs)
        query = "UPDATE experiments SET config = ? WHERE experiment_id = ?"
        self.db.execute(query, (json.dumps(config), experiment_id), commit=True)

    # --- Fit-specific Experiment Methods ---

    def set_experiment_current_task(self, task: ExperimentTask) -> None:
        """Set the current task of the running experiment (fit mode)."""
        query = "UPDATE experiments SET current_task = ? WHERE status = ?"
        self.db.execute(query, (task.value, ExperimentStatus.RUNNING.value), commit=True)

    def get_experiment_current_task(self) -> ExperimentTask:
        """Get the current task of the running experiment (fit mode)."""
        query = """
        SELECT current_task
        FROM experiments
        WHERE status = ?
        ORDER BY experiment_id DESC
        LIMIT 1
        """
        result = self.db.execute(query, (ExperimentStatus.RUNNING.value,), fetch=True)
        if result:
            return ExperimentTask(result[0][0]) if result[0][0] else ExperimentTask.IDLE
        raise Exception("No running experiment found")

    # --- Evals-specific Experiment Methods ---

    def set_experiment_num_shards(self, experiment_id: int, num_shards: int) -> None:
        """Update the number of shards for an experiment (evals mode)."""
        self.update_experiment_config(experiment_id, num_shards=num_shards)

    def set_experiment_resources(
        self,
        experiment_id: int,
        num_actors: int,
        num_cpus: float = None,
        num_gpus: float = None,
    ) -> None:
        """Update resource allocation for an experiment (evals mode)."""
        self.update_experiment_config(
            experiment_id,
            num_actors=num_actors,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
        )

    # ============================================================================
    # TABLE RESET METHODS
    # ============================================================================

    def reset_all_tables(self, experiments_table: bool = False) -> None:
        """
        Clear data from experiment tables.

        Args:
            experiments_table: If True, also clear the experiments table
        """
        # Clear tables in order (respecting foreign keys)
        tables = [
            "controller_progress",
            "worker_progress",
            "worker_task",
            "actor_tasks",
            "interactive_control",
            "pipelines",
            "contexts",
            "runs",
        ]

        if experiments_table:
            tables.append("experiments")

        for table in tables:
            try:
                self.db.execute(f"DELETE FROM {table}", commit=True)
            except Exception:
                pass  # Table might not exist

        # Reset auto-increment indices
        for table in tables:
            try:
                self.db.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,), commit=True)
            except Exception:
                pass

    def reset_experiment_states(self) -> None:
        """
        Reset experiment states when a running task is cancelled.
        Marks ongoing tasks as failed for both fit and evals modes.
        """
        # Mark all scheduled and in-progress worker tasks as failed (fit)
        query = """
        UPDATE worker_task
        SET status = ?
        WHERE status = ? OR status = ?
        """
        try:
            self.db.execute(
                query,
                (TaskStatus.FAILED.value, TaskStatus.IN_PROGRESS.value, TaskStatus.SCHEDULED.value),
                commit=True,
            )
        except Exception:
            pass

        # Mark all scheduled and in-progress actor tasks as failed (evals)
        try:
            self.db.execute(
                query.replace("worker_task", "actor_tasks"),
                (TaskStatus.FAILED.value, TaskStatus.IN_PROGRESS.value, TaskStatus.SCHEDULED.value),
                commit=True,
            )
        except Exception:
            pass

        # Mark ongoing and new runs as failed (fit)
        try:
            run_query = """
            UPDATE runs
            SET status = ?
            WHERE status = ? OR status = ?
            """
            self.db.execute(
                run_query,
                (RunStatus.FAILED.value, RunStatus.ONGOING.value, RunStatus.NEW.value),
                commit=True,
            )
        except Exception:
            pass

        # Mark ongoing and new pipelines as failed (evals)
        try:
            pipeline_query = """
            UPDATE pipelines
            SET status = ?
            WHERE status = ? OR status = ?
            """
            self.db.execute(
                pipeline_query,
                (PipelineStatus.FAILED.value, PipelineStatus.ONGOING.value, PipelineStatus.NEW.value),
                commit=True,
            )
        except Exception:
            pass

        # Mark ongoing and new contexts as failed (evals)
        try:
            context_query = """
            UPDATE contexts
            SET status = ?
            WHERE status = ? OR status = ?
            """
            self.db.execute(
                context_query,
                (ContextStatus.FAILED.value, ContextStatus.ONGOING.value, ContextStatus.NEW.value),
                commit=True,
            )
        except Exception:
            pass

        # Reset all pending interactive control tasks
        try:
            ic_query = """
            UPDATE interactive_control
            SET status = ?
            WHERE status = ?
            """
            self.db.execute(
                ic_query,
                (ICStatus.SKIPPED.value, ICStatus.PENDING.value),
                commit=True,
            )
        except Exception:
            pass

        # Reset progress tables (fit)
        for table in ["controller_progress", "worker_progress"]:
            try:
                self.db.execute(f"DELETE FROM {table}", commit=True)
            except Exception:
                pass

    # ============================================================================
    # INTERACTIVE CONTROL TABLE METHODS (Unified)
    # ============================================================================

    def create_ic_operation(
        self,
        target_id: int,
        target_type: str,
        operation: str | ICOperation,
        config_data: dict[str, Any] | str | None = None,
    ) -> int:
        """
        Create a new interactive control operation.

        Args:
            target_id: run_id (fit) or pipeline_id (evals)
            target_type: 'run' or 'pipeline'
            operation: Operation type (stop, resume, delete, clone, clone_warm)
            config_data: JSON config (config_leaf for fit, request_data for evals)

        Returns:
            ic_id of the created operation
        """
        if isinstance(operation, ICOperation):
            operation = operation.value
            
        if isinstance(config_data, dict):
            config_data = json.dumps(config_data)
        elif config_data is None:
            config_data = "{}"

        query = """
        INSERT INTO interactive_control (
            target_id, target_type, operation, config_data, status
        ) VALUES (?, ?, ?, ?, ?)
        """
        self.db.execute(
            query,
            (target_id, target_type, operation, config_data, ICStatus.PENDING.value),
            commit=True,
        )
        return self.db.cursor.lastrowid

    def get_pending_ic_operations(self, target_type: str = None) -> list[dict[str, Any]]:
        """
        Get all pending IC operations.

        Args:
            target_type: Optional filter for 'run' or 'pipeline'

        Returns:
            List of IC operation dictionaries
        """
        if target_type:
            query = """
            SELECT ic_id, target_id, target_type, operation, config_data, status, error, created_at, processed_at
            FROM interactive_control
            WHERE status = ? AND target_type = ?
            ORDER BY created_at ASC
            """
            results = self.db.execute(query, (ICStatus.PENDING.value, target_type), fetch=True)
        else:
            query = """
            SELECT ic_id, target_id, target_type, operation, config_data, status, error, created_at, processed_at
            FROM interactive_control
            WHERE status = ?
            ORDER BY created_at ASC
            """
            results = self.db.execute(query, (ICStatus.PENDING.value,), fetch=True)

        operations = []
        for row in results or []:
            operations.append({
                "ic_id": row[0],
                "target_id": row[1],
                "target_type": row[2],
                "operation": row[3],
                "config_data": json.loads(row[4]) if row[4] else {},
                "status": row[5],
                "error": row[6],
                "created_at": row[7],
                "processed_at": row[8],
            })
        return operations

    def get_ic_operation(self, ic_id: int) -> dict[str, Any] | None:
        """Get IC operation by ID."""
        query = """
        SELECT ic_id, target_id, target_type, operation, config_data, status, error, created_at, processed_at
        FROM interactive_control
        WHERE ic_id = ?
        """
        result = self.db.execute(query, (ic_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "ic_id": row[0],
                "target_id": row[1],
                "target_type": row[2],
                "operation": row[3],
                "config_data": json.loads(row[4]) if row[4] else {},
                "status": row[5],
                "error": row[6],
                "created_at": row[7],
                "processed_at": row[8],
            }
        return None

    def update_ic_operation_status(self, ic_id: int, status: str | ICStatus, error: str = "") -> None:
        """Update IC operation status."""
        import time
        
        if isinstance(status, ICStatus):
            status = status.value
            
        query = """
        UPDATE interactive_control
        SET status = ?, error = ?, processed_at = ?
        WHERE ic_id = ?
        """
        self.db.execute(query, (status, error, time.time(), ic_id), commit=True)

    def get_all_ic_operations(self) -> list[dict[str, Any]]:
        """Get all IC operations."""
        query = """
        SELECT ic_id, target_id, target_type, operation, config_data, status, error, created_at, processed_at
        FROM interactive_control
        ORDER BY created_at DESC
        """
        results = self.db.execute(query, fetch=True)
        operations = []
        for row in results or []:
            operations.append({
                "ic_id": row[0],
                "target_id": row[1],
                "target_type": row[2],
                "operation": row[3],
                "config_data": json.loads(row[4]) if row[4] else {},
                "status": row[5],
                "error": row[6],
                "created_at": row[7],
                "processed_at": row[8],
            })
        return operations

    # ============================================================================
    # FIT MODE: RUNS TABLE METHODS
    # ============================================================================

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
        """Create a new run (fit mode)."""
        query = """
        INSERT INTO runs (
            status, metric_run_id, flattened_config, config_leaf,
            completed_steps, total_steps, num_chunks_visited_curr_epoch,
            num_epochs_completed, chunk_offset, error, source, ended_by,
            warm_started_from, cloned_from
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        raise Exception("Failed to create run")

    def get_run(self, run_id: int) -> dict[str, Any]:
        """Get a run's details (fit mode)."""
        query = """
        SELECT status, metric_run_id, flattened_config, config_leaf,
               completed_steps, total_steps, num_chunks_visited_curr_epoch,
               num_epochs_completed, chunk_offset, error, source, ended_by,
               warm_started_from, cloned_from
        FROM runs
        WHERE run_id = ?
        """
        result = self.db.execute(query, (run_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "status": RunStatus(row[0]),
                "metric_run_id": row[1],
                "flattened_config": json.loads(row[2]) if row[2] else {},
                "config_leaf": decode_db_payload(row[3]) if row[3] and row[3] != "{}" else {},
                "completed_steps": row[4],
                "total_steps": row[5],
                "num_chunks_visited_curr_epoch": row[6],
                "num_epochs_completed": row[7],
                "chunk_offset": row[8],
                "error": row[9],
                "source": RunSource(row[10]) if row[10] else None,
                "ended_by": RunEndedBy(row[11]) if row[11] else None,
                "warm_started_from": row[12],
                "cloned_from": row[13],
            }
        raise Exception("Run not found")

    def get_all_runs(self) -> dict[int, dict[str, Any]]:
        """Get all runs (fit mode)."""
        query = """
        SELECT run_id, status, metric_run_id, flattened_config, config_leaf,
               completed_steps, total_steps, num_chunks_visited_curr_epoch,
               num_epochs_completed, chunk_offset, error, source, ended_by,
               warm_started_from, cloned_from
        FROM runs
        """
        results = self.db.execute(query, fetch=True)
        formatted = {}
        if results:
            for row in results:
                formatted[row[0]] = {
                    "status": RunStatus(row[1]),
                    "metric_run_id": row[2],
                    "flattened_config": json.loads(row[3]) if row[3] else {},
                    "config_leaf": decode_db_payload(row[4]) if row[4] and row[4] != "{}" else {},
                    "completed_steps": row[5],
                    "total_steps": row[6],
                    "num_chunks_visited_curr_epoch": row[7],
                    "num_epochs_completed": row[8],
                    "chunk_offset": row[9],
                    "error": row[10],
                    "source": RunSource(row[11]) if row[11] else None,
                    "ended_by": RunEndedBy(row[12]) if row[12] else None,
                    "warm_started_from": row[13],
                    "cloned_from": row[14],
                }
        return formatted

    def get_runs_by_status(self, statuses: list[RunStatus]) -> dict[int, dict[str, Any]]:
        """Get all runs by statuses (fit mode)."""
        if not statuses:
            return {}

        placeholders = ",".join(["?"] * len(statuses))
        query = f"""
        SELECT run_id, status, metric_run_id, flattened_config, config_leaf,
               completed_steps, total_steps, num_chunks_visited_curr_epoch,
               num_epochs_completed, chunk_offset, error, source, ended_by,
               warm_started_from, cloned_from
        FROM runs
        WHERE status IN ({placeholders})
        """
        status_values = [s.value for s in statuses]
        results = self.db.execute(query, status_values, fetch=True)
        formatted = {}
        if results:
            for row in results:
                formatted[row[0]] = {
                    "status": RunStatus(row[1]),
                    "metric_run_id": row[2],
                    "flattened_config": json.loads(row[3]) if row[3] else {},
                    "config_leaf": decode_db_payload(row[4]) if row[4] and row[4] != "{}" else {},
                    "completed_steps": row[5],
                    "total_steps": row[6],
                    "num_chunks_visited_curr_epoch": row[7],
                    "num_epochs_completed": row[8],
                    "chunk_offset": row[9],
                    "error": row[10],
                    "source": RunSource(row[11]) if row[11] else None,
                    "ended_by": RunEndedBy(row[12]) if row[12] else None,
                    "warm_started_from": row[13],
                    "cloned_from": row[14],
                }
        return formatted

    def set_run_status(self, run_id: int, status: RunStatus) -> None:
        """Set the status of a run (fit mode)."""
        query = "UPDATE runs SET status = ? WHERE run_id = ?"
        self.db.execute(query, (status.value, run_id), commit=True)

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
        """Set details of an existing run (fit mode)."""
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

        columns = {k: v for k, v in columns.items() if v is not None}
        if not columns:
            return

        query_parts = [f"{col} = ?" for col in columns]
        values = list(columns.values())
        values.append(run_id)

        query = f"UPDATE runs SET {', '.join(query_parts)} WHERE run_id = ?"
        self.db.execute(query, tuple(values), commit=True)

    def set_completed_steps(self, run_id: int, completed_steps: int) -> None:
        """Set the completed steps for a run (fit mode)."""
        query = "UPDATE runs SET completed_steps = ? WHERE run_id = ?"
        self.db.execute(query, (completed_steps, run_id), commit=True)

    def get_completed_steps(self, run_id: int) -> int:
        """Get the completed steps for a run (fit mode)."""
        query = "SELECT completed_steps FROM runs WHERE run_id = ?"
        result = self.db.execute(query, (run_id,), fetch=True)
        if result:
            return result[0][0]
        raise Exception("Run not found")

    # ============================================================================
    # FIT MODE: WORKER TASK TABLE METHODS
    # ============================================================================

    def create_worker_task(
        self,
        worker_id: int,
        task_type: WorkerTask,
        status: TaskStatus,
        run_id: int,
        chunk_id: int = -1,
        config_options: dict[str, Any] | None = None,
    ) -> int:
        """Create a worker task (fit mode)."""
        query = """
        INSERT INTO worker_task (worker_id, task_type, status, run_id, chunk_id, config_options)
        VALUES (?, ?, ?, ?, ?, ?)
        """
        config_str = encode_payload(config_options) if config_options else "{}"
        self.db.execute(
            query,
            (worker_id, task_type.value, status.value, run_id, chunk_id, config_str),
            commit=True,
        )
        result = self.db.execute("SELECT last_insert_rowid()", fetch=True)
        if result:
            return result[0][0]
        raise Exception("Failed to create worker task")

    def get_worker_scheduled_task(self, worker_id: int) -> dict[str, Any]:
        """Get the latest scheduled task for a worker (fit mode)."""
        query = """
        SELECT task_id, task_type, run_id, chunk_id, config_options
        FROM worker_task
        WHERE worker_id = ? AND status = ?
        ORDER BY task_id DESC
        LIMIT 1
        """
        result = self.db.execute(query, (worker_id, TaskStatus.SCHEDULED.value), fetch=True)
        if result:
            row = result[0]
            return {
                "task_id": row[0],
                "task_type": WorkerTask(row[1]),
                "run_id": row[2],
                "chunk_id": row[3],
                "config_options": decode_db_payload(row[4]) if row[4] and row[4] != "{}" else {},
            }
        return {}

    def get_all_worker_tasks(self) -> dict[int, dict[str, Any]]:
        """Get the latest task of each worker (fit mode)."""
        query = """
        SELECT worker_id, task_id, task_type, status, run_id, chunk_id, config_options
        FROM worker_task wt1
        WHERE task_id = (
            SELECT MAX(task_id)
            FROM worker_task wt2
            WHERE wt2.worker_id = wt1.worker_id
        )
        """
        results = self.db.execute(query, fetch=True)
        formatted = {}
        if results:
            for row in results:
                formatted[row[0]] = {
                    "task_id": row[1],
                    "task_type": WorkerTask(row[2]),
                    "status": TaskStatus(row[3]),
                    "run_id": row[4],
                    "chunk_id": row[5],
                    "config_options": decode_db_payload(row[6]) if row[6] and row[6] != "{}" else {},
                }
        return formatted

    def set_worker_task_status(self, worker_id: int, status: TaskStatus) -> None:
        """Set the status of the latest task of a worker (fit mode)."""
        query = """
        UPDATE worker_task
        SET status = ?
        WHERE task_id = (
            SELECT task_id FROM worker_task
            WHERE worker_id = ?
            ORDER BY task_id DESC
            LIMIT 1
        )
        """
        self.db.execute(query, (status.value, worker_id), commit=True)

    # ============================================================================
    # FIT MODE: PROGRESS TABLE METHODS
    # ============================================================================

    def set_controller_progress(self, run_id: int, progress: float) -> None:
        """Set the controller progress for a run (fit mode)."""
        query = """
        INSERT INTO controller_progress (run_id, progress)
        VALUES (?, ?)
        ON CONFLICT (run_id)
        DO UPDATE SET progress = EXCLUDED.progress
        """
        self.db.execute(query, (run_id, round(progress, 2)), commit=True)

    def get_controller_progress(self, run_id: int) -> float:
        """Get the controller progress for a run (fit mode)."""
        query = "SELECT progress FROM controller_progress WHERE run_id = ?"
        result = self.db.execute(query, (run_id,), fetch=True)
        return result[0][0] if result else 0.0

    def set_worker_progress(self, run_id: int, subchunk_progress: float) -> None:
        """Set the worker progress for a run (fit mode)."""
        query = """
        INSERT INTO worker_progress (run_id, subchunk_progress)
        VALUES (?, ?)
        ON CONFLICT (run_id)
        DO UPDATE SET subchunk_progress = EXCLUDED.subchunk_progress
        """
        self.db.execute(query, (run_id, round(subchunk_progress, 2)), commit=True)

    def get_worker_progress(self, run_id: int) -> float:
        """Get the worker progress for a run (fit mode)."""
        query = "SELECT subchunk_progress FROM worker_progress WHERE run_id = ?"
        result = self.db.execute(query, (run_id,), fetch=True)
        return result[0][0] if result else 0.0

    # ============================================================================
    # EVALS MODE: CONTEXTS TABLE METHODS
    # ============================================================================

    def create_context(
        self,
        context_hash: str,
        rag_config_json: str = None,
        prompt_config_json: str = None,
        status: ContextStatus = ContextStatus.NEW,
    ) -> int:
        """Create a new context record (evals mode)."""
        # Check if context with this hash already exists
        query = "SELECT context_id FROM contexts WHERE context_hash = ?"
        result = self.db.execute(query, (context_hash,), fetch=True)
        if result:
            return result[0][0]

        query = """
        INSERT INTO contexts (context_hash, rag_config_json, prompt_config_json, status, error)
        VALUES (?, ?, ?, ?, '')
        """
        self.db.execute(query, (context_hash, rag_config_json, prompt_config_json, status.value), commit=True)
        return self.db.cursor.lastrowid

    def get_context(self, context_id: int) -> dict[str, Any] | None:
        """Get context by ID (evals mode)."""
        query = """
        SELECT context_id, context_hash, rag_config_json, prompt_config_json,
               status, error, started_at, completed_at, duration_seconds
        FROM contexts
        WHERE context_id = ?
        """
        result = self.db.execute(query, (context_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "context_id": row[0],
                "context_hash": row[1],
                "rag_config_json": row[2],
                "prompt_config_json": row[3],
                "status": ContextStatus(row[4]),
                "error": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "duration_seconds": row[8],
            }
        return None

    def get_context_by_hash(self, context_hash: str) -> dict[str, Any] | None:
        """Get context by hash (evals mode)."""
        query = """
        SELECT context_id, context_hash, rag_config_json, prompt_config_json,
               status, error, started_at, completed_at, duration_seconds
        FROM contexts
        WHERE context_hash = ?
        """
        result = self.db.execute(query, (context_hash,), fetch=True)
        if result:
            row = result[0]
            return {
                "context_id": row[0],
                "context_hash": row[1],
                "rag_config_json": row[2],
                "prompt_config_json": row[3],
                "status": ContextStatus(row[4]),
                "error": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "duration_seconds": row[8],
            }
        return None

    def set_context_status(self, context_id: int, status: ContextStatus) -> None:
        """Update context status (evals mode)."""
        query = "UPDATE contexts SET status = ? WHERE context_id = ?"
        self.db.execute(query, (status.value, context_id), commit=True)

    def set_context_start_time(self, context_id: int, start_time: str) -> None:
        """Set start time for context building (evals mode)."""
        query = "UPDATE contexts SET started_at = ? WHERE context_id = ?"
        self.db.execute(query, (start_time, context_id), commit=True)

    def set_context_end_time(self, context_id: int, end_time: str, duration_seconds: float) -> None:
        """Set end time and duration for context building (evals mode)."""
        query = "UPDATE contexts SET completed_at = ?, duration_seconds = ? WHERE context_id = ?"
        self.db.execute(query, (end_time, duration_seconds, context_id), commit=True)

    def set_context_error(self, context_id: int, error: str) -> None:
        """Set error message for a context (evals mode)."""
        query = "UPDATE contexts SET error = ? WHERE context_id = ?"
        self.db.execute(query, (error, context_id), commit=True)

    # ============================================================================
    # EVALS MODE: PIPELINES TABLE METHODS
    # ============================================================================

    def create_pipeline(
        self,
        pipeline_type: str,
        pipeline_config: Any,
        context_id: int = None,
        status: PipelineStatus = PipelineStatus.NEW,
        flattened_config: dict[str, Any] = None,
    ) -> int:
        """Create a new pipeline record (evals mode)."""
        encoded_config = encode_payload(pipeline_config)

        # Extract JSON-serializable data
        from rapidfireai.utils.serialize import extract_pipeline_config_json
        json_config_dict = extract_pipeline_config_json(pipeline_config)
        json_config_str = json.dumps(json_config_dict) if json_config_dict else "{}"
        flattened_config_str = json.dumps(flattened_config) if flattened_config else "{}"

        query = """
        INSERT INTO pipelines (
            context_id, pipeline_type, pipeline_config, pipeline_config_json,
            flattened_config, status, error, current_shard_id, shards_completed,
            total_samples_processed, metric_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, '', 0, 0, 0, NULL)
        """
        self.db.execute(
            query,
            (context_id, pipeline_type, encoded_config, json_config_str, flattened_config_str, status.value),
            commit=True,
        )
        return self.db.cursor.lastrowid

    def get_pipeline(self, pipeline_id: int | str) -> dict[str, Any] | None:
        """Get a single pipeline by ID (evals mode)."""
        # Try as metric_run_id first if it's a string
        if isinstance(pipeline_id, str):
            pipeline = self.get_pipeline_by_metric_run_id(pipeline_id)
            if pipeline:
                return pipeline

        query = """
        SELECT pipeline_id, context_id, pipeline_type, pipeline_config,
               pipeline_config_json, flattened_config, status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        WHERE pipeline_id = ?
        """
        result = self.db.execute(query, (pipeline_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "pipeline_id": row[0],
                "context_id": row[1],
                "pipeline_type": row[2],
                "pipeline_config": decode_db_payload(row[3]) if row[3] else None,
                "pipeline_config_json": json.loads(row[4]) if row[4] else None,
                "flattened_config": json.loads(row[5]) if row[5] else {},
                "status": PipelineStatus(row[6]),
                "current_shard_id": row[7],
                "shards_completed": row[8],
                "total_samples_processed": row[9],
                "metric_run_id": row[10],
                "error": row[11],
                "created_at": row[12],
            }
        return None

    def get_pipeline_by_metric_run_id(self, metric_run_id: str) -> dict[str, Any] | None:
        """Get pipeline by its metric_run_id (evals mode)."""
        query = """
        SELECT pipeline_id, context_id, pipeline_type, pipeline_config,
               pipeline_config_json, flattened_config, status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        WHERE metric_run_id = ?
        """
        result = self.db.execute(query, (metric_run_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "pipeline_id": row[0],
                "context_id": row[1],
                "pipeline_type": row[2],
                "pipeline_config": decode_db_payload(row[3]) if row[3] else None,
                "pipeline_config_json": json.loads(row[4]) if row[4] else None,
                "flattened_config": json.loads(row[5]) if row[5] else {},
                "status": PipelineStatus(row[6]),
                "current_shard_id": row[7],
                "shards_completed": row[8],
                "total_samples_processed": row[9],
                "metric_run_id": row[10],
                "error": row[11],
                "created_at": row[12],
            }
        return None

    def get_all_pipelines(self) -> list[dict[str, Any]]:
        """Get all pipelines (evals mode)."""
        query = """
        SELECT pipeline_id, context_id, pipeline_type, pipeline_config,
               pipeline_config_json, flattened_config, status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        ORDER BY pipeline_id DESC
        """
        results = self.db.execute(query, fetch=True)
        pipelines = []
        if results:
            for row in results:
                pipelines.append({
                    "pipeline_id": row[0],
                    "context_id": row[1],
                    "pipeline_type": row[2],
                    "pipeline_config": decode_db_payload(row[3]) if row[3] else None,
                    "pipeline_config_json": json.loads(row[4]) if row[4] else None,
                    "flattened_config": json.loads(row[5]) if row[5] else {},
                    "status": PipelineStatus(row[6]),
                    "current_shard_id": row[7],
                    "shards_completed": row[8],
                    "total_samples_processed": row[9],
                    "metric_run_id": row[10],
                    "error": row[11],
                    "created_at": row[12],
                })
        return pipelines

    def get_all_pipeline_ids(self) -> list[dict[str, Any]]:
        """Get lightweight list of pipeline IDs with minimal info (evals mode)."""
        query = """
        SELECT pipeline_id, status, shards_completed, total_samples_processed
        FROM pipelines
        ORDER BY pipeline_id DESC
        """
        results = self.db.execute(query, fetch=True)
        pipelines = []
        if results:
            for row in results:
                pipelines.append({
                    "pipeline_id": row[0],
                    "status": row[1],
                    "shards_completed": row[2],
                    "total_samples_processed": row[3],
                })
        return pipelines

    def get_pipeline_config_json(self, pipeline_id: int) -> dict[str, Any] | None:
        """Get only the config JSON for a specific pipeline (evals mode)."""
        query = "SELECT pipeline_config_json, context_id FROM pipelines WHERE pipeline_id = ?"
        result = self.db.execute(query, (pipeline_id,), fetch=True)
        if result and result[0][0]:
            return {
                "pipeline_config_json": json.loads(result[0][0]),
                "context_id": result[0][1],
            }
        return None

    def set_pipeline_status(self, pipeline_id: int, status: PipelineStatus) -> None:
        """Update pipeline status (evals mode)."""
        query = "UPDATE pipelines SET status = ? WHERE pipeline_id = ?"
        self.db.execute(query, (status.value, pipeline_id), commit=True)

    def set_pipeline_progress(
        self,
        pipeline_id: int,
        current_shard_id: int,
        shards_completed: int,
        total_samples_processed: int,
    ) -> None:
        """Update pipeline progress metrics (evals mode)."""
        query = """
        UPDATE pipelines
        SET current_shard_id = ?, shards_completed = ?, total_samples_processed = ?
        WHERE pipeline_id = ?
        """
        self.db.execute(query, (current_shard_id, shards_completed, total_samples_processed, pipeline_id), commit=True)

    def set_pipeline_current_shard(self, pipeline_id: int, shard_id: int) -> None:
        """Update the current shard being processed by a pipeline (evals mode)."""
        query = "UPDATE pipelines SET current_shard_id = ? WHERE pipeline_id = ?"
        self.db.execute(query, (shard_id, pipeline_id), commit=True)

    def set_pipeline_error(self, pipeline_id: int, error: str) -> None:
        """Set error message for a pipeline (evals mode)."""
        query = "UPDATE pipelines SET error = ? WHERE pipeline_id = ?"
        self.db.execute(query, (error, pipeline_id), commit=True)

    def set_pipeline_metric_run_id(self, pipeline_id: int, metric_run_id: str) -> None:
        """Set MetricLogger run ID for a pipeline (evals mode)."""
        query = "UPDATE pipelines SET metric_run_id = ? WHERE pipeline_id = ?"
        self.db.execute(query, (metric_run_id, pipeline_id), commit=True)

    # ============================================================================
    # EVALS MODE: ACTOR TASKS TABLE METHODS
    # ============================================================================

    def create_actor_task(
        self,
        pipeline_id: int,
        actor_id: int,
        shard_id: int,
        status: TaskStatus = TaskStatus.SCHEDULED,
    ) -> int:
        """Create a new actor task record (evals mode)."""
        query = """
        INSERT INTO actor_tasks (pipeline_id, actor_id, shard_id, status, error_message)
        VALUES (?, ?, ?, ?, '')
        """
        self.db.execute(query, (pipeline_id, actor_id, shard_id, status.value), commit=True)
        return self.db.cursor.lastrowid

    def get_actor_task(self, task_id: int) -> dict[str, Any] | None:
        """Get actor task by ID (evals mode)."""
        query = """
        SELECT task_id, pipeline_id, actor_id, shard_id, status,
               error_message, started_at, completed_at, duration_seconds
        FROM actor_tasks
        WHERE task_id = ?
        """
        result = self.db.execute(query, (task_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "task_id": row[0],
                "pipeline_id": row[1],
                "actor_id": row[2],
                "shard_id": row[3],
                "status": TaskStatus(row[4]),
                "error_message": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "duration_seconds": row[8],
            }
        return None

    def get_running_actor_tasks(self) -> list[dict[str, Any]]:
        """Get all currently running actor tasks (evals mode)."""
        query = """
        SELECT task_id, pipeline_id, actor_id, shard_id, status,
               error_message, started_at, completed_at, duration_seconds
        FROM actor_tasks
        WHERE status = ?
        ORDER BY task_id DESC
        """
        results = self.db.execute(query, (TaskStatus.IN_PROGRESS.value,), fetch=True)
        tasks = []
        if results:
            for row in results:
                tasks.append({
                    "task_id": row[0],
                    "pipeline_id": row[1],
                    "actor_id": row[2],
                    "shard_id": row[3],
                    "status": TaskStatus(row[4]),
                    "error_message": row[5],
                    "started_at": row[6],
                    "completed_at": row[7],
                    "duration_seconds": row[8],
                })
        return tasks

    def set_actor_task_status(self, task_id: int, status: TaskStatus) -> None:
        """Update actor task status (evals mode)."""
        query = "UPDATE actor_tasks SET status = ? WHERE task_id = ?"
        self.db.execute(query, (status.value, task_id), commit=True)

    def set_actor_task_start_time(self, task_id: int, start_time: str) -> None:
        """Set start time for an actor task (evals mode)."""
        query = "UPDATE actor_tasks SET started_at = ? WHERE task_id = ?"
        self.db.execute(query, (start_time, task_id), commit=True)

    def set_actor_task_end_time(self, task_id: int, end_time: str, duration_seconds: float) -> None:
        """Set end time and duration for an actor task (evals mode)."""
        query = "UPDATE actor_tasks SET completed_at = ?, duration_seconds = ? WHERE task_id = ?"
        self.db.execute(query, (end_time, duration_seconds, task_id), commit=True)

    def set_actor_task_error(self, task_id: int, error_message: str) -> None:
        """Set error message for an actor task (evals mode)."""
        query = "UPDATE actor_tasks SET error_message = ? WHERE task_id = ?"
        self.db.execute(query, (error_message, task_id), commit=True)


# Backwards compatibility aliases
RFDatabase = RfDb  # Evals used this name

__all__ = ["RfDb", "RFDatabase", "encode_payload", "decode_db_payload"]
