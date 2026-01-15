"""
RF-Inferno Database Manager.

Provides high-level interface for CRUD operations on the experiment database.
"""

import json
import os
from typing import Any

from rapidfireai.evals.db.db_interface import DatabaseInterface
from rapidfireai.evals.utils.constants import (
    ContextStatus,
    ExperimentStatus,
    PipelineStatus,
    TaskStatus,
)
from rapidfireai.evals.utils.serialize import (
    decode_db_payload,
    encode_payload,
    extract_pipeline_config_json,
)


class RFDatabase:
    """Database manager for RF-Inferno experiments."""

    def __init__(self, db_interface: DatabaseInterface = None):
        """
        Initialize the database manager.

        Args:
            db_interface: Optional DatabaseInterface instance. If not provided, creates a new one.
        """
        self.db = db_interface if db_interface else DatabaseInterface()
        self._initialize_schema()

    def _initialize_schema(self):
        """Initialize database schema from tables.sql file."""
        schema_path = os.path.join(os.path.dirname(__file__), "tables.sql")
        if os.path.exists(schema_path):
            with open(schema_path) as f:
                schema_sql = f.read()
                self.db.conn.executescript(schema_sql)
                self.db.conn.commit()
        
        # Migration: Add metric_run_id to pipelines table if they don't exist
        try:
            cursor = self.db.conn.execute("PRAGMA table_info(pipelines)")
            columns = [row[1] for row in cursor.fetchall()]
            if "metric_run_id" not in columns:
                self.db.conn.execute("ALTER TABLE pipelines ADD COLUMN metric_run_id TEXT")
                self.db.conn.commit()
        except Exception:
            pass
    
        # Migration: Add metric_experiment_id to experiments table if they don't exist
        try:
            cursor = self.db.conn.execute("PRAGMA table_info(experiments)")
            columns = [row[1] for row in cursor.fetchall()]
            if "metric_experiment_id" not in columns:
                self.db.conn.execute("ALTER TABLE experiments ADD COLUMN metric_experiment_id TEXT")
                self.db.conn.commit()
        except Exception:
            pass

    def close(self):
        """Close the database connection."""
        self.db.close()

    def create_tables(self):
        """Create database tables.

        Public method for Gunicorn on_starting() callback.
        Re-runs _initialize_schema() which uses CREATE TABLE IF NOT EXISTS,
        making it safe to call multiple times.
        """
        self._initialize_schema()

    # ============================================================================
    # EXPERIMENTS TABLE METHODS
    # ============================================================================

    def create_experiment(
        self,
        experiment_name: str,
        num_actors: int,
        num_cpus: int = None,
        num_gpus: int = None,
        metric_experiment_id: str = None,
        status: ExperimentStatus = ExperimentStatus.RUNNING,
        num_shards: int = 0,
    ) -> int:
        """
        Create a new experiment record.

        Args:
            experiment_name: Name of the experiment
            num_actors: Number of query processing actors
            num_cpus: Number of CPUs allocated
            num_gpus: Number of GPUs allocated
            metric_experiment_id: Optional MetricLogger experiment ID
            status: Initial status (default: ExperimentStatus.RUNNING)
            num_shards: Number of shards for the dataset (default: 0)

        Returns:
            experiment_id of the created experiment
        """
        query = """
        INSERT INTO experiments (
            experiment_name, num_actors, num_shards, num_cpus, num_gpus,
            metric_experiment_id, status, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, '')
        """
        self.db.execute(
            query,
            (
                experiment_name,
                num_actors,
                num_shards,
                num_cpus,
                num_gpus,
                metric_experiment_id,
                status.value,
            ),
            commit=True,
        )
        return self.db.cursor.lastrowid

    def set_experiment_status(self, experiment_id: int, status: ExperimentStatus):
        """
        Update experiment status.

        Args:
            experiment_id: ID of the experiment
            status: New status (ExperimentStatus enum)
        """
        query = "UPDATE experiments SET status = ? WHERE experiment_id = ?"
        self.db.execute(query, (status.value, experiment_id), commit=True)

    def set_experiment_error(self, experiment_id: int, error: str):
        """
        Set error message for an experiment.

        Args:
            experiment_id: ID of the experiment
            error: Error message
        """
        query = "UPDATE experiments SET error = ? WHERE experiment_id = ?"
        self.db.execute(query, (error, experiment_id), commit=True)

    def set_experiment_num_shards(self, experiment_id: int, num_shards: int):
        """
        Update the number of shards for an experiment.

        Args:
            experiment_id: ID of the experiment
            num_shards: Number of shards
        """
        query = "UPDATE experiments SET num_shards = ? WHERE experiment_id = ?"
        self.db.execute(query, (num_shards, experiment_id), commit=True)

    def set_experiment_resources(
        self,
        experiment_id: int,
        num_actors: int,
        num_cpus: int = None,
        num_gpus: int = None,
    ):
        """
        Update resource allocation for an experiment.

        Args:
            experiment_id: ID of the experiment
            num_actors: Number of actors
            num_cpus: Number of CPUs (optional)
            num_gpus: Number of GPUs (optional)
        """
        query = "UPDATE experiments SET num_actors = ?, num_cpus = ?, num_gpus = ? WHERE experiment_id = ?"
        self.db.execute(
            query, (num_actors, num_cpus, num_gpus, experiment_id), commit=True
        )

    def reset_all_tables(self, experiments_table: bool = False) -> None:
        """
        Clear data from experiment tables.

        Args:
            experiments_table: If True, also clear the experiments table (default: False)
        """
        # Clear dependent tables first (due to foreign keys)
        tables = ["actor_tasks", "contexts", "interactive_control", "pipelines"]

        for table in tables:
            self.db.execute(f"DELETE FROM {table}", commit=True)

        # Optionally clear experiments table
        if experiments_table:
            self.db.execute("DELETE FROM experiments", commit=True)
            tables.append("experiments")

        # Reset auto-increment indices
        for table in tables:
            self.db.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,), commit=True)

    def reset_experiment_states(self) -> None:
        """
        Reset the experiment states when a running task is cancelled.
        Marks ongoing/new pipelines as FAILED and their contexts as FAILED.
        Similar to fit mode's reset_experiment_states().
        """
        from rapidfireai.evals.utils.constants import ContextStatus, PipelineStatus, TaskStatus

        # Mark all scheduled and in-progress actor tasks as failed
        query = """
            UPDATE actor_tasks
            SET status = ?
            WHERE status = ? OR status = ?
        """
        self.db.execute(
            query, (TaskStatus.FAILED.value, TaskStatus.IN_PROGRESS.value, TaskStatus.SCHEDULED.value), commit=True
        )

        # Mark ongoing and new pipelines as failed
        query = """
            UPDATE pipelines
            SET status = ?
            WHERE status = ? OR status = ?
        """
        self.db.execute(
            query, (PipelineStatus.FAILED.value, PipelineStatus.ONGOING.value, PipelineStatus.NEW.value), commit=True
        )

        # Mark ongoing and new contexts as failed
        query = """
            UPDATE contexts
            SET status = ?
            WHERE status = ? OR status = ?
        """
        self.db.execute(
            query, (ContextStatus.FAILED.value, ContextStatus.ONGOING.value, ContextStatus.NEW.value), commit=True
        )

        # Reset all pending interactive control tasks
        query = """
            UPDATE interactive_control
            SET status = ?
            WHERE status = ?
        """
        self.db.execute(query, (TaskStatus.FAILED.value, TaskStatus.SCHEDULED.value), commit=True)

    def get_experiment(self, experiment_id: int) -> dict[str, Any] | None:
        """
        Get experiment details by ID.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Experiment dictionary with all fields, or None if not found
        """
        query = """
        SELECT experiment_id, experiment_name, num_actors, num_cpus, num_gpus,
               metric_experiment_id, status, num_shards, error, created_at
        FROM experiments
        WHERE experiment_id = ?
        """
        result = self.db.execute(query, params=(experiment_id,), fetch=True)
        if result and len(result) > 0:
            row = result[0]
            return {
                "experiment_id": row[0],
                "experiment_name": row[1],
                "num_actors": row[2],
                "num_cpus": row[3],
                "num_gpus": row[4],
                "metric_experiment_id": row[5],
                "status": row[6],
                "num_shards": row[7],
                "error": row[8],
                "created_at": row[9],
            }
        return None

    def get_experiment_error(self, experiment_id: int) -> str:
        """
        Get error message for an experiment.

        Args:
            experiment_id: ID of the experiment

        Returns:
            Error message string
        """
        query = "SELECT error FROM experiments WHERE experiment_id = ?"
        result = self.db.execute(query, (experiment_id,), fetch=True)
        return result[0][0] if result else ""

    def get_all_experiment_names(self) -> list[str]:
        """
        Get all experiment names.

        Returns:
            List of experiment names
        """
        query = "SELECT experiment_name FROM experiments"
        result = self.db.execute(query, fetch=True)
        return [row[0] for row in result] if result else []

    def get_running_experiment(self) -> dict[str, Any] | None:
        """
        Get the currently running experiment (most recent if multiple).

        Returns:
            Dictionary with all experiment fields, or None if no running experiment
        """
        query = """
        SELECT experiment_id, experiment_name, metric_experiment_id, num_shards,
               num_actors, num_cpus, num_gpus, status, error, created_at
        FROM experiments
        WHERE status = ?
        ORDER BY experiment_id DESC
        LIMIT 1
        """
        result = self.db.execute(query, (ExperimentStatus.RUNNING.value,), fetch=True)
        if result:
            row = result[0]
            return {
                "experiment_id": row[0],
                "experiment_name": row[1],
                "metric_experiment_id": row[2],
                "num_shards": row[3],
                "num_actors": row[4],
                "num_cpus": row[5],
                "num_gpus": row[6],
                "status": row[7],
                "error": row[8],
                "created_at": row[9],
            }
        return None

    # ============================================================================
    # CONTEXTS TABLE METHODS
    # ============================================================================

    def create_context(
        self,
        context_hash: str,
        rag_config_json: str = None,
        prompt_config_json: str = None,
        status: ContextStatus = ContextStatus.NEW,
    ) -> int:
        """
        Create a new context record.

        Args:
            context_hash: SHA256 hash of the context configuration
            rag_config_json: JSON string of RAG configuration
            prompt_config_json: JSON string of prompt manager configuration
            status: Initial status (default: ContextStatus.NEW)

        Returns:
            context_id of the created context (or existing if hash matches)
        """
        # Check if context with this hash already exists
        query = "SELECT context_id FROM contexts WHERE context_hash = ?"
        result = self.db.execute(query, (context_hash,), fetch=True)
        if result:
            return result[0][0]

        # Create new context
        query = """
        INSERT INTO contexts (context_hash, rag_config_json, prompt_config_json, status, error)
        VALUES (?, ?, ?, ?, '')
        """
        self.db.execute(
            query,
            (context_hash, rag_config_json, prompt_config_json, status.value),
            commit=True,
        )
        return self.db.cursor.lastrowid

    def get_context(self, context_id: int) -> dict[str, Any] | None:
        """
        Get context by ID.

        Args:
            context_id: ID of the context

        Returns:
            Dictionary with all context fields, or None if not found
        """
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
                "status": row[4],
                "error": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "duration_seconds": row[8],
            }
        return None

    def get_context_by_hash(self, context_hash: str) -> dict[str, Any] | None:
        """
        Get context by hash.

        Args:
            context_hash: SHA256 hash of the context configuration

        Returns:
            Dictionary with all context fields, or None if not found
        """
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
                "status": row[4],
                "error": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "duration_seconds": row[8],
            }
        return None

    def set_context_status(self, context_id: int, status: ContextStatus):
        """
        Update context status.

        Args:
            context_id: ID of the context
            status: New status (ContextStatus enum)
        """
        query = "UPDATE contexts SET status = ? WHERE context_id = ?"
        self.db.execute(query, (status.value, context_id), commit=True)

    def set_context_start_time(self, context_id: int, start_time: str):
        """
        Set start time for context building.

        Args:
            context_id: ID of the context
            start_time: Start timestamp (use datetime.now().isoformat())
        """
        query = "UPDATE contexts SET started_at = ? WHERE context_id = ?"
        self.db.execute(query, (start_time, context_id), commit=True)

    def set_context_end_time(
        self, context_id: int, end_time: str, duration_seconds: float
    ):
        """
        Set end time and duration for context building.

        Args:
            context_id: ID of the context
            end_time: End timestamp (use datetime.now().isoformat())
            duration_seconds: Duration in seconds
        """
        query = "UPDATE contexts SET completed_at = ?, duration_seconds = ? WHERE context_id = ?"
        self.db.execute(query, (end_time, duration_seconds, context_id), commit=True)

    def set_context_error(self, context_id: int, error: str):
        """
        Set error message for a context.

        Args:
            context_id: ID of the context
            error: Error message
        """
        query = "UPDATE contexts SET error = ? WHERE context_id = ?"
        self.db.execute(query, (error, context_id), commit=True)

    # ============================================================================
    # PIPELINES TABLE METHODS
    # ============================================================================

    def create_pipeline(
        self,
        pipeline_type: str,
        pipeline_config: Any,
        context_id: int = None,
        status: PipelineStatus = PipelineStatus.NEW,
        flattened_config: dict[str, Any] = None,
    ) -> int:
        """
        Create a new pipeline record.

        Args:
            pipeline_name: Name/identifier for the pipeline
            pipeline_type: Type of pipeline ('vllm', 'openai_api', etc.)
            pipeline_config: Pipeline configuration object (with classes/functions - will be encoded for pipeline_config column,
                            and JSON-serialized for pipeline_config_json column)
            context_id: Optional context ID for RAG
            status: Initial status (default: PipelineStatus.NEW)
            flattened_config: Flattened configuration dict for IC Ops panel display

        Returns:
            pipeline_id of the created pipeline
        """
        # Serialize the full pipeline config using encode_payload (includes functions/classes)
        encoded_config = encode_payload(pipeline_config)

        # Extract JSON-serializable data (excludes functions/classes)
        import json

        json_config_dict = extract_pipeline_config_json(pipeline_config)
        json_config_str = json.dumps(json_config_dict) if json_config_dict is not None else "{}"
        flattened_config_str = json.dumps(flattened_config) if flattened_config is not None else "{}"

        query = """
        INSERT INTO pipelines (
            context_id, pipeline_type,
            pipeline_config, pipeline_config_json, flattened_config, status, error,
            current_shard_id, shards_completed, total_samples_processed, metric_run_id
        ) VALUES (?, ?, ?, ?, ?, ?, '', '', 0, 0, NULL)
        """
        self.db.execute(
            query,
            (
                context_id,
                pipeline_type,
                encoded_config,
                json_config_str,
                flattened_config_str,
                status.value,
            ),
            commit=True,
        )
        return self.db.cursor.lastrowid

    def set_pipeline_progress(self, pipeline_id: int) -> dict[str, Any] | None:
        """
        Get pipeline by ID (legacy method name - actually gets pipeline, not sets progress).

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            Dictionary with all pipeline fields, or None if not found
        """
        query = """
        SELECT pipeline_id, context_id, pipeline_type,
               pipeline_config, pipeline_config_json, status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        WHERE pipeline_id = ?
        """
        result = self.db.execute(query, (pipeline_id,), fetch=True)
        if result:
            row = result[0]
            # Decode the pipeline config from the database (use pipeline_config column)
            decoded_config = decode_db_payload(row[3]) if row[3] else None
            # Parse JSON config for display/analytics
            import json

            json_config = json.loads(row[4]) if row[4] else None
            return {
                "pipeline_id": row[0],
                "context_id": row[1],
                "pipeline_type": row[2],
                "pipeline_config": decoded_config,  # Use decoded config for actual pipeline object
                "pipeline_config_json": json_config,  # JSON version for display/analytics
                "status": row[5],
                "current_shard_id": row[6],
                "shards_completed": row[7],
                "total_samples_processed": row[8],
                "metric_run_id": row[9],
                "error": row[10],
                "created_at": row[11],
            }
        return None

    def get_pipeline_by_metric_run_id(self, metric_run_id: str) -> dict[str, Any] | None:
        """
        Get pipeline by its metric_run_id (MLflow/Trackio run UUID).

        Args:
            metric_run_id: The metric tracking run ID (UUID string)

        Returns:
            Pipeline dictionary, or None if not found
        """
        query = """
        SELECT pipeline_id, context_id, pipeline_type,
               pipeline_config, pipeline_config_json, flattened_config, status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        WHERE metric_run_id = ?
        """
        result = self.db.execute(query, params=(metric_run_id,), fetch=True)
        if result and len(result) > 0:
            row = result[0]
            decoded_config = decode_db_payload(row[3]) if row[3] else None
            json_config = json.loads(row[4]) if row[4] else None
            flattened_config = json.loads(row[5]) if row[5] else {}
            return {
                "pipeline_id": row[0],
                "context_id": row[1],
                "pipeline_type": row[2],
                "pipeline_config": decoded_config,
                "pipeline_config_json": json_config,
                "flattened_config": flattened_config,
                "status": row[6],
                "current_shard_id": row[7],
                "shards_completed": row[8],
                "total_samples_processed": row[9],
                "metric_run_id": row[10],
                "error": row[11],
                "created_at": row[12],
            }
        return None

    def get_pipeline(self, pipeline_id: int | str) -> dict[str, Any] | None:
        """
        Get a single pipeline by ID.

        Args:
            pipeline_id: ID of the pipeline to retrieve

        Returns:
            Pipeline dictionary, or None if not found
        """
        query = """
        SELECT pipeline_id, context_id, pipeline_type,
               pipeline_config, pipeline_config_json, flattened_config,status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        WHERE pipeline_id = ?
        """
        pipeline = None
        if isinstance(pipeline_id, str):
            # Try as MLflow run ID (UUID string)
            pipeline = self.get_pipeline_by_metric_run_id(pipeline_id)
            # Fallback: try parsing as int
            if pipeline:
                return pipeline
        result = self.db.execute(query, params=(pipeline_id,), fetch=True)
        if result and len(result) > 0:
            row = result[0]
            # Decode the pipeline config from the database (use pipeline_config column)
            decoded_config = decode_db_payload(row[3]) if row[3] else None
            # Parse JSON config for display/analytics

            json_config = json.loads(row[4]) if row[4] else None
            flattened_config = json.loads(row[5]) if row[5] else {}
            return {
                "pipeline_id": row[0],
                "context_id": row[1],
                "pipeline_type": row[2],
                "pipeline_config": decoded_config,  # Use decoded config for actual pipeline object
                "pipeline_config_json": json_config,  # JSON version for display/analytics
                "flattened_config": flattened_config,  # Flattened configuration for IC Ops panel display
                "status": row[6],
                "current_shard_id": row[7],
                "shards_completed": row[8],
                "total_samples_processed": row[9],
                "metric_run_id": row[10],
                "error": row[11],
                "created_at": row[12],
            }
        return None

    def get_all_pipeline_ids(self) -> list[dict[str, Any]]:
        """
        Get lightweight list of all pipelines with minimal info (no config).

        Optimized for auto-polling - returns only IDs and status without deserializing configs.

        Returns:
            List of dicts with: pipeline_id, status, shards_completed, total_samples_processed
        """
        query = """
        SELECT pipeline_id, status, shards_completed, total_samples_processed
        FROM pipelines
        ORDER BY pipeline_id DESC
        """
        result = self.db.execute(query, fetch=True)
        pipelines = []
        if result:
            for row in result:
                pipelines.append(
                    {
                        "pipeline_id": row[0],
                        "status": row[1],
                        "shards_completed": row[2],
                        "total_samples_processed": row[3],
                    }
                )
        return pipelines

    def get_pipeline_config_json(self, pipeline_id: int) -> dict[str, Any] | None:
        """
        Get only the JSON config for a specific pipeline (for display/clone).

        Args:
            pipeline_id: ID of the pipeline

        Returns:
            Dictionary with pipeline_config_json, or None if not found
        """
        import json

        query = """
        SELECT pipeline_config_json, context_id
        FROM pipelines
        WHERE pipeline_id = ?
        """
        result = self.db.execute(query, (pipeline_id,), fetch=True)
        if result and result[0][0]:
            json_config = json.loads(result[0][0])
            return {
                "pipeline_config_json": json_config,
                "context_id": result[0][1]
            }
        return None

    def get_all_pipelines(self) -> list[dict[str, Any]]:
        """
        Get all pipelines, ordered by pipeline ID.

        Returns:
            List of pipeline dictionaries (ordered by pipeline_id DESC)
        """
        query = """
        SELECT pipeline_id, context_id, pipeline_type,
               pipeline_config, pipeline_config_json, flattened_config, status, current_shard_id,
               shards_completed, total_samples_processed, metric_run_id, error, created_at
        FROM pipelines
        ORDER BY pipeline_id DESC
        """
        result = self.db.execute(query, fetch=True)
        pipelines = []
        if result:
            import json

            for row in result:
                # Decode the pipeline config from the database (use pipeline_config column)
                decoded_config = decode_db_payload(row[3]) if row[3] else None
                # Parse JSON config for display/analytics
                json_config = json.loads(row[4]) if row[4] else None
                flattened_config = json.loads(row[5]) if row[5] else {}
                pipelines.append(
                    {
                        "pipeline_id": row[0],
                        "context_id": row[1],
                        "pipeline_type": row[2],
                        "pipeline_config": decoded_config,  # Use decoded config for actual pipeline object
                        "pipeline_config_json": json_config,     # JSON version for display/analytics
                        "flattened_config": flattened_config,  # Flattened configuration for IC Ops panel display
                        "status": row[6],
                        "current_shard_id": row[7],
                        "shards_completed": row[8],
                        "total_samples_processed": row[9],
                        "metric_run_id": row[10],
                        "error": row[11],
                        "created_at": row[12],
                    }
                )
        return pipelines

    def set_pipeline_status(self, pipeline_id: int, status: PipelineStatus):
        """
        Update pipeline status.

        Args:
            pipeline_id: ID of the pipeline
            status: New status (PipelineStatus enum)
        """
        query = "UPDATE pipelines SET status = ? WHERE pipeline_id = ?"
        self.db.execute(query, (status.value, pipeline_id), commit=True)

    def set_pipeline_progress(
        self,
        pipeline_id: int,
        current_shard_id: int,
        shards_completed: int,
        total_samples_processed: int,
    ):
        """
        Update pipeline progress metrics.

        Args:
            pipeline_id: ID of the pipeline
            current_shard_id: Current shard being processed
            shards_completed: Number of shards completed
            total_samples_processed: Total number of samples processed
        """
        query = """
        UPDATE pipelines
        SET current_shard_id = ?, shards_completed = ?, total_samples_processed = ?
        WHERE pipeline_id = ?
        """
        self.db.execute(
            query,
            (current_shard_id, shards_completed, total_samples_processed, pipeline_id),
            commit=True,
        )

    def set_pipeline_current_shard(self, pipeline_id: int, shard_id: int):
        """
        Update the current shard being processed by a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            shard_id: Current shard ID being processed
        """
        query = "UPDATE pipelines SET current_shard_id = ? WHERE pipeline_id = ?"
        self.db.execute(query, (shard_id, pipeline_id), commit=True)

    def set_pipeline_error(self, pipeline_id: int, error: str):
        """
        Set error message for a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            error: Error message
        """
        query = "UPDATE pipelines SET error = ? WHERE pipeline_id = ?"
        self.db.execute(query, (error, pipeline_id), commit=True)

    def set_pipeline_metric_run_id(self, pipeline_id: int, metric_run_id: str):
        """
        Set MetricLogger run ID for a pipeline.

        Args:
            pipeline_id: ID of the pipeline
            metric_run_id: MetricLogger run ID
        """
        query = "UPDATE pipelines SET metric_run_id = ? WHERE pipeline_id = ?"
        self.db.execute(query, (metric_run_id, pipeline_id), commit=True)

    # ============================================================================
    # ACTOR_TASKS TABLE METHODS
    # ============================================================================

    def create_actor_task(
        self,
        pipeline_id: int,
        actor_id: int,
        shard_id: int,
        status: TaskStatus = TaskStatus.SCHEDULED,
    ) -> int:
        """
        Create a new actor task record.

        Args:
            pipeline_id: ID of the pipeline
            actor_id: ID of the actor
            shard_id: ID of the shard
            status: Initial status (default: TaskStatus.SCHEDULED)

        Returns:
            task_id of the created task
        """
        query = """
        INSERT INTO actor_tasks (
            pipeline_id, actor_id, shard_id, status, error_message
        ) VALUES (?, ?, ?, ?, '')
        """
        self.db.execute(
            query, (pipeline_id, actor_id, shard_id, status.value), commit=True
        )
        return self.db.cursor.lastrowid

    def get_actor_task(self, task_id: int) -> dict[str, Any] | None:
        """
        Get actor task by ID.

        Args:
            task_id: ID of the task

        Returns:
            Dictionary with all task fields, or None if not found
        """
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
                "status": row[4],
                "error_message": row[5],
                "started_at": row[6],
                "completed_at": row[7],
                "duration_seconds": row[8],
            }
        return None

    def get_running_actor_tasks(self) -> list[dict[str, Any]]:
        """
        Get all currently running actor tasks.

        Returns:
            List of running task dictionaries (ordered by task_id DESC)
        """
        query = """
        SELECT task_id, pipeline_id, actor_id, shard_id, status,
               error_message, started_at, completed_at, duration_seconds
        FROM actor_tasks
        WHERE status = ?
        ORDER BY task_id DESC
        """
        result = self.db.execute(query, (TaskStatus.IN_PROGRESS.value,), fetch=True)
        tasks = []
        if result:
            for row in result:
                tasks.append(
                    {
                        "task_id": row[0],
                        "pipeline_id": row[1],
                        "actor_id": row[2],
                        "shard_id": row[3],
                        "status": row[4],
                        "error_message": row[5],
                        "started_at": row[6],
                        "completed_at": row[7],
                        "duration_seconds": row[8],
                    }
                )
        return tasks

    def set_actor_task_status(self, task_id: int, status: TaskStatus):
        """
        Update actor task status.

        Args:
            task_id: ID of the task
            status: New status (TaskStatus enum)
        """
        query = "UPDATE actor_tasks SET status = ? WHERE task_id = ?"
        self.db.execute(query, (status.value, task_id), commit=True)

    def set_actor_task_start_time(self, task_id: int, start_time: str):
        """
        Set start time for an actor task.

        Args:
            task_id: ID of the task
            start_time: Start timestamp (use datetime.now().isoformat())
        """
        query = "UPDATE actor_tasks SET started_at = ? WHERE task_id = ?"
        self.db.execute(query, (start_time, task_id), commit=True)

    def set_actor_task_end_time(
        self, task_id: int, end_time: str, duration_seconds: float
    ):
        """
        Set end time and duration for an actor task.

        Args:
            task_id: ID of the task
            end_time: End timestamp (use datetime.now().isoformat())
            duration_seconds: Duration in seconds
        """
        query = "UPDATE actor_tasks SET completed_at = ?, duration_seconds = ? WHERE task_id = ?"
        self.db.execute(query, (end_time, duration_seconds, task_id), commit=True)

    def set_actor_task_error(self, task_id: int, error_message: str):
        """
        Set error message for an actor task.

        Args:
            task_id: ID of the task
            error_message: Error message
        """
        query = "UPDATE actor_tasks SET error_message = ? WHERE task_id = ?"
        self.db.execute(query, (error_message, task_id), commit=True)

    # ============================================================================
    # INTERACTIVE_CONTROL TABLE METHODS
    # ============================================================================

    def create_ic_operation(
        self,
        operation: str,
        pipeline_id: int = None,
        request_data: str = None,
    ) -> int:
        """
        Create a new interactive control operation.

        Args:
            operation: Type of operation ('stop', 'resume', 'delete', 'clone')
            pipeline_id: ID of the pipeline (None for clone operation)
            request_data: JSON string with operation data (e.g., model_config for clone)

        Returns:
            ic_id of the created operation
        """
        import time

        query = """
        INSERT INTO interactive_control (
            pipeline_id, operation, status, request_data, error, created_at
        ) VALUES (?, ?, ?, ?, '', ?)
        """
        from rapidfireai.evals.utils.constants import ICStatus

        self.db.execute(
            query,
            (pipeline_id, operation, ICStatus.PENDING.value, request_data, time.time()),
            commit=True,
        )
        return self.db.cursor.lastrowid

    def get_pending_ic_operations(self) -> list[dict[str, Any]]:
        """
        Get all pending IC operations.

        Returns:
            List of dictionaries with IC operation fields
        """
        from rapidfireai.evals.utils.constants import ICStatus

        query = """
        SELECT ic_id, pipeline_id, operation, status, request_data, error, created_at, processed_at
        FROM interactive_control
        WHERE status = ?
        ORDER BY created_at ASC
        """
        results = self.db.execute(query, (ICStatus.PENDING.value,), fetch=True)
        operations = []
        for row in results:
            operations.append(
                {
                    "ic_id": row[0],
                    "pipeline_id": row[1],
                    "operation": row[2],
                    "status": row[3],
                    "request_data": row[4],
                    "error": row[5],
                    "created_at": row[6],
                    "processed_at": row[7],
                }
            )
        return operations

    def get_ic_operation(self, ic_id: int) -> dict[str, Any] | None:
        """
        Get IC operation by ID.

        Args:
            ic_id: ID of the IC operation

        Returns:
            Dictionary with IC operation fields, or None if not found
        """
        query = """
        SELECT ic_id, pipeline_id, operation, status, request_data, error, created_at, processed_at
        FROM interactive_control
        WHERE ic_id = ?
        """
        result = self.db.execute(query, (ic_id,), fetch=True)
        if result:
            row = result[0]
            return {
                "ic_id": row[0],
                "pipeline_id": row[1],
                "operation": row[2],
                "status": row[3],
                "request_data": row[4],
                "error": row[5],
                "created_at": row[6],
                "processed_at": row[7],
            }
        return None

    def update_ic_operation_status(self, ic_id: int, status: str, error: str = ""):
        """
        Update IC operation status.

        Args:
            ic_id: ID of the IC operation
            status: New status ('pending', 'processing', 'completed', 'failed')
            error: Error message (if status is 'failed')
        """
        import time

        query = """
        UPDATE interactive_control
        SET status = ?, error = ?, processed_at = ?
        WHERE ic_id = ?
        """
        self.db.execute(query, (status, error, time.time(), ic_id), commit=True)

    def get_all_ic_operations(self) -> list[dict[str, Any]]:
        """
        Get all IC operations, ordered by creation time.

        Returns:
            List of dictionaries with IC operation fields
        """
        query = """
        SELECT ic_id, pipeline_id, operation, status, request_data, error, created_at, processed_at
        FROM interactive_control
        ORDER BY created_at DESC
        """
        results = self.db.execute(query, fetch=True)
        operations = []
        for row in results:
            operations.append(
                {
                    "ic_id": row[0],
                    "pipeline_id": row[1],
                    "operation": row[2],
                    "status": row[3],
                    "request_data": row[4],
                    "error": row[5],
                    "created_at": row[6],
                    "processed_at": row[7],
                }
            )
        return operations


# Export for external use
__all__ = ["RFDatabase"]