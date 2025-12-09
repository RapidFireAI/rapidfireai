"""
Dispatcher REST API for Interactive Control of RF-Inferno Experiments.

Provides HTTP endpoints for dynamic pipeline management during experiment execution.
FIXED: Now properly handles CORS preflight (OPTIONS) requests for VS Code/Cursor webview.
"""

import json
import logging
import threading
import traceback

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from waitress import serve

from rapidfireai.evals.db import RFDatabase
from rapidfireai.utils.constants import DispatcherConfig
from rapidfireai.utils.colab import is_running_in_colab
from rapidfireai.evals.utils.constants import ICOperation

CORS_ALLOWED_ORIGINS = "*" # Allow all origins

class Dispatcher:
    """
    REST API server for interactive control of running experiments.

    Handles user requests to:
    - Stop pipelines (pause execution, can be resumed)
    - Resume pipelines (continue from where stopped)
    - Delete pipelines (permanently remove)
    - Clone pipelines (create new pipeline with existing context)
    """

    def __init__(self) -> None:
        """Initialize the Dispatcher with database connection and Flask app."""
        # Create database handle
        self.db: RFDatabase = RFDatabase()

        # Create Flask app
        self.app: Flask = Flask(__name__)

        # Enable CORS for local development
        # Dispatcher runs on localhost, safe to allow all origins
        # supports_credentials=True is required for Colab proxy auth (credentials: 'include' in JS)
        _ = CORS(
            self.app,
            resources={
                r"/*": {
                    "origins": CORS_ALLOWED_ORIGINS,
                    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"],
                    "expose_headers": ["Content-Type"],
                    "supports_credentials": True if is_running_in_colab() else False,
                }
            },
        )

        # Register routes
        self.register_routes()

    def register_routes(self) -> None:
        """Register all REST API routes with OPTIONS support for CORS preflight."""
        route_prefix = "/dispatcher"

        # CRITICAL: Add before_request handler to handle OPTIONS preflight requests globally
        @self.app.before_request
        def handle_preflight():
            if request.method == "OPTIONS":
                response = jsonify({})
                response.headers.add("Access-Control-Allow-Origin", CORS_ALLOWED_ORIGINS)
                response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
                response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
                response.headers.add("Access-Control-Max-Age", "3600")
                return response

        # Health check
        self.app.add_url_rule(
            f"{route_prefix}/health-check", "health_check", self.health_check, methods=["GET", "OPTIONS"]
        )

        # Interactive control operations
        self.app.add_url_rule(
            f"{route_prefix}/stop-pipeline", "stop_pipeline", self.stop_pipeline, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/resume-pipeline", "resume_pipeline", self.resume_pipeline, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/delete-pipeline", "delete_pipeline", self.delete_pipeline, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/clone-pipeline", "clone_pipeline", self.clone_pipeline, methods=["POST", "OPTIONS"]
        )

        # Status queries
        self.app.add_url_rule(
            f"{route_prefix}/operation-status/<int:ic_id>",
            "get_operation_status",
            self.get_operation_status,
            methods=["GET", "OPTIONS"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/all-operations", "get_all_operations", self.get_all_operations, methods=["GET", "OPTIONS"]
        )

        # Pipeline queries (for UI)
        self.app.add_url_rule(
            f"{route_prefix}/list-all-pipeline-ids",
            "list_all_pipeline_ids",
            self.list_all_pipeline_ids,
            methods=["GET", "OPTIONS"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-pipeline-config-json/<int:pipeline_id>",
            "get_pipeline_config_json",
            self.get_pipeline_config_json,
            methods=["GET", "OPTIONS"],
        )
        # Legacy endpoints (kept for backwards compatibility)
        self.app.add_url_rule(
            f"{route_prefix}/get-all-pipelines", "get_all_pipelines", self.get_all_pipelines, methods=["GET", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-pipeline", "get_pipeline", self.get_pipeline, methods=["POST", "OPTIONS"]
        )

        # Frontend-compatible routes (map "run" terminology to "pipeline" for dashboard compatibility)
        # These allow the fit frontend to work with evals dispatcher
        self.app.add_url_rule(
            f"{route_prefix}/get-all-runs", "get_all_runs", self.get_all_runs, methods=["GET", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-run", "get_run", self.get_run, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/stop-run", "stop_run", self.stop_run, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/resume-run", "resume_run", self.resume_run, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/delete-run", "delete_run", self.delete_run, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/clone-modify-run", "clone_modify_run", self.clone_modify_run, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-running-experiment", "get_running_experiment", self.get_running_experiment, methods=["GET", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-all-experiment-names", "get_all_experiment_names", self.get_all_experiment_names, methods=["GET", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-experiment-logs", "get_experiment_logs", self.get_experiment_logs, methods=["POST", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-ic-logs", "get_ic_logs", self.get_ic_logs, methods=["POST", "OPTIONS"]
        )

    def health_check(self) -> tuple[Response, int]:
        """Health check endpoint."""
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        try:
            return jsonify({"status": "ok", "message": "Dispatcher is running"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def stop_pipeline(self) -> tuple[Response, int]:
        """
        Stop a running pipeline.

        Request body:
            {
                "pipeline_id": int
            }

        Returns:
            ic_id of the created operation
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            pipeline_id = data.get("pipeline_id")
            if pipeline_id is None:
                return jsonify({"error": "pipeline_id is required"}), 400

            # Validate pipeline exists
            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.STOP.value,
                pipeline_id=pipeline_id,
            )

            return jsonify({"ic_id": ic_id, "message": f"Stop request created for pipeline {pipeline_id}"}), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def resume_pipeline(self) -> tuple[Response, int]:
        """
        Resume a stopped pipeline.

        Request body:
            {
                "pipeline_id": int
            }

        Returns:
            ic_id of the created operation
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            pipeline_id = data.get("pipeline_id")
            if pipeline_id is None:
                return jsonify({"error": "pipeline_id is required"}), 400

            # Validate pipeline exists
            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.RESUME.value,
                pipeline_id=pipeline_id,
            )

            return jsonify({"ic_id": ic_id, "message": f"Resume request created for pipeline {pipeline_id}"}), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def delete_pipeline(self) -> tuple[Response, int]:
        """
        Delete a pipeline permanently.

        Request body:
            {
                "pipeline_id": int
            }

        Returns:
            ic_id of the created operation
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            pipeline_id = data.get("pipeline_id")
            if pipeline_id is None:
                return jsonify({"error": "pipeline_id is required"}), 400

            # Validate pipeline exists
            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.DELETE.value,
                pipeline_id=pipeline_id,
            )

            return jsonify({"ic_id": ic_id, "message": f"Delete request created for pipeline {pipeline_id}"}), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def clone_pipeline(self) -> tuple[Response, int]:
        """
        Clone a new pipeline from a parent pipeline with edited configuration.

        The clone inherits the parent's context_id, RAG, and prompt_manager.
        Only the JSON-editable parameters can be modified.

        Request body:
            {
                "parent_pipeline_id": int,  # ID of the pipeline to clone
                "config_json": {            # Edited configuration
                    "pipeline_type": "vllm" | "openai",
                    "model_config": {...},
                    "sampling_params": {...},  # for vLLM
                    "client_config": {...},    # for OpenAI
                    "batch_size": int,         # optional
                    "online_strategy_kwargs": {...}  # optional
                }
            }

        Returns:
            ic_id of the created operation
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            parent_pipeline_id = data.get("parent_pipeline_id")
            if parent_pipeline_id is None:
                return jsonify({"error": "parent_pipeline_id is required"}), 400

            config_json = data.get("config_json")
            if not config_json:
                return jsonify({"error": "config_json is required"}), 400

            # Validate parent pipeline exists
            parent_pipeline = self.db.get_pipeline(parent_pipeline_id)
            if not parent_pipeline:
                return jsonify({"error": f"Parent pipeline {parent_pipeline_id} not found"}), 404

            # Validate config_json has required fields
            pipeline_type = config_json.get("pipeline_type")
            if not pipeline_type:
                return jsonify({"error": "config_json must include 'pipeline_type'"}), 400

            if pipeline_type.lower() not in ["vllm", "openai"]:
                return jsonify({"error": "pipeline_type must be 'vllm' or 'openai'"}), 400

            # Type-specific validation
            if pipeline_type.lower() == "vllm":
                if "model_config" not in config_json or "sampling_params" not in config_json:
                    return jsonify({"error": "vLLM pipelines require 'model_config' and 'sampling_params'"}), 400

            elif pipeline_type.lower() == "openai":
                if "client_config" not in config_json or "model_config" not in config_json:
                    return jsonify({"error": "OpenAI pipelines require 'client_config' and 'model_config'"}), 400

            # Prepare request data for IC operation
            request_data = {
                "parent_pipeline_id": parent_pipeline_id,
                "config_json": config_json,
            }

            # Create IC operation (pipeline_id is None for CLONE, as new ID will be generated)
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.CLONE.value,
                pipeline_id=None,
                request_data=json.dumps(request_data),
            )

            return (
                jsonify(
                    {
                        "ic_id": ic_id,
                        "message": f"Clone request created from parent pipeline {parent_pipeline_id}",
                    }
                ),
                200,
            )

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_operation_status(self, ic_id: int) -> tuple[Response, int]:
        """
        Get status of a specific IC operation.

        Args:
            ic_id: ID of the IC operation

        Returns:
            Operation details including status
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            operation = self.db.get_ic_operation(ic_id)
            if not operation:
                return jsonify({"error": f"Operation {ic_id} not found"}), 404

            return jsonify(operation), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_all_operations(self) -> tuple[Response, int]:
        """
        Get all IC operations (for monitoring/debugging).

        Returns:
            List of all operations
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            operations = self.db.get_all_ic_operations()
            return jsonify(operations), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def list_all_pipeline_ids(self) -> tuple[Response, int]:
        """
        Get lightweight list of pipeline IDs with minimal info (optimized for polling).

        Returns:
            List of pipelines with only: pipeline_id, status, shards_completed, total_samples_processed
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            pipelines = self.db.get_all_pipeline_ids()
            return jsonify(pipelines), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_pipeline_config_json(self, pipeline_id: int) -> tuple[Response, int]:
        """
        Get only the config JSON for a specific pipeline (for clone operations).

        Args:
            pipeline_id: ID of the pipeline (from URL path)

        Returns:
            Pipeline config JSON
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            config_data = self.db.get_pipeline_config_json(pipeline_id)
            if not config_data:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            return jsonify(config_data), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_all_pipelines(self) -> tuple[Response, int]:
        """
        Get all pipelines (for UI dropdown).

        LEGACY: Use list_all_pipeline_ids() for better performance.

        Returns:
            List of all pipelines
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            pipelines = self.db.get_all_pipelines()
            return jsonify(pipelines), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_pipeline(self) -> tuple[Response, int]:
        """
        Get details of a specific pipeline.

        Request body:
            {
                "pipeline_id": int
            }

        Returns:
            Pipeline details
        """
        # Handle OPTIONS preflight
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            pipeline_id = data.get("pipeline_id")
            if pipeline_id is None:
                return jsonify({"error": "pipeline_id is required"}), 400

            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            return jsonify(pipeline), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # =========================================================================
    # Frontend-compatible methods (map "run" terminology to "pipeline")
    # These allow the fit frontend dashboard to work with evals mode
    # =========================================================================

    def _pipeline_to_run_format(self, pipeline: dict) -> dict:
        """
        Convert a pipeline dict to run format for frontend compatibility.

        Maps pipeline fields to the format expected by the fit frontend.
        """
        # Parse config JSON if it's a string
        config = {}
        if pipeline.get("pipeline_config_json"):
            try:
                config = json.loads(pipeline["pipeline_config_json"]) if isinstance(
                    pipeline["pipeline_config_json"], str
                ) else pipeline["pipeline_config_json"]
            except (json.JSONDecodeError, TypeError):
                config = {}

        # Map pipeline status to run status format
        status_map = {
            "new": "NEW",
            "ongoing": "ONGOING",
            "completed": "COMPLETED",
            "stopped": "STOPPED",
            "deleted": "DELETED",
            "failed": "FAILED",
        }
        status = status_map.get(pipeline.get("status", "").lower(), pipeline.get("status", ""))

        return {
            "run_id": pipeline.get("pipeline_id"),
            "status": status,
            "mlflow_run_id": pipeline.get("mlflow_run_id"),
            "config": config,
            "flattened_config": config,  # Same as config for evals
            "completed_steps": pipeline.get("shards_completed", 0),
            "total_steps": pipeline.get("total_samples_processed", 0),
            "num_epochs_completed": 0,  # Not applicable for evals
            "error": pipeline.get("error", ""),
            "source": "USER",
            "ended_by": None,
            # Evals-specific fields (for frontend to identify mode)
            "pipeline_type": pipeline.get("pipeline_type"),
            "context_id": pipeline.get("context_id"),
        }

    def get_all_runs(self) -> tuple[Response, int]:
        """
        Get all pipelines formatted as runs for frontend compatibility.

        Returns pipelines in the format expected by the fit frontend.
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            pipelines = self.db.get_all_pipelines()
            runs = [self._pipeline_to_run_format(p) for p in pipelines]
            return jsonify(runs), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_run(self) -> tuple[Response, int]:
        """
        Get a single pipeline formatted as run for frontend compatibility.

        Request body: {"run_id": int}
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            run_id = data.get("run_id")
            if run_id is None:
                return jsonify({"error": "run_id is required"}), 400

            pipeline = self.db.get_pipeline(run_id)
            if not pipeline:
                return jsonify({"error": f"Run {run_id} not found"}), 404

            run = self._pipeline_to_run_format(pipeline)
            return jsonify(run), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def stop_run(self) -> tuple[Response, int]:
        """
        Stop a pipeline (frontend-compatible alias for stop_pipeline).

        Request body: {"run_id": int}
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            run_id = data.get("run_id")
            if run_id is None:
                return jsonify({"error": "run_id is required"}), 400

            # Validate pipeline exists
            pipeline = self.db.get_pipeline(run_id)
            if not pipeline:
                return jsonify({"error": f"Run {run_id} not found"}), 404

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.STOP.value,
                pipeline_id=run_id,
            )

            return jsonify({"ic_id": ic_id, "message": f"Stop request created for run {run_id}"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def resume_run(self) -> tuple[Response, int]:
        """
        Resume a stopped pipeline (frontend-compatible alias for resume_pipeline).

        Request body: {"run_id": int}
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            run_id = data.get("run_id")
            if run_id is None:
                return jsonify({"error": "run_id is required"}), 400

            # Validate pipeline exists
            pipeline = self.db.get_pipeline(run_id)
            if not pipeline:
                return jsonify({"error": f"Run {run_id} not found"}), 404

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.RESUME.value,
                pipeline_id=run_id,
            )

            return jsonify({"ic_id": ic_id, "message": f"Resume request created for run {run_id}"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def delete_run(self) -> tuple[Response, int]:
        """
        Delete a pipeline (frontend-compatible alias for delete_pipeline).

        Request body: {"run_id": int}
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            run_id = data.get("run_id")
            if run_id is None:
                return jsonify({"error": "run_id is required"}), 400

            # Validate pipeline exists
            pipeline = self.db.get_pipeline(run_id)
            if not pipeline:
                return jsonify({"error": f"Run {run_id} not found"}), 404

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.DELETE.value,
                pipeline_id=run_id,
            )

            return jsonify({"ic_id": ic_id, "message": f"Delete request created for run {run_id}"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def clone_modify_run(self) -> tuple[Response, int]:
        """
        Clone a pipeline with modifications (frontend-compatible alias for clone_pipeline).

        Request body:
            {
                "run_id": int,  # ID of the pipeline to clone
                "config_leaf": {...},  # Modified configuration
                "warm_start": bool  # Not used in evals, but accepted for compatibility
            }
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            run_id = data.get("run_id")
            if run_id is None:
                return jsonify({"error": "run_id is required"}), 400

            config_leaf = data.get("config_leaf", {})

            # Validate parent pipeline exists
            parent_pipeline = self.db.get_pipeline(run_id)
            if not parent_pipeline:
                return jsonify({"error": f"Run {run_id} not found"}), 404

            # Get current config and merge with modifications
            current_config = {}
            if parent_pipeline.get("pipeline_config_json"):
                try:
                    current_config = json.loads(parent_pipeline["pipeline_config_json"]) if isinstance(
                        parent_pipeline["pipeline_config_json"], str
                    ) else parent_pipeline["pipeline_config_json"]
                except (json.JSONDecodeError, TypeError):
                    current_config = {}

            # Merge config_leaf into current config
            merged_config = {**current_config, **config_leaf}

            # Ensure pipeline_type is present
            if "pipeline_type" not in merged_config:
                merged_config["pipeline_type"] = parent_pipeline.get("pipeline_type", "vllm")

            # Prepare request data for IC operation
            request_data = {
                "parent_pipeline_id": run_id,
                "config_json": merged_config,
            }

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.CLONE.value,
                pipeline_id=None,  # New pipeline will be created
                request_data=json.dumps(request_data),
            )

            return jsonify({
                "ic_id": ic_id,
                "message": f"Clone-modify request created for run {run_id}",
            }), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_running_experiment(self) -> tuple[Response, int]:
        """
        Get the currently running experiment info for frontend compatibility.

        Returns basic experiment info that the frontend expects.
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            # Get the most recent experiment from the database
            experiments = self.db.get_all_experiments()
            if not experiments:
                return jsonify({"error": "No experiment found"}), 404

            # Return the most recent experiment
            latest = experiments[-1] if isinstance(experiments, list) else experiments
            return jsonify({
                "experiment_id": latest.get("experiment_id"),
                "experiment_name": latest.get("experiment_name", "Evals Experiment"),
                "status": latest.get("status", "running"),
                "mlflow_experiment_id": latest.get("mlflow_experiment_id"),
            }), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_all_experiment_names(self) -> tuple[Response, int]:
        """
        Get all experiment names for frontend compatibility.
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            experiments = self.db.get_all_experiments()
            names = [exp.get("experiment_name", f"Experiment {exp.get('experiment_id')}") for exp in experiments]
            return jsonify(names), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_experiment_logs(self) -> tuple[Response, int]:
        """
        Get experiment logs for frontend compatibility.

        Request body: {"experiment_name": str} (optional)
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            # Evals mode doesn't have the same logging structure as fit
            # Return empty logs or a placeholder message
            return jsonify({
                "logs": "Experiment logs not available in evals mode. Check console output.",
                "message": "Evals mode uses console logging"
            }), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_ic_logs(self) -> tuple[Response, int]:
        """
        Get IC Ops logs for frontend compatibility.

        Request body: optional filters
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            # Return IC operations as logs
            operations = self.db.get_all_ic_operations()
            logs = []
            for op in operations:
                logs.append({
                    "timestamp": op.get("created_at"),
                    "operation": op.get("operation"),
                    "pipeline_id": op.get("pipeline_id"),
                    "status": op.get("status"),
                    "error": op.get("error", ""),
                })
            return jsonify({"logs": logs}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


def serve_forever() -> Flask:
    """Start the Dispatcher via Gunicorn (for external process mode)."""
    return Dispatcher().app


def run_dispatcher(host: str = "0.0.0.0", port: int = 8851) -> None:
    """
    Run the dispatcher server.

    This function is designed to be called in a separate thread from the main experiment.

    Args:
        host: Host to bind to (default: 0.0.0.0)
        port: Port to bind to (default: 8851)
    """
    try:
        dispatcher = Dispatcher()

        # Suppress Flask/werkzeug request logging
        log = logging.getLogger("werkzeug")
        log.setLevel(logging.ERROR)

        # Use waitress to serve the Flask app
        serve(dispatcher.app, host=host, port=port, threads=6)
    except Exception as e:
        # Catch all exceptions to prevent thread crashes
        print(f"CRITICAL: Dispatcher crashed: {e}")
        traceback.print_exc()


def _check_port_in_use(host: str,port: int) -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _cleanup_old_dispatcher(port: int, logger=None) -> None:
    """Kill any old dispatcher processes using the port."""
    import subprocess

    try:
        # Find process using the port
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                try:
                    subprocess.run(["kill", "-9", pid], timeout=2)
                    msg = f"Killed old process (PID {pid}) on port {port}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                except:
                    pass
    except:
        pass  # lsof might not be available


def start_dispatcher_thread(host: str = "0.0.0.0", port: int = 8851, logger=None) -> threading.Thread | None:
    """
    Start the dispatcher REST API server in a background daemon thread.

    The dispatcher enables interactive control (stop/resume/delete/clone pipelines)
    during experiment execution. It runs as a daemon thread and automatically
    cleans up when the experiment ends.

    Args:
        host: Host to bind to (default: 0.0.0.0, localhost only)
        port: Port to bind to (default: 8851)
        logger: Optional logger instance for logging (if None, uses print)

    Returns:
        The dispatcher thread object, or None if startup failed
    """
    try:
        # Check if port is in use
        if _check_port_in_use(host, port):
            msg = f"Port {port} is already in use. Attempting cleanup..."
            logger.warning(msg)

            # Try to clean up old process
            _cleanup_old_dispatcher(port, logger)

            # Wait a moment and check again
            import time

            time.sleep(0.5)
            if _check_port_in_use(host, port):
                error_msg = f"Port {port} still in use after cleanup. Restart your kernel or system."
                logger.error(error_msg)
                return None

        # Create daemon thread (auto-cleanup when main process ends)
        dispatcher_thread = threading.Thread(
            target=run_dispatcher, kwargs={"host": host, "port": port}, daemon=True, name="DispatcherThread"
        )
        dispatcher_thread.start()

        msg = f"Started interactive control dispatcher on http://{host}:{port}"
        if logger:
            logger.info(msg)
            logger.info("Use dispatcher API to dynamically stop/resume/delete/clone pipelines")
        else:
            print(msg)
            print("Use dispatcher API to dynamically stop/resume/delete/clone pipelines")

        return dispatcher_thread

    except Exception as e:
        error_msg = f"Failed to start dispatcher: {e}. Interactive control will not be available."
        if logger:
            logger.warning(error_msg)
        else:
            print(f"WARNING: {error_msg}")
        return None


if __name__ == "__main__":
    # For standalone testing
    run_dispatcher()
