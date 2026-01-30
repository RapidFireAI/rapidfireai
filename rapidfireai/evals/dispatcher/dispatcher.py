"""
Dispatcher REST API for Interactive Control of RF-Inferno Experiments.

Provides HTTP endpoints for dynamic pipeline management during experiment execution.
FIXED: Now properly handles CORS preflight (OPTIONS) requests for VS Code/Cursor webview.
"""

import json
import logging
import os
import threading
import traceback
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from waitress import serve

from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.utils.logger import RFLogger, SafeLoggerAdapter
from rapidfireai.utils.constants import DispatcherConfig, ColabConfig, RF_LOG_PATH, RF_LOG_FILENAME
from rapidfireai.utils.dispatcher_utils import check_experiment_running
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

    Also provides frontend-compatible "run" aliases for pipeline endpoints
    to support the MLflow dashboard IC Ops panel.
    """

    # Status mapping from evals (lowercase) to fit (capitalized) format
    STATUS_MAP = {
        "new": "New",
        "ongoing": "Ongoing",
        "completed": "Completed",
        "stopped": "Stopped",
        "deleted": "Deleted",
        "failed": "Failed",
    }

    def __init__(self) -> None:
        """Initialize the Dispatcher with database connection and Flask app."""
        # initialize loggers
        self.logger_experiment_name: str | None = None
        self.logger: SafeLoggerAdapter | None = None

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
                    "supports_credentials": True if ColabConfig.ON_COLAB else False,
                }
            },
        )

        # Register routes
        self.register_routes()

    def _get_logger(self) -> SafeLoggerAdapter:
        """Get the latest logger for the dispatcher"""
        current_experiment_name = self.db.get_running_experiment()["experiment_name"]
        if self.logger is None or self.logger_experiment_name != current_experiment_name:
            self.logger = RFLogger(experiment_name=current_experiment_name).get_logger("dispatcher")
            self.logger_experiment_name = current_experiment_name
        return self.logger

    def _transform_pipeline_to_run(self, pipeline: dict) -> dict:
        """
        Transform pipeline data to run format for frontend compatibility.

        The MLflow frontend expects run data in a specific format that differs
        from the evals pipeline format. This method converts pipeline data
        to match the expected run format.

        Args:
            pipeline: Pipeline data from database

        Returns:
            Run-formatted data for frontend consumption
        """
        status = pipeline.get("status", "")
        return {
            "run_id": pipeline.get("pipeline_id"),
            "status": self.STATUS_MAP.get(status.lower(), status),
            "metric_run_id": pipeline.get("metric_run_id"),
            "config": pipeline.get("pipeline_config_json", {}),
            "flattened_config": pipeline.get("flattened_config", {}),
            "num_chunks_visited": pipeline.get("shards_completed", 0),
            "total_samples_processed": pipeline.get("total_samples_processed", 0),
            "error": pipeline.get("error", ""),
        }

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

        # ============================================================
        # Frontend-compatible "run" aliases (for MLflow dashboard IC Ops)
        # These endpoints mirror the fit dispatcher API to allow the
        # frontend to work with evals mode without modifications.
        # ============================================================

        # Run query aliases
        self.app.add_url_rule(
            f"{route_prefix}/get-all-runs", "get_all_runs", self.get_all_runs, methods=["GET", "OPTIONS"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-run", "get_run", self.get_run, methods=["POST", "OPTIONS"]
        )

        # IC Ops aliases (run → pipeline)
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

        # Experiment info endpoints (for frontend context)
        self.app.add_url_rule(
            f"{route_prefix}/get-running-experiment",
            "get_running_experiment",
            self.get_running_experiment,
            methods=["GET", "OPTIONS"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-all-experiment-names",
            "get_all_experiment_names",
            self.get_all_experiment_names,
            methods=["GET", "OPTIONS"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/is-experiment-running",
            "is_experiment_running",
            self.is_experiment_running,
            methods=["POST", "OPTIONS"],
        )

        # Logging endpoints (placeholders for frontend compatibility)
        self.app.add_url_rule(
            f"{route_prefix}/get-experiment-logs",
            "get_experiment_logs",
            self.get_experiment_logs,
            methods=["POST", "OPTIONS"],
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

    # ============================================================
    # Frontend-compatible "run" alias handlers
    # ============================================================

    def get_all_runs(self) -> tuple[Response, int]:
        """
        Get all runs (alias for get_all_pipelines with response transformation).

        This endpoint provides frontend compatibility by returning pipeline data
        in the "run" format expected by the MLflow dashboard.

        Returns:
            List of runs in frontend-compatible format
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            pipelines = self.db.get_all_pipelines()
            runs = [self._transform_pipeline_to_run(p) for p in pipelines]
            return jsonify(runs), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_run(self) -> tuple[Response, int]:
        """
        Get a specific run (alias for get_pipeline with response transformation).

        Request body:
            {
                "run_id": int or str (pipeline_id or metricrun_id)
            }

        Returns:
            Run details in frontend-compatible format
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
                return jsonify({"error": f"Run {run_id} not found"}), 400

            return jsonify(self._transform_pipeline_to_run(pipeline)), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def stop_run(self) -> tuple[Response, int]:
        """Stop the run"""
        try:
            data = request.get_json()
            run_id = data["run_id"]
            task = ICOperation.STOP

            # get pipeline from db
            pipeline = self.db.get_pipeline(run_id)

            # create ic ops task
            _ = self.db.create_ic_operation(
                operation=task.value,
                pipeline_id=pipeline["pipeline_id"],
            )

            # log the task
            logger = self._get_logger()
            logger.info(f"Received stop task for run_id: {run_id}")
            return jsonify({}), 200
        except Exception as e:
            logger = self._get_logger()
            logger.error(f"Error in stop_run: {e}", exc_info=True)
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def resume_run(self) -> tuple[Response, int]:
        """Resume the run"""
        try:
            data = request.get_json()
            run_id = data["run_id"]
            task = ICOperation.RESUME

            # get pipeline from db
            pipeline = self.db.get_pipeline(run_id)

            # set resume run task
            _ = self.db.create_ic_operation(
                operation=task.value,
                pipeline_id=pipeline["pipeline_id"],
            )

            # log the task
            logger = self._get_logger()
            logger.info(f"Received resume task for run_id: {run_id}")
            return jsonify({}), 200
        except Exception as e:
            logger = self._get_logger()
            logger.error(f"Error in resume_run: {e}", exc_info=True)
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def delete_run(self) -> tuple[Response, int]:
        """Delete the run"""
        try:
            data = request.get_json()
            run_id = data["run_id"]
            task = ICOperation.DELETE

            # get pipeline from db
            pipeline = self.db.get_pipeline(run_id)

            # set delete run task
            _ = self.db.create_ic_operation(
                operation=task.value,
                pipeline_id=pipeline["pipeline_id"],
            )

            # log the task
            logger = self._get_logger()
            logger.info(f"Received delete task for run_id: {run_id}")
            return jsonify({}), 200
        except Exception as e:
            logger = self._get_logger()
            logger.error(f"Error in delete_run: {e}", exc_info=True)
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def clone_modify_run(self) -> tuple[Response, int]:
        """
        Clone and modify a run (alias for clone_pipeline with payload transformation).

        Request body:
            {
                "run_id": int,           # ID of the run to clone (maps to parent_pipeline_id)
                "config": dict,          # New configuration
                "warm_start": bool       # Optional, passed through but may not apply to evals
            }

        Returns:
            IC operation result
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

            config = data.get("config", {})
            warm_start = data.get("warm_start", False)

            # Validate parent pipeline exists
            parent_pipeline = self.db.get_pipeline(run_id)
            if not parent_pipeline:
                return jsonify({"error": f"Run {run_id} not found"}), 404

            # Get the parent's config to use as base if config is empty
            if not config:
                config = parent_pipeline.get("pipeline_config_json", {})

            # Prepare request data for IC operation
            request_data = {
                "parent_pipeline_id": run_id,
                "config_json": config,
                "warm_start": warm_start,  # Passed through for potential future use
            }

            # Create IC operation
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.CLONE.value,
                pipeline_id=None,
                request_data=json.dumps(request_data),
            )

            return jsonify({"ic_id": ic_id, "message": f"Clone request created from run {run_id}"}), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ============================================================
    # Experiment info endpoints
    # ============================================================

    def get_running_experiment(self) -> tuple[Response, int]:
        """
        Get the currently running experiment.

        Returns:
            Experiment details including ID, name, status, and MLflow experiment ID
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            experiment = self.db.get_running_experiment()
            if not experiment:
                return jsonify({"error": "No running experiment found"}), 404

            return jsonify({
                "experiment_id": experiment.get("experiment_id"),
                "experiment_name": experiment.get("experiment_name"),
                "status": experiment.get("status"),
                "metric_experiment_id": experiment.get("metric_experiment_id"),
            }), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_all_experiment_names(self) -> tuple[Response, int]:
        """
        Get all experiment names.

        Returns:
            List of experiment names
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            names = self.db.get_all_experiment_names()
            return jsonify(names), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def is_experiment_running(self) -> tuple[Response, int]:
        """
        Check if a specific experiment is currently running.

        Request body:
            {
                "experiment_name": str - The experiment name to check
            }

        Returns:
            {
                "is_running": bool - True if the experiment is currently running
            }
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            data = request.get_json()
            if not data or "experiment_name" not in data:
                return jsonify({"error": "experiment_name is required"}), 400

            is_running = check_experiment_running(self.db, data["experiment_name"])
            return jsonify({"is_running": is_running}), 200

        except Exception:
            # If anything fails, assume experiment is not running (safer to disable button)
            return jsonify({"is_running": False}), 200

    # ============================================================
    # Logging endpoints
    # ============================================================

    def get_experiment_logs(self) -> tuple[Response, int]:
        """
        Get experiment logs.

        Reads logs from the experiment log file and returns entries matching
        the experiment name.

        Request body:
            {
                "experiment_name": str (optional) - if not provided, uses running experiment
            }

        Returns:
            List of log entries for the experiment
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            experiment_name = None
            if request.is_json:
                data = request.get_json()
                if data and data.get("experiment_name"):
                    experiment_name = data["experiment_name"]

            if not experiment_name:
                running_exp = self.db.get_running_experiment()
                if running_exp:
                    experiment_name = running_exp.get("experiment_name")

            if not experiment_name:
                return jsonify([]), 200

            log_file_path = os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

            if not os.path.exists(log_file_path):
                return jsonify([]), 200

            experiment_logs = []
            with open(log_file_path, encoding="utf-8") as f:
                for line in f:
                    # Filter for lines containing the experiment name
                    if f"[{experiment_name}:" in line or f"| {experiment_name} |" in line:
                        experiment_logs.append(line.strip())

            return jsonify(experiment_logs), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_ic_logs(self) -> tuple[Response, int]:
        """
        Get interactive control logs.

        Reads logs from the experiment log file and returns entries related
        to interactive control operations.

        Request body:
            {
                "experiment_name": str (optional) - if not provided, uses running experiment
            }

        Returns:
            List of IC log entries for the experiment
        """
        if request.method == "OPTIONS":
            return jsonify({}), 200

        try:
            experiment_name = None
            if request.is_json:
                data = request.get_json()
                if data and data.get("experiment_name"):
                    experiment_name = data["experiment_name"]

            if not experiment_name:
                running_exp = self.db.get_running_experiment()
                if running_exp:
                    experiment_name = running_exp.get("experiment_name")

            if not experiment_name:
                return jsonify([]), 200

            log_file_path = os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

            if not os.path.exists(log_file_path):
                return jsonify([]), 200

            # Read and filter logs for interactive-control entries
            ic_logs = []
            with open(log_file_path, encoding="utf-8") as f:
                for line in f:
                    if f"| {experiment_name} | interactive-control |" in line:
                        ic_logs.append(line.strip())

            return jsonify(ic_logs), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


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

        # Enable verbose logging for waitress
        logging.getLogger("waitress").setLevel(logging.WARNING)

        # Use waitress to serve the Flask app
        serve(dispatcher.app, host=host, port=port, threads=6)
    except Exception as e:
        # Catch all exceptions to prevent thread crashes
        print(f"CRITICAL: Dispatcher crashed: {e}")
        traceback.print_exc()


# Global dispatcher thread tracking to avoid killing our own process
_dispatcher_thread: threading.Thread | None = None
_dispatcher_lock = threading.Lock()


def _check_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def _is_dispatcher_thread_alive() -> bool:
    """Check if our dispatcher thread is still running."""
    global _dispatcher_thread
    return _dispatcher_thread is not None and _dispatcher_thread.is_alive()


def _cleanup_old_dispatcher(port: int, logger=None) -> None:
    """
    Kill any old dispatcher processes using the port.

    IMPORTANT: Only kills external processes, not threads in the current process.
    If the port is in use by our own dispatcher thread, this is a no-op.
    """
    import os
    import subprocess

    # If our dispatcher thread is alive, don't try to kill anything
    if _is_dispatcher_thread_alive():
        msg = "Dispatcher thread is already running in this process, skipping cleanup"
        if logger:
            logger.debug(msg)
        return

    current_pid = str(os.getpid())

    try:
        # Find process using the port
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True, timeout=2)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                # CRITICAL: Never kill our own process
                if pid.strip() == current_pid:
                    msg = f"Port {port} is used by current process (PID {pid}), skipping kill"
                    if logger:
                        logger.warning(msg)
                    else:
                        print(f"WARNING: {msg}")
                    continue
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

    If a dispatcher thread is already running in this process, returns the existing
    thread instead of starting a new one.

    Args:
        host: Host to bind to (default: 0.0.0.0, localhost only)
        port: Port to bind to (default: 8851)
        logger: Optional logger instance for logging (if None, uses print)

    Returns:
        The dispatcher thread object, or None if startup failed
    """
    global _dispatcher_thread

    with _dispatcher_lock:
        # Check if our dispatcher thread is already running
        if _is_dispatcher_thread_alive():
            msg = f"Dispatcher thread already running on port {port}, reusing existing thread"
            if logger:
                logger.info(msg)
            else:
                print(msg)
            return _dispatcher_thread

        try:
            # Check if port is in use (by an external process)
            if _check_port_in_use(host, port):
                msg = f"Port {port} is already in use. Attempting cleanup..."
                if logger:
                    logger.warning(msg)
                else:
                    print(f"WARNING: {msg}")

                # Try to clean up old process (will skip if it's our own process)
                _cleanup_old_dispatcher(port, logger)

                # Wait a moment and check again
                import time

                time.sleep(0.5)
                if _check_port_in_use(host, port):
                    error_msg = f"Port {port} still in use after cleanup. Restart your kernel or system."
                    if logger:
                        logger.error(error_msg)
                    else:
                        print(f"ERROR: {error_msg}")
                    return None

            # Create daemon thread (auto-cleanup when main process ends)
            _dispatcher_thread = threading.Thread(
                target=run_dispatcher, kwargs={"host": host, "port": port}, daemon=True, name="DispatcherThread"
            )
            _dispatcher_thread.start()

            msg = f"Started interactive control dispatcher on http://{host}:{port}"
            if logger:
                logger.info(msg)
                logger.info("Use dispatcher API to dynamically stop/resume/delete/clone pipelines")
            else:
                print(msg)
                print("Use dispatcher API to dynamically stop/resume/delete/clone pipelines")

            return _dispatcher_thread

        except Exception as e:
            error_msg = f"Failed to start dispatcher: {e}. Interactive control will not be available."
            if logger:
                logger.warning(error_msg)
            else:
                print(f"WARNING: {error_msg}")
            return None


def serve_forever() -> Flask:
    """Start the Dispatcher via Gunicorn.

    This function is the entry point for Gunicorn WSGI server.
    It returns the Flask app instance for Gunicorn to serve.

    Returns:
        Flask app instance
    """
    return Dispatcher().app


if __name__ == "__main__":
    # For standalone testing
    run_dispatcher()
