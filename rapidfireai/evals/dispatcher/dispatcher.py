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
from rapidfireai.evals.utils.constants import ICOperation, DispatcherConfig

CORS_ALLOWED_ORIGINS = ["http://localhost:8853", "http://localhost", DispatcherConfig.URL]


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
        _ = CORS(
            self.app,
            resources={
                r"/*": {
                    "origins": "*",
                    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"],
                    "expose_headers": ["Content-Type"],
                    "supports_credentials": False,
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
                response.headers.add("Access-Control-Allow-Origin", "*")
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


def run_dispatcher(host: str = "127.0.0.1", port: int = 8851) -> None:
    """
    Run the dispatcher server.

    This function is designed to be called in a separate thread from the main experiment.

    Args:
        host: Host to bind to (default: 127.0.0.1)
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


def start_dispatcher_thread(host: str = "127.0.0.1", port: int = 8851, logger=None) -> threading.Thread | None:
    """
    Start the dispatcher REST API server in a background daemon thread.

    The dispatcher enables interactive control (stop/resume/delete/clone pipelines)
    during experiment execution. It runs as a daemon thread and automatically
    cleans up when the experiment ends.

    Args:
        host: Host to bind to (default: 127.0.0.1, localhost only)
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
