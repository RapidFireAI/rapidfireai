"""
Dispatcher REST API for Interactive Control of RF-Inferno Experiments.

Provides HTTP endpoints for dynamic pipeline management during experiment execution.
"""

import json
import threading
import traceback
import logging


from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from waitress import serve

from rapidfireai.evals.db import RFDatabase
from rapidfireai.evals.utils.constants import ICOperation

CORS_ALLOWED_ORIGINS = ["http://localhost:3000", "http://localhost"]


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
        _ = CORS(self.app, resources={r"/*": {"origins": "*"}})

        # Register routes
        self.register_routes()

    def register_routes(self) -> None:
        """Register all REST API routes."""
        route_prefix = "/dispatcher"

        # Health check
        self.app.add_url_rule(f"{route_prefix}/health-check", "health_check", self.health_check, methods=["GET"])

        # Interactive control operations
        self.app.add_url_rule(f"{route_prefix}/stop-pipeline", "stop_pipeline", self.stop_pipeline, methods=["POST"])
        self.app.add_url_rule(
            f"{route_prefix}/resume-pipeline", "resume_pipeline", self.resume_pipeline, methods=["POST"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/delete-pipeline", "delete_pipeline", self.delete_pipeline, methods=["POST"]
        )
        self.app.add_url_rule(f"{route_prefix}/clone-pipeline", "clone_pipeline", self.clone_pipeline, methods=["POST"])

        # Status queries
        self.app.add_url_rule(
            f"{route_prefix}/operation-status/<int:ic_id>",
            "get_operation_status",
            self.get_operation_status,
            methods=["GET"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/all-operations", "get_all_operations", self.get_all_operations, methods=["GET"]
        )

        # Pipeline queries (for UI)
        self.app.add_url_rule(
            f"{route_prefix}/get-all-pipelines", "get_all_pipelines", self.get_all_pipelines, methods=["GET"]
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-pipeline", "get_pipeline", self.get_pipeline, methods=["POST"]
        )

    def health_check(self) -> tuple[Response, int]:
        """Health check endpoint."""
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
        Clone a new pipeline using an existing context.

        Request body (VLLM):
            {
                "context_id": int,
                "pipeline_name": str (optional),
                "pipeline_type": "vllm" (optional, default),
                "model_config": dict,
                "sampling_params": dict
            }

        Request body (OpenAI):
            {
                "context_id": int,
                "pipeline_name": str (optional),
                "pipeline_type": "openai",
                "client_config": dict,
                "model_config": dict,
                "rpm_limit": int (optional),
                "tpm_limit": int (optional)
            }

        Returns:
            ic_id of the created operation
        """
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            context_id = data.get("context_id")
            if context_id is None:
                return jsonify({"error": "context_id is required"}), 400

            pipeline_type = data.get("pipeline_type", "vllm").lower()
            if pipeline_type not in ["vllm", "openai"]:
                return jsonify({"error": "pipeline_type must be 'vllm' or 'openai'"}), 400

            model_config = data.get("model_config")
            if not model_config:
                return jsonify({"error": "model_config is required"}), 400

            # Validate required fields based on pipeline type
            if pipeline_type == "vllm":
                sampling_params = data.get("sampling_params")
                if not sampling_params:
                    return jsonify({"error": "sampling_params is required for VLLM pipelines"}), 400

            elif pipeline_type == "openai":
                client_config = data.get("client_config")
                if not client_config:
                    return jsonify({"error": "client_config is required for OpenAI pipelines"}), 400

            # Validate context exists
            context = self.db.get_context(context_id)
            if not context:
                return jsonify({"error": f"Context {context_id} not found"}), 404

            # Prepare request data (pass through all fields from user)
            request_data = {
                "context_id": context_id,
                "pipeline_name": data.get("pipeline_name", f"cloned_{context_id}"),
                "pipeline_type": pipeline_type,
            }

            # Add type-specific fields
            if pipeline_type == "vllm":
                request_data["model_config"] = model_config
                request_data["sampling_params"] = data["sampling_params"]

            elif pipeline_type == "openai":
                request_data["client_config"] = data["client_config"]
                request_data["model_config"] = model_config
                request_data["rpm_limit"] = data.get("rpm_limit", 500)
                request_data["tpm_limit"] = data.get("tpm_limit", 90000)

            # Create IC operation (pipeline_id is None for CLONE)
            ic_id = self.db.create_ic_operation(
                operation=ICOperation.CLONE.value,
                pipeline_id=None,
                request_data=json.dumps(request_data),
            )

            return (
                jsonify(
                    {
                        "ic_id": ic_id,
                        "message": f"Clone request created with context {context_id} (type: {pipeline_type})",
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
        try:
            operations = self.db.get_all_ic_operations()
            return jsonify(operations), 200

        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_all_pipelines(self) -> tuple[Response, int]:
        """
        Get all pipelines (for UI dropdown).

        Returns:
            List of all pipelines
        """
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


def run_dispatcher(host: str = "127.0.0.1", port: int = 5000) -> None:
    """
    Run the dispatcher server.

    This function is designed to be called in a separate thread from the main experiment.

    Args:
        host: Host to bind to (default: 127.0.0.1)
        port: Port to bind to (default: 5000)
    """
    try:
        dispatcher = Dispatcher()

        # Suppress Flask/werkzeug request logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)

        # Use waitress to serve the Flask app
        serve(dispatcher.app, host=host, port=port, threads=6)
    except Exception as e:
        # Catch all exceptions to prevent thread crashes
        print(f"CRITICAL: Dispatcher crashed: {e}")
        traceback.print_exc()


def _check_port_in_use(port: int) -> bool:
    """Check if a port is already in use."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0


def _cleanup_old_dispatcher(port: int, logger=None) -> None:
    """Kill any old dispatcher processes using the port."""
    import subprocess
    try:
        # Find process using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                try:
                    subprocess.run(['kill', '-9', pid], timeout=2)
                    msg = f"Killed old process (PID {pid}) on port {port}"
                    if logger:
                        logger.info(msg)
                    else:
                        print(msg)
                except:
                    pass
    except:
        pass  # lsof might not be available


def start_dispatcher_thread(host: str = "127.0.0.1", port: int = 5000, logger=None) -> threading.Thread | None:
    """
    Start the dispatcher REST API server in a background daemon thread.

    The dispatcher enables interactive control (stop/resume/delete/clone pipelines)
    during experiment execution. It runs as a daemon thread and automatically
    cleans up when the experiment ends.

    Args:
        host: Host to bind to (default: 127.0.0.1, localhost only)
        port: Port to bind to (default: 5000)
        logger: Optional logger instance for logging (if None, uses print)

    Returns:
        The dispatcher thread object, or None if startup failed
    """
    try:
        # Check if port is in use
        if _check_port_in_use(port):
            msg = f"Port {port} is already in use. Attempting cleanup..."
            logger.warning(msg)

            # Try to clean up old process
            _cleanup_old_dispatcher(port, logger)

            # Wait a moment and check again
            import time
            time.sleep(0.5)
            if _check_port_in_use(port):
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
