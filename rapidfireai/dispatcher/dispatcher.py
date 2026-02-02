"""
RapidFire AI Dispatcher

REST API for Interactive Control of experiments.
Provides HTTP endpoints for dynamic run/pipeline management during experiment execution.
"""

import json
import logging
import os
import threading
import traceback

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from rapidfireai.db.rf_db import RfDb
from rapidfireai.utils.constants import (
    ColabConfig,
    DispatcherConfig,
    FrontendConfig,
    ICOperation,
    ICStatus,
    MLFlowConfig,
    PipelineStatus,
    RF_LOG_FILENAME,
    RF_LOG_PATH,
    RunStatus,
)
from rapidfireai.utils.dispatcher_utils import check_experiment_running

CORS_ALLOWED_ORIGINS = ["http://localhost", DispatcherConfig.URL, MLFlowConfig.URL, FrontendConfig.URL, "*"]


class Dispatcher:
    """
    REST API server for interactive control of experiments.
    Provides endpoints for:
    - Run management: /dispatcher/get-all-runs, /dispatcher/stop-run, etc.
    - Pipeline management: /dispatcher/get-all-pipelines, /dispatcher/stop-pipeline, etc.
    """

    # Status mapping for frontend compatibility (lowercase to capitalized)
    STATUS_MAP = {
        "new": "New",
        "ongoing": "Ongoing",
        "completed": "Completed",
        "stopped": "Stopped",
        "deleted": "Deleted",
        "failed": "Failed",
    }

    def __init__(self):
        """Initialize the Dispatcher with database connection and Flask app."""
        self.db = RfDb()
        self.app = Flask(__name__)

        # CORS configuration
        cors_credentials = True if ColabConfig.ON_COLAB else False
        CORS(
            self.app,
            resources={
                r"/*": {
                    "origins": "*",
                    "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                    "allow_headers": ["Content-Type", "Authorization"],
                    "expose_headers": ["Content-Type"],
                    "supports_credentials": cors_credentials,
                }
            },
        )

        self.register_routes()

    def register_routes(self):
        """Register all REST API routes."""
        route_prefix = "/dispatcher"

        # Handle OPTIONS preflight globally
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
        self.app.add_url_rule(f"{route_prefix}/health-check", "health_check", self.health_check, methods=["GET"])

        # ============================================================
        # SHARED ROUTES (both modes)
        # ============================================================
        self.app.add_url_rule(
            f"{route_prefix}/get-running-experiment",
            "get_running_experiment",
            self.get_running_experiment,
            methods=["GET"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-all-experiment-names",
            "get_all_experiment_names",
            self.get_all_experiment_names,
            methods=["GET"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-experiment-logs",
            "get_experiment_logs",
            self.get_experiment_logs,
            methods=["POST"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/get-ic-logs",
            "get_ic_logs",
            self.get_ic_logs,
            methods=["POST"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/is-experiment-running",
            "is_experiment_running",
            self.is_experiment_running,
            methods=["POST"],
        )

        # ============================================================
        # FIT MODE ROUTES (runs)
        # ============================================================
        self.app.add_url_rule(f"{route_prefix}/get-all-runs", "get_all_runs", self.get_all_runs, methods=["GET"])
        self.app.add_url_rule(f"{route_prefix}/get-run", "get_run", self.get_run, methods=["POST"])
        self.app.add_url_rule(f"{route_prefix}/stop-run", "stop_run", self.stop_run, methods=["POST"])
        self.app.add_url_rule(f"{route_prefix}/resume-run", "resume_run", self.resume_run, methods=["POST"])
        self.app.add_url_rule(f"{route_prefix}/delete-run", "delete_run", self.delete_run, methods=["POST"])
        self.app.add_url_rule(
            f"{route_prefix}/clone-modify-run",
            "clone_modify_run",
            self.clone_modify_run,
            methods=["POST"],
        )

        # ============================================================
        # EVALS MODE ROUTES (pipelines)
        # ============================================================
        self.app.add_url_rule(
            f"{route_prefix}/get-all-pipelines",
            "get_all_pipelines",
            self.get_all_pipelines,
            methods=["GET"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/list-all-pipeline-ids",
            "list_all_pipeline_ids",
            self.list_all_pipeline_ids,
            methods=["GET"],
        )
        self.app.add_url_rule(f"{route_prefix}/get-pipeline", "get_pipeline", self.get_pipeline, methods=["POST"])
        self.app.add_url_rule(
            f"{route_prefix}/get-pipeline-config-json/<int:pipeline_id>",
            "get_pipeline_config_json",
            self.get_pipeline_config_json,
            methods=["GET"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/stop-pipeline",
            "stop_pipeline",
            self.stop_pipeline,
            methods=["POST"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/resume-pipeline",
            "resume_pipeline",
            self.resume_pipeline,
            methods=["POST"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/delete-pipeline",
            "delete_pipeline",
            self.delete_pipeline,
            methods=["POST"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/clone-pipeline",
            "clone_pipeline",
            self.clone_pipeline,
            methods=["POST"],
        )

        # IC operations status
        self.app.add_url_rule(
            f"{route_prefix}/operation-status/<int:ic_id>",
            "get_operation_status",
            self.get_operation_status,
            methods=["GET"],
        )
        self.app.add_url_rule(
            f"{route_prefix}/all-operations",
            "get_all_operations",
            self.get_all_operations,
            methods=["GET"],
        )

    # ============================================================
    # HEALTH CHECK
    # ============================================================

    def health_check(self) -> tuple[Response, int]:
        """Health check endpoint."""
        return jsonify({"status": "ok", "message": "Dispatcher is running"}), 200

    # ============================================================
    # SHARED ROUTES
    # ============================================================

    def get_running_experiment(self) -> tuple[Response, int]:
        """Get the currently running experiment."""
        try:
            experiment = self.db.get_running_experiment()

            # Detect mode by checking if we have runs or pipelines
            has_runs = len(self.db.get_all_runs()) > 0
            has_pipelines = len(self.db.get_all_pipelines()) > 0
            mode = "fit" if has_runs else ("evals" if has_pipelines else "unknown")

            return jsonify({
                "experiment_id": experiment["experiment_id"],
                "experiment_name": experiment["experiment_name"],
                "status": experiment["status"].value if hasattr(experiment["status"], "value") else experiment["status"],
                "metric_experiment_id": experiment.get("metric_experiment_id"),
                "mode": mode,
            }), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def get_all_experiment_names(self) -> tuple[Response, int]:
        """Get all experiment names."""
        try:
            names = self.db.get_all_experiment_names()
            return jsonify(names), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def is_experiment_running(self) -> tuple[Response, int]:
        """Check if a specific experiment is currently running."""
        try:
            data = request.get_json()
            if not data or "experiment_name" not in data:
                return jsonify({"error": "experiment_name is required"}), 400

            is_running = check_experiment_running(self.db, data["experiment_name"])
            return jsonify({"is_running": is_running}), 200
        except Exception:
            # If anything fails, assume experiment is not running (safer to disable button)
            return jsonify({"is_running": False}), 200

    def get_experiment_logs(self) -> tuple[Response, int]:
        """Get experiment logs."""
        try:
            experiment_name = None
            if request.is_json:
                data = request.get_json()
                if data and data.get("experiment_name"):
                    experiment_name = data["experiment_name"]

            if not experiment_name:
                try:
                    running_exp = self.db.get_running_experiment()
                    experiment_name = running_exp["experiment_name"]
                except Exception:
                    return jsonify([]), 200

            log_file_path = os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

            if not os.path.exists(log_file_path):
                return jsonify([]), 200

            experiment_logs = []
            with open(log_file_path, encoding="utf-8") as f:
                for line in f:
                    if f"[{experiment_name}:" in line or f"| {experiment_name} |" in line:
                        experiment_logs.append(line.strip())

            return jsonify(experiment_logs), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_ic_logs(self) -> tuple[Response, int]:
        """Get interactive control logs."""
        try:
            experiment_name = None
            if request.is_json:
                data = request.get_json()
                if data and data.get("experiment_name"):
                    experiment_name = data["experiment_name"]

            if not experiment_name:
                try:
                    running_exp = self.db.get_running_experiment()
                    experiment_name = running_exp["experiment_name"]
                except Exception:
                    return jsonify([]), 200

            log_file_path = os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

            if not os.path.exists(log_file_path):
                return jsonify([]), 200

            ic_logs = []
            with open(log_file_path, encoding="utf-8") as f:
                for line in f:
                    if f"| {experiment_name} | interactive-control |" in line:
                        ic_logs.append(line.strip())

            return jsonify(ic_logs), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ============================================================
    # FIT MODE ROUTES (runs)
    # ============================================================

    def _format_run_for_response(self, run_id: int, run_data: dict) -> dict:
        """Format run data for API response."""
        config_leaf = run_data.get("config_leaf", {})
        if config_leaf:
            config_leaf = dict(config_leaf)
            config_leaf.pop("additional_kwargs", None)
            if "peft_params" in config_leaf:
                config_leaf["peft_params"].pop("task_type", None)
            if "model_kwargs" in config_leaf:
                config_leaf["model_kwargs"].pop("torch_dtype", None)
            if "reward_funcs" in config_leaf:
                config_leaf.pop("reward_funcs", None)

        status = run_data["status"]
        status_value = status.value if hasattr(status, "value") else status

        return {
            "run_id": run_id,
            "status": status_value,
            "metric_run_id": run_data.get("metric_run_id"),
            "config": config_leaf,
            "flattened_config": run_data.get("flattened_config", {}),
            "completed_steps": run_data.get("completed_steps", 0),
            "total_steps": run_data.get("total_steps", 0),
            "num_epochs_completed": run_data.get("num_epochs_completed", 0),
            "error": run_data.get("error", ""),
            "source": run_data["source"].value if run_data.get("source") else None,
            "ended_by": run_data["ended_by"].value if run_data.get("ended_by") else None,
        }

    def get_all_runs(self) -> tuple[Response, int]:
        """Get all runs (fit mode)."""
        try:
            runs = self.db.get_all_runs()
            result = [self._format_run_for_response(run_id, run_data) for run_id, run_data in runs.items()]
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_run(self) -> tuple[Response, int]:
        """Get a specific run (fit mode)."""
        try:
            data = request.get_json()
            run_id = data.get("run_id")
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            run_data = self.db.get_run(run_id)
            return jsonify(self._format_run_for_response(run_id, run_data)), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def stop_run(self) -> tuple[Response, int]:
        """Stop a run (fit mode)."""
        try:
            data = request.get_json()
            run_id = data.get("run_id")
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            ic_id = self.db.create_ic_operation(
                target_id=run_id,
                target_type="run",
                operation=ICOperation.STOP,
            )
            return jsonify({"ic_id": ic_id}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def resume_run(self) -> tuple[Response, int]:
        """Resume a run (fit mode)."""
        try:
            data = request.get_json()
            run_id = data.get("run_id")
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            ic_id = self.db.create_ic_operation(
                target_id=run_id,
                target_type="run",
                operation=ICOperation.RESUME,
            )
            return jsonify({"ic_id": ic_id}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def delete_run(self) -> tuple[Response, int]:
        """Delete a run (fit mode)."""
        try:
            data = request.get_json()
            run_id = data.get("run_id")
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            ic_id = self.db.create_ic_operation(
                target_id=run_id,
                target_type="run",
                operation=ICOperation.DELETE,
            )
            return jsonify({"ic_id": ic_id}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def clone_modify_run(self) -> tuple[Response, int]:
        """Clone and modify a run (fit mode)."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data received"}), 400

            run_id = data.get("run_id")
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            config = data.get("config")
            if not config:
                return jsonify({"error": "config is required"}), 400

            warm_start = data.get("warm_start", False)
            operation = ICOperation.CLONE_WARM if warm_start else ICOperation.CLONE

            ic_id = self.db.create_ic_operation(
                target_id=run_id,
                target_type="run",
                operation=operation,
                config_data=config,
            )
            return jsonify({"ic_id": ic_id}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ============================================================
    # EVALS MODE ROUTES (pipelines)
    # ============================================================

    def _format_pipeline_for_response(self, pipeline: dict) -> dict:
        """Format pipeline data for API response."""
        status = pipeline.get("status", "")
        status_value = status.value if hasattr(status, "value") else status

        # Parse flattened_config to use as 'config' field for IC compatibility
        flattened_config = pipeline.get("flattened_config", {})
        if isinstance(flattened_config, str):
            try:
                import json
                flattened_config = json.loads(flattened_config)
            except (json.JSONDecodeError, ValueError):
                flattened_config = {}

        return {
            "pipeline_id": pipeline.get("pipeline_id"),
            "context_id": pipeline.get("context_id"),
            "pipeline_type": pipeline.get("pipeline_type"),
            "pipeline_config_json": pipeline.get("pipeline_config_json", {}),
            "flattened_config": flattened_config,
            "config": flattened_config,  # Add 'config' field for InteractiveController compatibility
            "status": status_value,
            "current_shard_id": pipeline.get("current_shard_id", 0),
            "shards_completed": pipeline.get("shards_completed", 0),
            "total_samples_processed": pipeline.get("total_samples_processed", 0),
            "metric_run_id": pipeline.get("metric_run_id"),
            "error": pipeline.get("error", ""),
            "created_at": pipeline.get("created_at"),
        }

    def get_all_pipelines(self) -> tuple[Response, int]:
        """Get all pipelines (evals mode)."""
        try:
            pipelines = self.db.get_all_pipelines()
            result = [self._format_pipeline_for_response(p) for p in pipelines]
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def list_all_pipeline_ids(self) -> tuple[Response, int]:
        """Get lightweight list of pipeline IDs (evals mode)."""
        try:
            pipelines = self.db.get_all_pipeline_ids()
            return jsonify(pipelines), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_pipeline(self) -> tuple[Response, int]:
        """Get a specific pipeline (evals mode)."""
        try:
            data = request.get_json()
            pipeline_id = data.get("pipeline_id")
            if not pipeline_id:
                return jsonify({"error": "pipeline_id is required"}), 400

            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            return jsonify(self._format_pipeline_for_response(pipeline)), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_pipeline_config_json(self, pipeline_id: int) -> tuple[Response, int]:
        """Get config JSON for a pipeline (evals mode)."""
        try:
            config_data = self.db.get_pipeline_config_json(pipeline_id)
            if not config_data:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404
            return jsonify(config_data), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def stop_pipeline(self) -> tuple[Response, int]:
        """Stop a pipeline (evals mode)."""
        try:
            data = request.get_json()
            pipeline_id = data.get("pipeline_id")
            if not pipeline_id:
                return jsonify({"error": "pipeline_id is required"}), 400

            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            ic_id = self.db.create_ic_operation(
                target_id=pipeline_id,
                target_type="pipeline",
                operation=ICOperation.STOP,
            )
            return jsonify({"ic_id": ic_id, "message": f"Stop request created for pipeline {pipeline_id}"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def resume_pipeline(self) -> tuple[Response, int]:
        """Resume a pipeline (evals mode)."""
        try:
            data = request.get_json()
            pipeline_id = data.get("pipeline_id")
            if not pipeline_id:
                return jsonify({"error": "pipeline_id is required"}), 400

            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            ic_id = self.db.create_ic_operation(
                target_id=pipeline_id,
                target_type="pipeline",
                operation=ICOperation.RESUME,
            )
            return jsonify({"ic_id": ic_id, "message": f"Resume request created for pipeline {pipeline_id}"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def delete_pipeline(self) -> tuple[Response, int]:
        """Delete a pipeline (evals mode)."""
        try:
            data = request.get_json()
            pipeline_id = data.get("pipeline_id")
            if not pipeline_id:
                return jsonify({"error": "pipeline_id is required"}), 400

            pipeline = self.db.get_pipeline(pipeline_id)
            if not pipeline:
                return jsonify({"error": f"Pipeline {pipeline_id} not found"}), 404

            ic_id = self.db.create_ic_operation(
                target_id=pipeline_id,
                target_type="pipeline",
                operation=ICOperation.DELETE,
            )
            return jsonify({"ic_id": ic_id, "message": f"Delete request created for pipeline {pipeline_id}"}), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def clone_pipeline(self) -> tuple[Response, int]:
        """Clone a pipeline (evals mode)."""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data provided"}), 400

            parent_pipeline_id = data.get("parent_pipeline_id")
            if not parent_pipeline_id:
                return jsonify({"error": "parent_pipeline_id is required"}), 400

            config_json = data.get("config_json")
            if not config_json:
                return jsonify({"error": "config_json is required"}), 400

            parent_pipeline = self.db.get_pipeline(parent_pipeline_id)
            if not parent_pipeline:
                return jsonify({"error": f"Parent pipeline {parent_pipeline_id} not found"}), 404

            request_data = {
                "parent_pipeline_id": parent_pipeline_id,
                "config_json": config_json,
            }

            ic_id = self.db.create_ic_operation(
                target_id=0,  # Will be set when the new pipeline is created
                target_type="pipeline",
                operation=ICOperation.CLONE,
                config_data=request_data,
            )
            return jsonify({
                "ic_id": ic_id,
                "message": f"Clone request created from parent pipeline {parent_pipeline_id}",
            }), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    # ============================================================
    # IC OPERATIONS STATUS
    # ============================================================

    def get_operation_status(self, ic_id: int) -> tuple[Response, int]:
        """Get status of a specific IC operation."""
        try:
            operation = self.db.get_ic_operation(ic_id)
            if not operation:
                return jsonify({"error": f"Operation {ic_id} not found"}), 404
            return jsonify(operation), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

    def get_all_operations(self) -> tuple[Response, int]:
        """Get all IC operations."""
        try:
            operations = self.db.get_all_ic_operations()
            return jsonify(operations), 200
        except Exception as e:
            return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


# ============================================================================
# SERVER FUNCTIONS
# ============================================================================

def run_dispatcher(host: str = "0.0.0.0", port: int = 8851) -> None:
    """Run the dispatcher server (blocking)."""
    try:
        from waitress import serve

        dispatcher = Dispatcher()
        logging.getLogger("waitress").setLevel(logging.WARNING)
        serve(dispatcher.app, host=host, port=port, threads=6)
    except Exception as e:
        print(f"CRITICAL: Dispatcher crashed: {e}")
        traceback.print_exc()


# Global dispatcher thread tracking
_dispatcher_thread: threading.Thread | None = None
_dispatcher_lock = threading.Lock()


def _check_port_in_use(host: str, port: int) -> bool:
    """Check if a port is already in use."""
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def start_dispatcher_thread(host: str = "0.0.0.0", port: int = 8851, logger=None) -> threading.Thread | None:
    """
    Start the dispatcher REST API server in a background daemon thread.

    Args:
        host: Host to bind to
        port: Port to bind to
        logger: Optional logger instance

    Returns:
        The dispatcher thread object, or None if startup failed
    """
    global _dispatcher_thread

    with _dispatcher_lock:
        # Check if our dispatcher thread is already running
        if _dispatcher_thread is not None and _dispatcher_thread.is_alive():
            msg = f"Dispatcher thread already running on port {port}, reusing existing thread"
            if logger:
                logger.info(msg)
            return _dispatcher_thread

        try:
            if _check_port_in_use(host, port):
                msg = f"Port {port} is already in use"
                if logger:
                    logger.warning(msg)
                return None

            _dispatcher_thread = threading.Thread(
                target=run_dispatcher,
                kwargs={"host": host, "port": port},
                daemon=True,
                name="DispatcherThread",
            )
            _dispatcher_thread.start()

            msg = f"Started interactive control dispatcher on http://{host}:{port}"
            if logger:
                logger.info(msg)
            else:
                print(msg)

            return _dispatcher_thread

        except Exception as e:
            error_msg = f"Failed to start dispatcher: {e}"
            if logger:
                logger.warning(error_msg)
            else:
                print(f"WARNING: {error_msg}")
            return None


def serve_forever() -> Flask:
    """Start the Dispatcher via Gunicorn (WSGI entry point)."""
    return Dispatcher().app


if __name__ == "__main__":
    run_dispatcher()
