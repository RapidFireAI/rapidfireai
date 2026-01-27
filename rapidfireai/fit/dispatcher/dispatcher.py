"""This module contains functions for the dispatcher module."""

import os
import traceback
from logging import Logger
from typing import Any

from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.utils.constants import DispatcherConfig, FrontendConfig, MLFlowConfig, RF_LOG_FILENAME, RF_LOG_PATH
from rapidfireai.utils.dispatcher_utils import check_experiment_running
from rapidfireai.fit.utils.constants import ControllerTask
from rapidfireai.fit.utils.exceptions import DispatcherException
from rapidfireai.fit.utils.logging import RFLogger

CORS_ALLOWED_ORIGINS = ["http://localhost", DispatcherConfig.URL, MLFlowConfig.URL, FrontendConfig.URL]


class Dispatcher:
    """Class to co-ordinate the flow of tasks between the user and Controllers"""

    def __init__(self) -> None:
        # initialize loggers
        self.logger_experiment_name: str | None = None
        self.logger: Logger | None = None

        # create Db handle
        self.db: RfDb = RfDb()

        # create Flask app
        self.app: Flask = Flask(__name__)

        # Enable CORS for all routes
        # Allow all origins for local development (dispatcher runs on localhost)
        # This is safe since the API is not exposed to the internet
        _ = CORS(self.app, resources={r"/*": {"origins": "*"}})

        # register routes
        self.register_routes()

    def _get_logger(self) -> Logger | None:
        """Get the latest logger for the dispatcher"""
        current_experiment_name = self.db.get_running_experiment()["experiment_name"]
        if self.logger is None or self.logger_experiment_name != current_experiment_name:
            self.logger = RFLogger().create_logger("dispatcher")
            self.logger_experiment_name = current_experiment_name
        return self.logger

    def register_routes(self) -> None:
        """Register the routes for the dispatcher"""

        try:
            route_prefix = "/dispatcher"

            # health check route
            self.app.add_url_rule(f"{route_prefix}/health-check", "health_check", self.health_check, methods=["GET"])

            # UI routes
            self.app.add_url_rule(f"{route_prefix}/get-all-runs", "get_all_runs", self.get_all_runs, methods=["GET"])
            self.app.add_url_rule(f"{route_prefix}/get-run", "get_run", self.get_run, methods=["POST"])
            self.app.add_url_rule(
                f"{route_prefix}/get-all-experiment-names",
                "get_all_experiment_names",
                self.get_all_experiment_names,
                methods=["GET"],
            )
            self.app.add_url_rule(
                f"{route_prefix}/get-running-experiment",
                "get_running_experiment",
                self.get_running_experiment,
                methods=["GET"],
            )
            self.app.add_url_rule(
                f"{route_prefix}/is-experiment-running",
                "is_experiment_running",
                self.is_experiment_running,
                methods=["POST"],
            )

            # Interactive Control routes
            self.app.add_url_rule(
                f"{route_prefix}/clone-modify-run", "clone_modify_run", self.clone_modify_run, methods=["POST"]
            )
            self.app.add_url_rule(f"{route_prefix}/stop-run", "stop_run", self.stop_run, methods=["POST"])
            self.app.add_url_rule(f"{route_prefix}/resume-run", "resume_run", self.resume_run, methods=["POST"])
            self.app.add_url_rule(f"{route_prefix}/delete-run", "delete_run", self.delete_run, methods=["POST"])
            self.app.add_url_rule(
                f"{route_prefix}/get-ic-logs", "get_ic_ops_logs", self.get_ic_ops_logs, methods=["POST"]
            )
            self.app.add_url_rule(
                f"{route_prefix}/get-experiment-logs", "get_experiment_logs", self.get_experiment_logs, methods=["POST"]
            )
        except Exception as e:
            raise DispatcherException(f"Error while registering routes: {e}") from e

    # Misc routes
    def health_check(self) -> tuple[Response, int]:
        """Health check route"""
        try:
            return jsonify("Dispatcher is up and running"), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    # UI routes
    def get_all_runs(self) -> tuple[Response, int]:
        """Get all the runs for the UI"""
        try:
            results = self.db.get_all_runs()
            safe_results: list[dict[str, Any]] = []
            for run_id, result in results.items():
                # remove additional_kwargs from config_leaf
                result["config_leaf"].pop("additional_kwargs", None)

                # remove peft_params.task_type if it exists
                if "peft_params" in result["config_leaf"]:
                    result["config_leaf"]["peft_params"].pop("task_type", None)

                # remove model_kwargs.torch_dtype if it exists
                if "model_kwargs" in result["config_leaf"]:
                    result["config_leaf"]["model_kwargs"].pop("torch_dtype", None)

                if "reward_funcs" in result["config_leaf"]:
                    result["config_leaf"].pop("reward_funcs", None)

                safe_results.append(
                    {
                        "run_id": run_id,
                        "status": result["status"].value,
                        "metric_run_id": result["metric_run_id"],
                        "config": result["config_leaf"],
                        "flattened_config": result["flattened_config"],
                        "completed_steps": result["completed_steps"],
                        "total_steps": result["total_steps"],
                        "num_epochs_completed": result["num_epochs_completed"],
                        "error": result["error"],
                        "source": result["source"].value if result["source"] else None,
                        "ended_by": result["ended_by"].value if result["ended_by"] else None,
                    }
                )
            return jsonify(safe_results), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def get_run(self) -> tuple[Response, int]:
        """Get a run for the UI"""
        try:
            data = request.get_json()
            result = self.db.get_run(data["run_id"])
            if not result:
                return jsonify({"error": "Run not found"}), 404

            # remove additional_kwargs from config_leaf
            result["config_leaf"].pop("additional_kwargs", None)

            # remove peft_params.task_type if it exists
            if "peft_params" in result["config_leaf"]:
                result["config_leaf"]["peft_params"].pop("task_type", None)

            # remove model_kwargs.torch_dtype if it exists
            if "model_kwargs" in result["config_leaf"]:
                result["config_leaf"]["model_kwargs"].pop("torch_dtype", None)

            if "reward_funcs" in result["config_leaf"]:
                result["config_leaf"].pop("reward_funcs", None)

            safe_result = {
                "run_id": data["run_id"],
                "status": result["status"].value,
                "metric_run_id": result["metric_run_id"],
                "config": result["config_leaf"],
                "flattened_config": result["flattened_config"],
                "completed_steps": result["completed_steps"],
                "total_steps": result["total_steps"],
                "num_epochs_completed": result["num_epochs_completed"],
                "error": result["error"],
                "source": result["source"].value if result["source"] else None,
                "ended_by": result["ended_by"].value if result["ended_by"] else None,
            }
            return jsonify(safe_result), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def get_all_experiment_names(self) -> tuple[Response, int]:
        """Get all the experiment names for the UI"""
        try:
            results = self.db.get_all_experiment_names()
            return jsonify(results), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def get_running_experiment(self) -> tuple[Response, int]:
        """Get the running experiment for the UI"""
        try:
            result = self.db.get_running_experiment()
            return jsonify(result), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def is_experiment_running(self) -> tuple[Response, int]:
        """Check if a specific experiment is currently running.

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

    # Interactive Control routes
    def clone_modify_run(self) -> tuple[Response, int]:
        """Clone and modify a run"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data received"}), 400

            run_id = data["run_id"]
            if not run_id:
                return jsonify({"error": "run_id is required"}), 400

            if "config" not in data or not data["config"]:
                return jsonify({"error": "config is required"}), 400

            # validate and parse the ML config text
            # TODO: Implement validate_config
            # status = validate_config(data["config"])

            # get the ML config
            config = data["config"]

            # Validate the ML config text
            # data["warm_start"]: bool indicating if the run should be warm started
            task = ControllerTask.IC_CLONE_MODIFY_WARM if data["warm_start"] else ControllerTask.IC_CLONE_MODIFY

            # set create models subtask
            _ = self.db.create_ic_ops_task(
                run_id=run_id,
                ic_op=task,
                config_leaf=config,
            )

            # log the task
            logger = self._get_logger()
            if logger:
                logger.info(f"Received clone-modify task with warm start {data['warm_start']} for run_id {run_id}")
            return jsonify({}), 200
        except ValueError as ve:
            logger = self._get_logger()
            if logger:
                logger.opt(exception=True).error(f"ValueError in clone_modify_run: {ve}")
            return jsonify({"error": str(ve)}), 400
        except TypeError as te:
            logger = self._get_logger()
            if logger:
                logger.opt(exception=True).error(f"TypeError in clone_modify_run: {te}")
            return jsonify({"error": str(te)}), 400
        except Exception as e:
            logger = self._get_logger()
            if logger:
                logger.opt(exception=True).error(f"Unexpected error in clone_modify_run: {e}", exc_info=True)
            return jsonify({"error": "An unexpected error occurred", "details": str(e)}), 500

    def stop_run(self) -> tuple[Response, int]:
        """Stop the run"""
        try:
            data = request.get_json()
            run_id = data["run_id"]
            task = ControllerTask.IC_STOP

            # get ml config from db
            config_leaf = self.db.get_run(run_id)["config_leaf"]

            # create ic ops task
            _ = self.db.create_ic_ops_task(run_id, task, config_leaf)

            # log the task
            logger = self._get_logger()
            if logger:
                logger.info(f"Received stop task for run_id: {run_id}")
            return jsonify({}), 200
        except Exception as e:
            logger = self._get_logger()
            if logger:
                logger.opt(exception=True).error(f"Error in stop_run: {e}", exc_info=True)
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def resume_run(self) -> tuple[Response, int]:
        """Resume the run"""
        try:
            data = request.get_json()
            run_id = data["run_id"]
            task = ControllerTask.IC_RESUME

            # get ml config from db
            config_leaf = self.db.get_run(run_id)["config_leaf"]

            # set resume run task
            _ = self.db.create_ic_ops_task(
                run_id=run_id,
                ic_op=task,
                config_leaf=config_leaf,
            )

            # log the task
            logger = self._get_logger()
            if logger:
                logger.info(f"Received resume task for run_id: {run_id}")
            return jsonify({}), 200
        except Exception as e:
            logger = self._get_logger()
            if logger:
                logger.opt(exception=True).error(f"Error in resume_run: {e}", exc_info=True)
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def delete_run(self) -> tuple[Response, int]:
        """Delete the run"""
        try:
            data = request.get_json()
            run_id = data["run_id"]
            task = ControllerTask.IC_DELETE

            # get ml config from db
            config_leaf = self.db.get_run(run_id)["config_leaf"]

            # set delete run task
            _ = self.db.create_ic_ops_task(
                run_id=run_id,
                ic_op=task,
                config_leaf=config_leaf,
            )

            # log the task
            logger = self._get_logger()
            if logger:
                logger.info(f"Received delete task for run_id: {run_id}")
            return jsonify({}), 200
        except Exception as e:
            logger = self._get_logger()
            if logger:
                logger.opt(exception=True).error(f"Error in delete_run: {e}", exc_info=True)
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def get_ic_ops_logs(self) -> tuple[Response, int]:
        """Get the IC ops logs for the given experiment"""
        try:
            experiment_name = None
            if request.is_json:
                data = request.get_json()
                if data and data.get("experiment_name"):
                    experiment_name = data["experiment_name"]

            if not experiment_name:
                experiment_name = self.db.get_running_experiment()["experiment_name"]

            log_file_path = os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

            # Check if the log file exists
            if not os.path.exists(log_file_path):
                return jsonify({"error": f"Log file not found for experiment: {experiment_name}"}), 404

            # Read and filter logs for interactive-control entries
            interactive_control_logs = []
            with open(log_file_path,encoding="utf-8") as f:
                for line in f:
                    if f"| {experiment_name} | interactive-control |" in line:
                        interactive_control_logs.append(line.strip())

            return jsonify(interactive_control_logs), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500

    def get_experiment_logs(self) -> tuple[Response, int]:
        """Get all the logs for the given experiment"""
        # TODO: find a way to optimize this, instead of reading the entire log file, we can read the last N lines
        try:
            experiment_name = None
            if request.is_json:
                data = request.get_json()
                if data and data.get("experiment_name"):
                    experiment_name = data["experiment_name"]

            if not experiment_name:
                experiment_name = self.db.get_running_experiment()["experiment_name"]

            log_file_path = os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

            experiment_logs = []
            with open(log_file_path,encoding="utf-8") as f:
                for line in f:
                    if f"| {experiment_name} |" in line:
                        experiment_logs.append(line.strip())
                return jsonify(experiment_logs), 200
        except Exception as e:
            return jsonify({"error": str(e) + " " + str(traceback.format_exc())}), 500


def serve_forever() -> Flask:
    """start the Dispatcher via Gunicorn"""
    return Dispatcher().app


if __name__ == "__main__":
    # initialize the database tables
    rf_db = RfDb()
    rf_db.create_tables()
    print("Database tables initialized successfully")

    # start the Dispatcher on local via Flask
    dispatcher = Dispatcher()
    dispatcher.app.run(host=DispatcherConfig.HOST, port=DispatcherConfig.PORT)
