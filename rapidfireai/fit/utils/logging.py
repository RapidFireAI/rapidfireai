"""
RapidFire Fit Logging Module

Provides Python standard logging with Ray-compatible file handling.
Replaces loguru for compatibility with Ray actors.
"""

import logging
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.fit.utils.constants import TRAINING_LOG_FILENAME, LogType
from rapidfireai.utils.constants import RF_LOG_FILENAME, RF_LOG_PATH
from rapidfireai.utils.os_utils import mkdir_p


class BaseRFLogger(ABC):
    """Base class for RapidFire loggers using Python standard logging."""

    # Class-level state shared across instances
    _file_handlers: dict[str, logging.FileHandler] = {}
    _experiment_name: str = ""
    _lock = threading.Lock()

    def __init__(self, level: str = "DEBUG"):
        """
        Initialize the logger.

        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.level = level.upper()

        # Get experiment name from database
        try:
            db = RfDb()
            experiment_name = db.get_running_experiment()["experiment_name"]
            self._experiment_name = experiment_name
        except Exception as e:
            raise RuntimeError("Error getting experiment name from database") from e

        # Suppress third-party library logs via environment variables
        os.environ.setdefault("RAY_LOG_TO_STDERR", "0")

        # Check if we are in a Ray worker process
        is_ray_worker = os.environ.get("RAY_WORKER_MODE") == "WORKER"

        # Only set up file handlers on the main process (not Ray workers)
        # Ray workers will forward their logs to the driver's logging system
        if not is_ray_worker:
            self._setup_file_handler()

    def _setup_file_handler(self):
        """Set up file handler for this logger type."""
        with BaseRFLogger._lock:
            logger_type = self.get_logger_type()
            handler_key = f"{logger_type.value}_{self._experiment_name}"

            # Skip if already initialized for this experiment and logger type
            if handler_key in BaseRFLogger._file_handlers:
                return

            # Reset handlers if experiment changed
            if self._experiment_name != BaseRFLogger._experiment_name:
                BaseRFLogger._experiment_name = self._experiment_name
                # Close existing handlers
                for handler in BaseRFLogger._file_handlers.values():
                    handler.close()
                BaseRFLogger._file_handlers = {}

            # Create log directory
            log_dir = Path(RF_LOG_PATH) / self._experiment_name
            try:
                mkdir_p(log_dir.absolute())
            except (PermissionError, OSError) as e:
                print(f"Error creating log directory: {e}")
                raise

            # Set up the file handler
            log_file_path = self.get_log_file_path(self._experiment_name)
            log_format = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
            file_handler.setLevel(self.level)

            # Store handler
            BaseRFLogger._file_handlers[handler_key] = file_handler

            # Configure root logger (only once)
            if len(BaseRFLogger._file_handlers) == 1:
                root_logger = logging.getLogger()

                # Remove all existing handlers to prevent console output
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)

                root_logger.setLevel(self.level)

                # Suppress third-party library logs
                third_party_loggers = [
                    "ray",
                    "torch",
                    "transformers",
                    "datasets",
                    "huggingface_hub",
                    "bitsandbytes",
                    "peft",
                    "trl",
                    "accelerate",
                    "urllib3",
                    "filelock",
                ]
                for logger_name in third_party_loggers:
                    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
                    logging.getLogger(logger_name).propagate = False

            # Add handler to root logger
            logging.getLogger().addHandler(file_handler)

    @abstractmethod
    def get_log_file_path(self, experiment_name: str) -> str:
        """Get the log file path for this logger type."""
        pass

    @abstractmethod
    def get_logger_type(self) -> LogType:
        """Get the logger type identifier."""
        pass

    def create_logger(self, name: str):
        """
        Create a configured logger instance.

        Args:
            name: Name for the logger (e.g., 'controller', 'worker_0')

        Returns:
            A logging.LoggerAdapter with experiment context
        """
        base_logger = logging.getLogger(name)
        base_logger.setLevel(self.level)

        # Custom LoggerAdapter that prefixes messages with experiment and logger name
        # Also provides .opt() method for backwards compatibility with loguru-style calls
        class SafeLoggerAdapter(logging.LoggerAdapter):
            """LoggerAdapter that adds experiment context to log messages."""

            def __init__(self, logger, extra):
                super().__init__(logger, extra)
                self._exc_info = False

            def process(self, msg, kwargs):
                experiment = self.extra.get("experiment_name", "unknown")
                log_name = self.extra.get("logger_name", "unknown")
                return f"[{experiment}:{log_name}] {msg}", kwargs

            def opt(self, exception: bool = False, **kwargs):
                """
                Backwards-compatible method for loguru-style .opt(exception=True) calls.

                Returns self with exception flag set, so .opt(exception=True).error(msg)
                will log with exception info.
                """
                self._exc_info = exception
                return self

            def error(self, msg, *args, **kwargs):
                """Override error to support .opt(exception=True) pattern."""
                if self._exc_info:
                    kwargs["exc_info"] = True
                    self._exc_info = False  # Reset for next call
                super().error(msg, *args, **kwargs)

            def warning(self, msg, *args, **kwargs):
                """Override warning to support .opt(exception=True) pattern."""
                if self._exc_info:
                    kwargs["exc_info"] = True
                    self._exc_info = False
                super().warning(msg, *args, **kwargs)

            def info(self, msg, *args, **kwargs):
                """Override info to support .opt(exception=True) pattern."""
                if self._exc_info:
                    kwargs["exc_info"] = True
                    self._exc_info = False
                super().info(msg, *args, **kwargs)

            def debug(self, msg, *args, **kwargs):
                """Override debug to support .opt(exception=True) pattern."""
                if self._exc_info:
                    kwargs["exc_info"] = True
                    self._exc_info = False
                super().debug(msg, *args, **kwargs)

        return SafeLoggerAdapter(
            base_logger,
            {
                "experiment_name": self._experiment_name,
                "logger_name": name,
            },
        )

    # Alias for compatibility with evals-style API
    def get_logger(self, logger_name: str = "unknown"):
        """Alias for create_logger() for API compatibility with evals."""
        return self.create_logger(logger_name)


class RFLogger(BaseRFLogger):
    """Standard RapidFire logger for general logging."""

    def get_log_file_path(self, experiment_name: str) -> str:
        """Get the main log file path."""
        return os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

    def get_logger_type(self) -> LogType:
        """Return the RF_LOG type."""
        return LogType.RF_LOG


class TrainingLogger(BaseRFLogger):
    """Training-specific logger for training output."""

    def get_log_file_path(self, experiment_name: str) -> str:
        """Get the training log file path."""
        return os.path.join(RF_LOG_PATH, experiment_name, TRAINING_LOG_FILENAME)

    def get_logger_type(self) -> LogType:
        """Return the TRAINING_LOG type."""
        return LogType.TRAINING_LOG
