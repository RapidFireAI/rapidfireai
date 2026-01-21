"""
RapidFire AI Logging Module

Provides Python standard logging with Ray-compatible file handling.
"""

import logging
import os
import threading
from pathlib import Path

from rapidfireai.utils.constants import (
    RF_EXPERIMENT_PATH,
    RF_LOG_FILENAME,
    RF_LOG_PATH,
    RF_TRAINING_LOG_FILENAME,
    LogType,
)
from rapidfireai.utils.os_utils import mkdir_p


class AsyncioCleanupFilter(logging.Filter):
    """Filter out harmless asyncio cleanup errors during shutdown."""

    def filter(self, record):
        msg = str(record.getMessage())
        # Suppress TCPTransport cleanup errors
        if "TCPTransport" in msg and "closed=True" in msg:
            return False
        # Suppress Task exception errors for httpx/OpenAI client cleanup
        if "Task exception was never retrieved" in msg:
            if "AsyncClient.aclose" in msg:
                return False
        return True


# Third-party loggers to suppress
THIRD_PARTY_LOGGERS = [
    "ray",
    "vllm",
    "torch",
    "transformers",
    "datasets",
    "huggingface_hub",
    "bitsandbytes",
    "peft",
    "trl",
    "accelerate",
    "langchain",
    "langchain_core",
    "langchain_community",
    "openai",
    "httpx",
    "urllib3",
    "filelock",
]


class RFLogger:
    """
    RapidFire logger using Python standard logging.
    
    Features:
    - Ray worker detection (avoids file handler conflicts)
    - Thread-safe initialization
    - Third-party log suppression
    """

    # Class-level state shared across instances
    _file_handlers: dict[str, logging.FileHandler] = {}
    _initialized_experiments: set[str] = set()
    _lock = threading.Lock()

    def __init__(
        self,
        experiment_name: str = "unknown",
        experiment_path: str = RF_EXPERIMENT_PATH,
        level: str = "INFO",
        log_type: LogType = LogType.RF_LOG,
    ):
        self._experiment_name = experiment_name
        self._experiment_path = experiment_path
        self.level = level.upper()
        self.log_type = log_type

        # Suppress third-party logs via environment variables
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("RAY_LOG_TO_STDERR", "0")

        # Only set up file handlers on main process (not Ray workers)
        is_ray_worker = os.environ.get("RAY_WORKER_MODE") == "WORKER"
        if not is_ray_worker:
            self._setup_file_handler()

    def _get_log_file_path(self) -> str:
        """Get the log file path based on log type."""
        if self.log_type == LogType.TRAINING_LOG:
            return os.path.join(RF_LOG_PATH, self._experiment_name, RF_TRAINING_LOG_FILENAME)
        return os.path.join(RF_LOG_PATH, self._experiment_name, RF_LOG_FILENAME)

    def _setup_file_handler(self):
        """Set up file handler for this logger (thread-safe)."""
        with RFLogger._lock:
            handler_key = f"{self.log_type.value}_{self._experiment_name}"

            # Skip if already initialized
            if handler_key in RFLogger._file_handlers:
                return

            # Create log directory
            log_dir = Path(RF_LOG_PATH) / self._experiment_name
            mkdir_p(log_dir.absolute())

            # Create file handler
            log_file_path = self._get_log_file_path()
            log_format = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
            file_handler.setLevel(self.level)

            RFLogger._file_handlers[handler_key] = file_handler

            # Configure root logger once per experiment
            if self._experiment_name not in RFLogger._initialized_experiments:
                RFLogger._initialized_experiments.add(self._experiment_name)

                root_logger = logging.getLogger()

                # Remove existing handlers (prevent console output)
                for handler in root_logger.handlers[:]:
                    root_logger.removeHandler(handler)

                root_logger.setLevel(self.level)
                root_logger.addFilter(AsyncioCleanupFilter())

                # Suppress asyncio logger
                asyncio_logger = logging.getLogger("asyncio")
                asyncio_logger.addFilter(AsyncioCleanupFilter())
                asyncio_logger.setLevel(logging.CRITICAL)

                # Suppress third-party loggers
                for logger_name in THIRD_PARTY_LOGGERS:
                    logging.getLogger(logger_name).setLevel(logging.CRITICAL)
                    logging.getLogger(logger_name).propagate = False

            # Add handler to root logger
            logging.getLogger().addHandler(file_handler)

    def get_logger(self, name: str = "unknown") -> logging.Logger:
        """
        Get a configured logger instance.

        Args:
            name: Name for the logger (e.g., 'controller', 'worker_0')

        Returns:
            A standard Python logger
        """
        logger = logging.getLogger(f"{self._experiment_name}:{name}")
        logger.setLevel(self.level)
        return logger


class TrainingLogger(RFLogger):
    """Training-specific logger that writes to training.log instead of rapidfire.log."""

    def __init__(
        self,
        experiment_name: str = "unknown",
        experiment_path: str = RF_EXPERIMENT_PATH,
        level: str = "DEBUG",
    ):
        super().__init__(
            experiment_name=experiment_name,
            experiment_path=experiment_path,
            level=level,
            log_type=LogType.TRAINING_LOG,
        )


__all__ = ["RFLogger", "TrainingLogger"]
