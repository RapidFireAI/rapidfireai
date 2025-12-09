import logging
import os
from pathlib import Path

from rapidfireai.utils.constants import RF_LOG_FILENAME, RF_LOG_PATH, RF_EXPERIMENT_PATH
from rapidfireai.utils.os_utils import mkdir_p


class RFLogger:
    _file_handler = None
    _experiment_name = None
    _experiment_path = None

    def __init__(
        self, experiment_name: str = "unknown", experiment_path: str = RF_EXPERIMENT_PATH, level: str = "INFO"
    ):
        self._experiment_name = experiment_name
        self._experiment_path = experiment_path
        self.level = level.upper()

        # Suppress third-party library logs via environment variables
        os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
        os.environ.setdefault("RAY_LOG_TO_STDERR", "0")

        # Check if we are in a Ray worker process
        is_ray_worker = os.environ.get("RAY_WORKER_MODE") == "WORKER"

        # Only set up the file handler on the Controller/Experiment process
        # Ray workers will forward their logs to the driver's logging system
        if not is_ray_worker and RFLogger._file_handler is None:
            log_dir = Path(RF_LOG_PATH) / self._experiment_name
            try:
                mkdir_p(log_dir.absolute())
            except (PermissionError, OSError) as e:
                print(f"Error creating directory: {e}")
                raise

            # Use standard format fields only - LoggerAdapter will prefix messages
            log_format = "%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s"

            # Set up the file handler
            log_file_path = log_dir / RF_LOG_FILENAME
            RFLogger._file_handler = logging.FileHandler(log_file_path)
            RFLogger._file_handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%d %H:%M:%S"))
            RFLogger._file_handler.setLevel(self.level)

            # Get root logger
            root_logger = logging.getLogger()

            # Remove all existing handlers to prevent console output
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)

            # Add only our file handler to the root logger
            root_logger.addHandler(RFLogger._file_handler)
            root_logger.setLevel(self.level)

            # Add filter to suppress harmless asyncio cleanup errors
            class AsyncioCleanupFilter(logging.Filter):
                """Filter out harmless asyncio cleanup errors during shutdown."""
                def filter(self, record):
                    # Suppress TCPTransport cleanup errors
                    if "TCPTransport" in str(record.getMessage()) and "closed=True" in str(record.getMessage()):
                        return False
                    # Suppress "Task exception was never retrieved" for these specific cases
                    if "Task exception was never retrieved" in str(record.getMessage()):
                        # Only suppress if it's related to httpx/OpenAI client cleanup
                        if "AsyncClient.aclose" in str(record.getMessage()):
                            return False
                    return True

            # Add the filter to the root logger
            root_logger.addFilter(AsyncioCleanupFilter())

            # Also suppress asyncio logger specifically
            asyncio_logger = logging.getLogger("asyncio")
            asyncio_logger.addFilter(AsyncioCleanupFilter())
            asyncio_logger.setLevel(logging.CRITICAL)

            # Suppress third-party library logs more aggressively
            third_party_loggers = [
                "ray",
                "vllm",
                "torch",
                "transformers",
                "datasets",
                "huggingface_hub",
                "langchain",
                "langchain_core",
                "langchain_community",
                "openai",
                "httpx",
                "urllib3",
            ]
            for logger_name in third_party_loggers:
                logging.getLogger(logger_name).setLevel(logging.CRITICAL)
                logging.getLogger(logger_name).propagate = False

    def get_logger(self, logger_name: str = "unknown"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(self.level)

        # Custom LoggerAdapter that prefixes messages instead of using format fields
        class SafeLoggerAdapter(logging.LoggerAdapter):
            def process(self, msg, kwargs):
                # Prefix message with experiment and logger name
                experiment = self.extra.get("experiment_name", "unknown")
                log_name = self.extra.get("logger_name", "unknown")
                return f"[{experiment}:{log_name}] {msg}", kwargs

        return SafeLoggerAdapter(
            logger,
            {
                "experiment_name": self._experiment_name,
                "logger_name": logger_name,
            },
        )

