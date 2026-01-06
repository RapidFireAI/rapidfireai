import os
import threading
from abc import ABC, abstractmethod

from loguru import logger

from rapidfireai.fit.db.rf_db import RfDb
from rapidfireai.utils.constants import RF_LOG_FILENAME, RF_LOG_PATH
from rapidfireai.fit.utils.constants import TRAINING_LOG_FILENAME, LogType


class BaseRFLogger(ABC):
    """Base class for RapidFire loggers"""

    _experiment_name = ""
    _initialized_loggers: dict[str, bool] = {}
    _lock = threading.Lock()

    def __init__(self, level: str = "DEBUG"):
        try:
            db = RfDb()
            experiment_name = db.get_running_experiment()["experiment_name"]        
        except Exception:
            experiment_name = "no_active_experiment"
        log_file_path = self.get_log_file_path(experiment_name)

        with BaseRFLogger._lock:
            # Reset loggers if experiment changed
            if experiment_name != BaseRFLogger._experiment_name:
                BaseRFLogger._experiment_name = experiment_name
                BaseRFLogger._initialized_loggers = {}
                logger.remove()

            # Each process gets its own handler per logger type
            logger_type = self.get_logger_type()
            logger_key = f"{logger_type.value}_{experiment_name}_{os.getpid()}"
            if logger_key not in BaseRFLogger._initialized_loggers:
                logger.add(
                    log_file_path,
                    format="{time:YYYY-MM-DD HH:mm:ss} | "
                    + "{extra[experiment_name]} | "
                    + "{extra[logger_name]} | {level} | "
                    + "{file}:{line} | {message}",
                    level=level.upper(),
                    enqueue=True,
                    filter=lambda record, logger_type=logger_type: (record["extra"].get("logger_type") == logger_type),
                )
                BaseRFLogger._initialized_loggers[logger_key] = True

    @abstractmethod
    def get_log_file_path(self, experiment_name: str):
        """Get the log file path for this logger type"""
        pass

    @abstractmethod
    def get_logger_type(self) -> LogType:
        """Get the logger type identifier"""
        pass

    def create_logger(self, name: str):
        """Create a configured logger instance"""
        return logger.bind(
            logger_name=name,
            experiment_name=BaseRFLogger._experiment_name,
            logger_type=self.get_logger_type(),
            pid=os.getpid(),
        )


class RFLogger(BaseRFLogger):
    """Standard RapidFire logger"""

    def get_log_file_path(self, experiment_name: str):
        return os.path.join(RF_LOG_PATH, experiment_name, RF_LOG_FILENAME)

    def get_logger_type(self) -> LogType:
        return LogType.RF_LOG


class TrainingLogger(BaseRFLogger):
    """Training-specific logger"""

    def get_log_file_path(self, experiment_name: str):
        return os.path.join(RF_LOG_PATH, experiment_name, TRAINING_LOG_FILENAME)

    def get_logger_type(self) -> LogType:
        return LogType.TRAINING_LOG
