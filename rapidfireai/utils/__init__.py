"""
RapidFire AI Utility functions and helpers.

This module provides unified utilities for both fit and evals modes.
"""

from rapidfireai.platform.colab import get_colab_auth_token, is_running_in_colab
from .constants import (
    # Configs
    DispatcherConfig,
    FrontendConfig,
    MLFlowConfig,
    JupyterConfig,
    RayConfig,
    ColabConfig,
    # Paths
    RF_HOME,
    RF_LOG_PATH,
    RF_LOG_FILENAME,
    RF_EXPERIMENT_PATH,
    RF_DB_PATH,
    RF_TRAINING_LOG_FILENAME,
    # Enums
    ExperimentStatus,
    RunStatus,
    PipelineStatus,
    ContextStatus,
    TaskStatus,
    ICOperation,
    ICStatus,
    ExperimentTask,
    ControllerTask,
    WorkerTask,
    RunSource,
    RunEndedBy,
    LogType,
    SHMObjectType,
    # Functions
    get_dispatcher_url,
    get_dispatcher_headers,
)
from .exceptions import (
    RFException,
    ExperimentException,
    DispatcherException,
    DBException,
    ControllerException,
    WorkerException,
    PipelineException,
    ActorException,
)
from .logging import RFLogger, TrainingLogger
from .serialize import encode_payload, decode_db_payload, extract_pipeline_config_json
from .experiment_utils import ExperimentUtils

__all__ = [
    # Colab
    "get_colab_auth_token",
    "is_running_in_colab",
    # Configs
    "DispatcherConfig",
    "FrontendConfig",
    "MLFlowConfig",
    "JupyterConfig",
    "RayConfig",
    "ColabConfig",
    # Paths
    "RF_HOME",
    "RF_LOG_PATH",
    "RF_LOG_FILENAME",
    "RF_EXPERIMENT_PATH",
    "RF_DB_PATH",
    "RF_TRAINING_LOG_FILENAME",
    # Enums
    "ExperimentStatus",
    "RunStatus",
    "PipelineStatus",
    "ContextStatus",
    "TaskStatus",
    "ICOperation",
    "ICStatus",
    "ExperimentTask",
    "ControllerTask",
    "WorkerTask",
    "RunSource",
    "RunEndedBy",
    "LogType",
    "SHMObjectType",
    # Functions
    "get_dispatcher_url",
    "get_dispatcher_headers",
    # Exceptions
    "RFException",
    "ExperimentException",
    "DispatcherException",
    "DBException",
    "ControllerException",
    "WorkerException",
    "PipelineException",
    "ActorException",
    # Logging
    "RFLogger",
    "TrainingLogger",
    # Serialization
    "encode_payload",
    "decode_db_payload",
    "extract_pipeline_config_json",
    # Experiment
    "ExperimentUtils",
]
