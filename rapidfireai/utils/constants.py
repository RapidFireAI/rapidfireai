"""
Constants for the RapidFire AI package

This module contains constants, configuration classes, and enums.
"""

import os
from enum import Enum

from rapidfireai.platform.colab import is_running_in_colab
from rapidfireai.utils.os_utils import mkdir_p


if is_running_in_colab():
    RF_HOME = "/content/rapidfireai"
else:
    RF_HOME = os.getenv("RF_HOME", os.path.join(os.path.expanduser("~"), "rapidfireai"))
RF_LOG_FILENAME = os.getenv("RF_LOG_FILENAME", "rapidfire.log")
RF_LOG_PATH = os.getenv("RF_LOG_PATH", os.path.join(RF_HOME, "logs"))
RF_EXPERIMENT_PATH = os.getenv("RF_EXPERIMENT_PATH", os.path.join(RF_HOME, "rapidfire_experiments"))
RF_DB_PATH = os.getenv("RF_DB_PATH", os.path.expanduser(os.path.join(RF_HOME, "db")))
RF_TENSORBOARD_LOG_DIR = os.getenv("RF_TENSORBOARD_LOG_DIR", f"{RF_EXPERIMENT_PATH}/tensorboard_logs")
RF_TRAINING_LOG_FILENAME = os.getenv("RF_TRAINING_LOG_FILENAME", "training.log")
RF_TRAINER_OUTPUT = os.getenv("RF_TRAINER_OUTPUT", os.path.join(RF_HOME, "trainer_output"))

# Evals Actor Constants
NUM_QUERY_PROCESSING_ACTORS = 4
NUM_CPUS_PER_DOC_ACTOR = 2 if os.cpu_count() > 2 else 1

# Rate Limiting Constants
MAX_RATE_LIMIT_RETRIES = 5
RATE_LIMIT_BACKOFF_BASE = 2

try:
    mkdir_p(RF_LOG_PATH)
    mkdir_p(RF_HOME)
except (PermissionError, OSError) as e:
    print(f"Error creating directory: {e}")
    raise


class DispatcherConfig:
    """Class to manage the dispatcher configuration"""

    HOST: str = os.getenv("RF_API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_API_PORT", "8851"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"DispatcherConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"


# Frontend Constants
class FrontendConfig:
    """Class to manage the frontend configuration"""

    HOST: str = os.getenv("RF_FRONTEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_FRONTEND_PORT", "8853"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"FrontendConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"


# MLFlow Constants
class MLFlowConfig:
    """Class to manage the MLFlow configuration"""

    HOST: str = os.getenv("RF_MLFLOW_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_MLFLOW_PORT", "8852"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"MLFlowConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"


# Jupyter Constants
class JupyterConfig:
    """Class to manage the Jupyter configuration"""

    HOST: str = os.getenv("RF_JUPYTER_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_JUPYTER_PORT", "8850"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"JupyterConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"


# Ray Constants
class RayConfig:
    """Class to manage the Ray configuration"""

    HOST: str = os.getenv("RF_RAY_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("RF_RAY_PORT", "8855"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"RayConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"


# Colab Constants
class ColabConfig:
    """Class to manage the Colab configuration"""

    ON_COLAB: bool = is_running_in_colab()
    RF_COLAB_MODE: str = os.getenv("RF_COLAB_MODE", str(ON_COLAB)).lower()

    def __str__(self):
        return f"ColabConfig(ON_COLAB={self.ON_COLAB}, RF_COLAB_MODE={self.RF_COLAB_MODE})"


RF_MLFLOW_ENABLED = os.getenv("RF_MLFLOW_ENABLED", "true" if not ColabConfig.ON_COLAB else "false")
RF_TENSORBOARD_ENABLED = os.getenv("RF_TENSORBOARD_ENABLED", "false" if not ColabConfig.ON_COLAB else "true")
RF_TRACKIO_ENABLED = os.getenv("RF_TRACKIO_ENABLED", "false")
RF_TRACKING_BACKEND = os.getenv("RF_TRACKING_BACKEND", "mlflow" if not ColabConfig.ON_COLAB else "tensorboard")


# ============================================================================
# UNIFIED STATUS ENUMS (lowercase values for consistency)
# ============================================================================


class ExperimentStatus(str, Enum):
    """Status for experiments (both fit and evals)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunStatus(str, Enum):
    """Status for fit training runs."""

    NEW = "new"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    STOPPED = "stopped"
    DELETED = "deleted"
    FAILED = "failed"


class PipelineStatus(str, Enum):
    """Status for evals inference pipelines."""

    NEW = "new"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    STOPPED = "stopped"
    DELETED = "deleted"
    FAILED = "failed"


class ContextStatus(str, Enum):
    """Status for RAG contexts (evals mode)."""

    NEW = "new"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    DELETED = "deleted"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status for worker tasks (fit) and actor tasks (evals)."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"  # Fit-specific


class ICOperation(str, Enum):
    """Interactive control operation types."""

    STOP = "stop"
    RESUME = "resume"
    DELETE = "delete"
    CLONE = "clone"
    CLONE_WARM = "clone_warm"  # Fit-specific (warm-start)


class ICStatus(str, Enum):
    """Status for interactive control operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


# ============================================================================
# FIT-SPECIFIC ENUMS
# ============================================================================


class ExperimentTask(str, Enum):
    """Fit-mode experiment tasks (current_task column)."""

    IDLE = "idle"
    CREATE_MODELS = "create_models"
    IC_OPS = "ic_ops"
    RUN_FIT = "run_fit"


class ControllerTask(str, Enum):
    """Fit-mode controller tasks."""

    RUN_FIT = "run_fit"
    CREATE_MODELS = "create_models"
    IC_DELETE = "ic_delete"
    IC_STOP = "ic_stop"
    IC_RESUME = "ic_resume"
    IC_CLONE_MODIFY = "ic_clone_modify"
    IC_CLONE_MODIFY_WARM = "ic_clone_modify_warm"
    EPOCH_BOUNDARY = "epoch_boundary"
    GET_RUN_METRICS = "get_run_metrics"


class WorkerTask(str, Enum):
    """Fit-mode worker tasks."""

    CREATE_MODELS = "create_models"
    TRAIN_VAL = "train_val"


class RunSource(str, Enum):
    """How a fit run was created."""

    SHA = "sha"  # Successive Halving Algorithm
    INITIAL = "initial"
    INTERACTIVE_CONTROL = "interactive_control"


class RunEndedBy(str, Enum):
    """How a fit run was ended."""

    SHA = "sha"  # Successive Halving Algorithm
    EPOCH_COMPLETED = "epoch_completed"
    INTERACTIVE_CONTROL = "interactive_control"
    TOLERANCE = "tolerance"


class LogType(str, Enum):
    """Log file types."""

    RF_LOG = "rf_log"
    TRAINING_LOG = "training_log"


class SHMObjectType(str, Enum):
    """Types of objects stored in shared memory (fit mode)."""

    BASE_MODEL = "base_model"
    FULL_MODEL = "full_model"
    REF_FULL_MODEL = "ref_full_model"
    REF_STATE_DICT = "ref_state_dict"
    CHECKPOINTS = "checkpoints"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_dispatcher_url() -> str:
    """
    Auto-detect dispatcher URL based on environment.

    Returns:
        - In Google Colab: Uses Colab's kernel proxy URL
        - In Jupyter/Local: Uses localhost URL
    """
    if ColabConfig.ON_COLAB:
        try:
            from google.colab.output import eval_js

            proxy_url = eval_js(f"google.colab.kernel.proxyPort({DispatcherConfig.PORT})")
            print(f"🌐 Google Colab detected. Dispatcher URL: {proxy_url}")
            return proxy_url
        except Exception as e:
            print(f"⚠️ Colab detected but failed to get proxy URL: {e}")
            return DispatcherConfig.URL
    else:
        return DispatcherConfig.URL


def get_dispatcher_headers() -> dict[str, str]:
    """
    Get the HTTP headers needed for dispatcher API requests.

    Returns:
        Dictionary with required headers, including Authorization header in Colab
    """
    from rapidfireai.platform.colab import get_colab_auth_token

    headers = {"Content-Type": "application/json"}

    auth_token = get_colab_auth_token()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    return headers
