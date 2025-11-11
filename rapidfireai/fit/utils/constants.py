import os
from enum import Enum

# Tracking Backend Configuration
def get_tracking_backend() -> str:
    """
    Get the tracking backend from environment variable at runtime.

    Returns:
        str: The tracking backend ('mlflow', 'tensorboard', or 'both')

    Note: This reads from os.environ at runtime to allow setting the env var
    after module import (important for notebook environments like Colab).
    """
    backend = os.getenv("RF_TRACKING_BACKEND", "mlflow")
    return backend


# Backwards compatibility: Keep constant but it will be stale if env var changes after import
TRACKING_BACKEND = get_tracking_backend()  # Options: 'mlflow', 'tensorboard', 'both'
TENSORBOARD_LOG_DIR = os.getenv("RF_TENSORBOARD_LOG_DIR", None)  # Default set by experiment path

# Shared Memory Constants
SHM_WARN_THRESHOLD = 80
SHM_MIN_FREE_SPACE = 1.0
USE_SHARED_MEMORY = True

# Logging Constants
LOG_FILENAME = "rapidfire.log"
TRAINING_LOG_FILENAME = "training.log"


class LogType(Enum):
    """Enum class for log types"""

    RF_LOG = "rf_log"
    TRAINING_LOG = "training_log"


# Dispatcher Constants
class DispatcherConfig:
    """Class to manage the dispatcher configuration"""

    HOST: str = os.getenv("RF_API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_API_PORT", "8851"))
    URL: str = f"http://{HOST}:{PORT}"

# MLFlow Constants
class MLFlowConfig:
    """Class to manage the MLFlow configuration"""

    HOST: str = os.getenv("RF_MLFLOW_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_MLFLOW_PORT", "8852"))
    URL: str = f"http://{HOST}:{PORT}"

# Frontend Constants
class FrontendConfig:
    """Class to manage the frontend configuration"""

    HOST: str = os.getenv("RF_FRONTEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_FRONTEND_PORT", "8853"))
    URL: str = f"http://{HOST}:{PORT}"

# Database Constants
class DBConfig:
    """Class to manage the database configuration for SQLite"""

    # Use user's home directory for database path

    DB_PATH: str = os.path.join(
        os.getenv("RF_DB_PATH", os.path.expanduser(os.path.join("~", "db"))), "rapidfire_fit.db"
    )

    # Connection settings
    CONNECTION_TIMEOUT: float = 30.0

    # Performance optimizations
    CACHE_SIZE: int = 10000
    MMAP_SIZE: int = 268435456  # 256MB
    PAGE_SIZE: int = 4096
    BUSY_TIMEOUT: int = 30000

    # Retry settings
    DEFAULT_MAX_RETRIES: int = 3
    DEFAULT_BASE_DELAY: float = 0.1
    DEFAULT_MAX_DELAY: float = 1.0


# Experiment Constants
class ExperimentTask(Enum):
    """Enum class for experiment tasks"""

    IDLE = "Idle"
    CREATE_MODELS = "Create Models"
    IC_OPS = "Interactive Control Operations"
    RUN_FIT = "Training and Validation"


class ExperimentStatus(Enum):
    """Enum class for experiment status"""

    RUNNING = "Running"
    COMPLETED = "Completed"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


# Task Constants
class TaskStatus(Enum):
    """Enum class for task status"""

    SCHEDULED = "Scheduled"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    SKIPPED = "Skipped"
    FAILED = "Failed"


class ControllerTask(Enum):
    """Enum class for ML controller tasks"""

    RUN_FIT = "Run Fit"
    CREATE_MODELS = "Create Models"
    IC_DELETE = "Interactive Control Delete"
    IC_STOP = "Interactive Control Stop"
    IC_RESUME = "Interactive Control Resume"
    IC_CLONE_MODIFY = "Interactive Control Clone Modify"
    IC_CLONE_MODIFY_WARM = "Interactive Control Clone Modify with Warm-Start"
    EPOCH_BOUNDARY = "Epoch Boundary Task"
    GET_RUN_METRICS = "Get Run Metrics"


class WorkerTask(Enum):
    """Enum class for ML worker tasks"""

    CREATE_MODELS = "Create Models"
    TRAIN_VAL = "Train Validation"


# Run Constants
class RunStatus(Enum):
    """Enum class for run status"""

    NEW = "New"
    ONGOING = "Ongoing"
    STOPPED = "Stopped"
    DELETED = "Deleted"
    COMPLETED = "Completed"
    FAILED = "Failed"


class RunSource(Enum):
    """Enum class for how a run was created"""

    SHA = "Successive Halving Algorithm"
    INITIAL = "Initial"
    INTERACTIVE_CONTROL = "Interactive Control"


class RunEndedBy(Enum):
    """Enum class for how a run was ended"""

    SHA = "Successive Halving Algorithm"
    EPOCH_COMPLETED = "Epoch Completed"
    INTERACTIVE_CONTROL = "Interactive Control"
    TOLERENCE = "Tolerence Threshold Met"


# SHM Model Type Constants
class SHMObjectType(Enum):
    """Enum class for model types in shared memory"""

    BASE_MODEL = "base_model"
    FULL_MODEL = "full_model"
    REF_FULL_MODEL = "ref_full_model"
    REF_STATE_DICT = "ref_state_dict"
    CHECKPOINTS = "checkpoints"
