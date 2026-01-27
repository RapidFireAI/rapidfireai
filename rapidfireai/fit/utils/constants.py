import os
from enum import Enum
from rapidfireai.utils.constants import RF_TRAINING_LOG_FILENAME, RF_DB_PATH, ExperimentStatus

# Backwards compatibility: Keep constant but it will be stale if env var changes after import

# Shared Memory Constants
SHM_WARN_THRESHOLD = 80
SHM_MIN_FREE_SPACE = 1.0
USE_SHARED_MEMORY = True

# Logging Constants
TRAINING_LOG_FILENAME = RF_TRAINING_LOG_FILENAME


class LogType(Enum):
    """Enum class for log types"""

    RF_LOG = "rf_log"
    TRAINING_LOG = "training_log"


# Database Constants
class DBConfig:
    """Class to manage the database configuration for SQLite"""

    # Use user's home directory for database path

    DB_PATH: str = os.path.join(
        RF_DB_PATH, "rapidfire_fit.db"
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


# Note: ExperimentStatus is imported from rapidfireai.utils.constants (shared)


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
