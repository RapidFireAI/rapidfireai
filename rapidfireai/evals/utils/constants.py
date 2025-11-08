from enum import Enum
import os
# Logging Constants
LOG_FILENAME = "rapidfire.log"

# Actor Constants
NUM_QUERY_PROCESSING_ACTORS = 4
NUM_CPUS_PER_DOC_ACTOR = 2

# Rate Limiting Constants
# Maximum number of retries for rate-limited API calls
MAX_RATE_LIMIT_RETRIES = 5
# Base wait time for exponential backoff (seconds)
RATE_LIMIT_BACKOFF_BASE = 2

class DispatcherConfig:
    """Class to manage the dispatcher configuration"""

    HOST: str = os.getenv("RF_API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_API_PORT", "8851"))
    URL: str = f"http://{HOST}:{PORT}"


def get_dispatcher_url() -> str:
    """
    Auto-detect dispatcher URL based on environment.

    Returns:
        - In Colab: Uses Colab's kernel proxy URL (e.g., https://xxx-8851-xxx.ngrok-free.app)
        - In Local: Uses localhost URL (http://127.0.0.1:8851)
    """
    try:
        # Check if running in Google Colab
        import google.colab
        from google.colab.output import eval_js

        # Get the Colab proxy URL for the dispatcher port
        proxy_url = eval_js(f"google.colab.kernel.proxyPort({DispatcherConfig.PORT})")
        print(f"🌐 Colab environment detected. Dispatcher URL: {proxy_url}")
        return proxy_url
    except ImportError:
        # Not in Colab - use localhost
        local_url = DispatcherConfig.URL
        return local_url


# TODO: Merge multiple Statuses into a single Status enum


# Status Enums
class ExperimentStatus(str, Enum):
    """Status values for experiments."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContextStatus(str, Enum):
    """Status values for RAG contexts."""

    NEW = "new"
    ONGOING = "ongoing"
    DELETED = "deleted"
    FAILED = "failed"


class PipelineStatus(str, Enum):
    """Status values for pipelines."""

    NEW = "new"
    ONGOING = "ongoing"
    COMPLETED = "completed"
    STOPPED = "stopped"
    DELETED = "deleted"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status values for actor tasks."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ICOperation(str, Enum):
    """Interactive Control operation types."""

    STOP = "stop"
    RESUME = "resume"
    DELETE = "delete"
    CLONE = "clone"


class ICStatus(str, Enum):
    """Status values for Interactive Control operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Database Constants
class DBConfig:
    """Class to manage the database configuration for SQLite"""

    # Use user's home directory for database path

    DB_PATH: str = os.path.join(
        os.getenv("RF_DB_PATH", os.path.expanduser(os.path.join("~", "db"))), "rapidfire_evals.db"
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
