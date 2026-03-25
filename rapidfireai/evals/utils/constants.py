import os
from enum import Enum
from rapidfireai.utils.colab import get_colab_auth_token
from rapidfireai.utils.constants import DispatcherConfig, ColabConfig, RF_DB_PATH, ExperimentStatus

# ---------------------------------------------------------------------------
# Reranker class registry
# ---------------------------------------------------------------------------
# Maps the string that appears in the clone-modify dialog (the class __qualname__)
# back to the live class object so we can restore it after round-tripping through
# JSON.  Add any new reranker class here to make it available in the UI.
# ---------------------------------------------------------------------------
def _build_reranker_registry() -> dict[str, type]:
    registry: dict[str, type] = {}
    try:
        from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
        registry[CrossEncoderReranker.__qualname__] = CrossEncoderReranker
    except ImportError:
        pass
    try:
        from langchain.retrievers.document_compressors import LLMChainExtractor
        registry[LLMChainExtractor.__qualname__] = LLMChainExtractor
    except ImportError:
        pass
    try:
        from langchain_cohere import CohereRerank
        registry[CohereRerank.__qualname__] = CohereRerank
    except ImportError:
        pass
    return registry

RERANKER_CLASS_REGISTRY: dict[str, type] = _build_reranker_registry()


# RAG search type defaults — keys are type-specific; only include kwargs relevant
# to each search type so irrelevant params are never forwarded to LangChain.
VALID_SEARCH_TYPES = {"similarity", "similarity_score_threshold", "mmr"}
SEARCH_DEFAULTS: dict[str, dict] = {
    "similarity":                 {"k": 5, "filter": None},
    "similarity_score_threshold": {"k": 5, "filter": None, "score_threshold": 0.5},
    "mmr":                        {"k": 5, "filter": None, "fetch_k": 20, "lambda_mult": 0.5},
}
# Keys shown/accepted per search type (used by serialize and interactive_control)
SEARCH_TYPE_KEYS: dict[str, set] = {search_type: set(defaults.keys()) for search_type, defaults in SEARCH_DEFAULTS.items()}

# Rate Limiting Constants
# Maximum number of retries for rate-limited API calls
MAX_RATE_LIMIT_RETRIES = 5
# Base wait time for exponential backoff (seconds)
RATE_LIMIT_BACKOFF_BASE = 2

def get_dispatcher_url() -> str:
    """
    Auto-detect dispatcher URL based on environment.

    Returns:
        - In Google Colab: Uses Colab's kernel proxy URL (e.g., https://xxx-8851-xxx.ngrok-free.app)
        - In Jupyter/Local: Uses localhost URL (http://127.0.0.1:8851)
    """
    if ColabConfig.ON_COLAB:
        try:
            from google.colab.output import eval_js

            # Get the Colab proxy URL for the dispatcher port
            proxy_url = eval_js(f"google.colab.kernel.proxyPort({DispatcherConfig.PORT})")
            print(f"🌐 Google Colab detected. Dispatcher URL: {proxy_url}")
            return proxy_url
        except Exception as e:
            print(f"⚠️ Colab detected but failed to get proxy URL: {e}")
            # Fall back to localhost
            return DispatcherConfig.URL
    else:
        # Running in Jupyter, local Python, or other environment
        return DispatcherConfig.URL



def get_dispatcher_headers() -> dict[str, str]:
    """
    Get the HTTP headers needed for dispatcher API requests.

    Returns:
        Dictionary with required headers, including Authorization header in Colab
    """
    headers = {"Content-Type": "application/json"}

    auth_token = get_colab_auth_token()
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    return headers


# TODO: Merge multiple Statuses into a single Status enum

# Note: ExperimentStatus is imported from rapidfireai.utils.constants (shared)


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
        RF_DB_PATH, "rapidfire_evals.db"
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
