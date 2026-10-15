"""
Constants for the RapidFire AI package
"""
import os
from enum import Enum

from rapidfireai.utils.colab import is_running_in_colab
from rapidfireai.utils.os_utils import mkdir_p


class ExperimentStatus(str, Enum):
    """Shared status values for experiments (used by both fit and evals)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


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

def init_directories():
    """Lazily initialize runtime directories."""
    try:
        mkdir_p(RF_LOG_PATH)
        mkdir_p(RF_HOME)
    except (PermissionError, OSError) as e:
        print(f"Error creating directory: {e}")
        raise

class DispatcherConfig:
    """Class to manage the dispatcher configuration.

    Network:
        HOST / PORT / URL          - bind address for the gunicorn server

    Gunicorn tuning (read by `evals/dispatcher/gunicorn.conf.py` and
    `fit/dispatcher/gunicorn.conf.py`):
        WORKERS                    - number of worker processes. Default 1
                                     keeps the SQLite footprint minimal
                                     (single-user / Colab-style setup).
        WORKER_CLASS               - gunicorn worker type. `gthread` is the
                                     default because polling clients hold
                                     long-lived keep-alive TCP connections;
                                     the default `sync` worker blocks inside
                                     `sock.recv()` and gets SIGKILL'd by the
                                     master with WORKER TIMEOUT / "no URI
                                     read" errors.
        THREADS                    - threads per worker (only meaningful for
                                     `gthread` / async worker classes).
        TIMEOUT                    - worker heartbeat timeout in seconds.
                                     Bumped above gunicorn's 30s default to
                                     absorb the first-request SQLite
                                     migrations on a cold start.
        GRACEFUL_TIMEOUT           - seconds the master waits for a worker
                                     to finish in-flight requests on
                                     shutdown / reload.
        KEEPALIVE                  - seconds an idle keep-alive connection
                                     can stay open. Lower than gunicorn's
                                     2s default would be too aggressive;
                                     higher than ~10s lets a misbehaving
                                     client tie up a thread for too long.
    """

    HOST: str = os.getenv("RF_API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_API_PORT", "8851"))
    URL: str = f"http://{HOST}:{PORT}"

    WORKERS: int = int(os.getenv("RF_API_WORKERS", "1"))
    WORKER_CLASS: str = os.getenv("RF_API_WORKER_CLASS", "gthread")
    THREADS: int = int(os.getenv("RF_API_THREADS", "4"))
    TIMEOUT: int = int(os.getenv("RF_API_TIMEOUT", "120"))
    GRACEFUL_TIMEOUT: int = int(os.getenv("RF_API_GRACEFUL_TIMEOUT", "30"))
    KEEPALIVE: int = int(os.getenv("RF_API_KEEPALIVE", "5"))

    def __str__(self):
        return (
            f"DispatcherConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL}, "
            f"WORKERS={self.WORKERS}, WORKER_CLASS={self.WORKER_CLASS}, "
            f"THREADS={self.THREADS}, TIMEOUT={self.TIMEOUT}, "
            f"GRACEFUL_TIMEOUT={self.GRACEFUL_TIMEOUT}, KEEPALIVE={self.KEEPALIVE})"
        )

# Frontend Constants
class FrontendConfig:
    """Class to manage the frontend configuration"""

    HOST: str = os.getenv("RF_FRONTEND_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_FRONTEND_PORT", "8853"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"FrontendConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"

# MLflow Constants
class MLflowConfig:
    """Class to manage the MLflow configuration"""

    HOST: str = os.getenv("RF_MLFLOW_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_MLFLOW_PORT", "8852"))
    URL: str = f"http://{HOST}:{PORT}"

    def __str__(self):
        return f"MLflowConfig(HOST={self.HOST}, PORT={self.PORT}, URL={self.URL})"

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
