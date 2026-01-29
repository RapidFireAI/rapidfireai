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