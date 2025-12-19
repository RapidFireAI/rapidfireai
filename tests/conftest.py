"""
Shared pytest fixtures for RapidFire AI tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def temp_tensorboard_dir(tmp_path):
    """
    Create a temporary directory for TensorBoard logs.

    Returns:
        str: Path to temporary TensorBoard log directory
    """
    tensorboard_dir = Path(tmp_path) /  "tensorboard_logs"
    tensorboard_dir.mkdir()
    return tensorboard_dir.absolute()


@pytest.fixture
def mock_mlflow_manager():
    """
    Create a mock MLflowManager for testing.

    Returns:
        Mock: Mocked MLflowManager with all required methods
    """
    from rapidfireai.utils.mlflow_manager import MLflowManager

    mock = Mock(spec=MLflowManager)
    mock.create_run.return_value = "test_run_id"
    mock.log_param.return_value = None
    mock.log_metric.return_value = None
    mock.end_run.return_value = None
    mock.delete_run.return_value = None
    mock.get_run_metrics.return_value = {
        "loss": [(0, 1.0), (1, 0.5)],
        "accuracy": [(0, 0.8), (1, 0.9)]
    }
    mock.clear_context.return_value = None
    mock.get_experiment.return_value = "test_experiment_id"

    return mock


@pytest.fixture
def mock_summary_writer():
    """
    Create a mock TensorBoard SummaryWriter for testing.

    Returns:
        Mock: Mocked SummaryWriter with all required methods
    """
    from torch.utils.tensorboard import SummaryWriter

    mock = Mock(spec=SummaryWriter)
    mock.add_scalar.return_value = None
    mock.add_text.return_value = None
    mock.flush.return_value = None
    mock.close.return_value = None

    return mock


@pytest.fixture
def mlflow_logger(mock_mlflow_manager):
    """
    Create an MLflowMetricLogger with mocked MLflowManager.

    Returns:
        MLflowMetricLogger: Logger with mocked backend
    """
    from rapidfireai.fit.utils.metric_logger import MLflowMetricLogger

    logger = MLflowMetricLogger("http://localhost:8852")
    logger.mlflow_manager = mock_mlflow_manager

    return logger


@pytest.fixture
def tensorboard_logger(temp_tensorboard_dir):
    """
    Create a TensorBoardMetricLogger with temporary directory.

    Returns:
        TensorBoardMetricLogger: Logger with temp directory
    """
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    from rapidfireai.fit.utils.metric_logger import TensorBoardMetricLogger

    return TensorBoardMetricLogger(temp_tensorboard_dir)


@pytest.fixture
def dual_logger(mock_mlflow_manager, temp_tensorboard_dir):
    """
    Create a DualMetricLogger with mocked MLflow and temp TensorBoard.

    Returns:
        DualMetricLogger: Logger with both backends
    """
    from rapidfireai.fit.utils.metric_logger import DualMetricLogger

    logger = DualMetricLogger("http://localhost:8852", temp_tensorboard_dir)
    logger.mlflow_logger.mlflow_manager = mock_mlflow_manager

    return logger


@pytest.fixture
def sample_metrics():
    """
    Sample metrics data for testing.

    Returns:
        dict: Sample metrics with various types
    """
    return {
        "loss": 0.5,
        "accuracy": 0.95,
        "learning_rate": 1e-4,
        "epoch": 1,
    }


@pytest.fixture
def sample_params():
    """
    Sample parameters data for testing.

    Returns:
        dict: Sample hyperparameters
    """
    return {
        "model_name": "TinyLlama",
        "batch_size": "4",
        "learning_rate": "1e-3",
        "optimizer": "adamw",
    }
