"""
RapidFire AI Metrics Module

Provides unified metric logging interfaces for different backends
(MLflow, TensorBoard, TrackIO).
"""

from rapidfireai.metrics.metric_logger import MetricLogger, MetricLoggerConfig, MetricLoggerType
from rapidfireai.metrics.metric_rfmetric_manager import RFMetricLogger

# Optional imports - these may fail if dependencies aren't installed
try:
    from rapidfireai.metrics.metric_mlflow_manager import MLflowMetricLogger
except ImportError:
    MLflowMetricLogger = None

try:
    from rapidfireai.metrics.metric_tensorboard_manager import TensorBoardMetricLogger
except ImportError:
    TensorBoardMetricLogger = None

try:
    from rapidfireai.metrics.metric_trackio_manager import TrackIOMetricLogger
except ImportError:
    TrackIOMetricLogger = None

__all__ = [
    "MetricLogger",
    "MetricLoggerConfig",
    "MetricLoggerType",
    "MLflowMetricLogger",
    "TensorBoardMetricLogger",
    "TrackIOMetricLogger",
    "RFMetricLogger",
]
