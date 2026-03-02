"""
Comprehensive test suite for metric logging abstraction layer.

Tests the MetricLogger interface and implementations:
- MLflowMetricLogger
- TensorBoardMetricLogger
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pytest

from rapidfireai.metrics.metric_logger import MetricLogger
from rapidfireai.metrics.metric_mlflow_manager import MLflowMetricLogger
from rapidfireai.metrics.metric_tensorboard_manager import TensorBoardMetricLogger


class TestMLflowMetricLogger:
    """Test suite for MLflowMetricLogger."""

    def test_create_run(self, mlflow_logger, mock_mlflow_manager):
        """Test that create_run calls MLflowManager.create_run()."""
        run_id = mlflow_logger.create_run("test_run")

        mock_mlflow_manager.create_run.assert_called_once_with("test_run")
        assert run_id == "test_run_id"

    def test_log_param(self, mlflow_logger, mock_mlflow_manager):
        """Test that log_param calls MLflowManager.log_param()."""
        mlflow_logger.log_param("run_1", "learning_rate", "1e-3")

        mock_mlflow_manager.log_param.assert_called_once_with("run_1", "learning_rate", "1e-3")

    def test_log_metric_with_step(self, mlflow_logger, mock_mlflow_manager):
        """Test logging metric with explicit step."""
        mlflow_logger.log_metric("run_1", "loss", 0.5, step=100)

        mock_mlflow_manager.log_metric.assert_called_once_with("run_1", "loss", 0.5, step=100)

    def test_log_metric_without_step(self, mlflow_logger, mock_mlflow_manager):
        """Test logging metric without step (defaults to None)."""
        mlflow_logger.log_metric("run_1", "accuracy", 0.95)

        mock_mlflow_manager.log_metric.assert_called_once_with("run_1", "accuracy", 0.95, step=None)

    def test_end_run(self, mlflow_logger, mock_mlflow_manager):
        """Test that end_run calls MLflowManager.end_run()."""
        mlflow_logger.end_run("run_1")

        mock_mlflow_manager.end_run.assert_called_once_with("run_1")

    def test_delete_run(self, mlflow_logger, mock_mlflow_manager):
        """Test that delete_run calls MLflowManager.delete_run()."""
        mlflow_logger.delete_run("run_1")

        mock_mlflow_manager.delete_run.assert_called_once_with("run_1")

    def test_get_run_metrics(self, mlflow_logger, mock_mlflow_manager):
        """Test that get_run_metrics calls MLflowManager.get_run_metrics()."""
        metrics = mlflow_logger.get_run_metrics("run_1")

        mock_mlflow_manager.get_run_metrics.assert_called_once_with("run_1")
        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_clear_context(self, mlflow_logger, mock_mlflow_manager):
        """Test that clear_context calls MLflowManager.clear_context()."""
        mlflow_logger.clear_context()

        mock_mlflow_manager.clear_context.assert_called_once()

    def test_get_experiment(self, mlflow_logger, mock_mlflow_manager):
        """Test that get_experiment calls MLflowManager.get_experiment()."""
        exp_id = mlflow_logger.get_experiment("test_experiment")

        mock_mlflow_manager.get_experiment.assert_called_once_with("test_experiment")
        assert exp_id == "test_experiment_id"

    def test_multiple_log_calls(self, mlflow_logger, mock_mlflow_manager):
        """Test multiple log calls in sequence."""
        mlflow_logger.log_metric("run_1", "loss", 1.0, step=0)
        mlflow_logger.log_metric("run_1", "loss", 0.8, step=1)
        mlflow_logger.log_metric("run_1", "loss", 0.6, step=2)

        assert mock_mlflow_manager.log_metric.call_count == 3


class TestTensorBoardMetricLogger:
    """Test suite for TensorBoardMetricLogger."""

    def test_creates_log_directory(self, tmp_path):
        """Test that logger creates log directory if it doesn't exist."""
        log_dir = tmp_path / "new_tensorboard_logs"
        logger = TensorBoardMetricLogger(str(log_dir))

        assert log_dir.exists()

    def test_create_run_creates_subdirectory(self, tensorboard_logger, temp_tensorboard_dir):
        """Test that create_run creates a subdirectory for the run."""
        run_id = tensorboard_logger.create_run("run_1")

        assert run_id == "run_1"
        assert os.path.exists(os.path.join(temp_tensorboard_dir, "run_1"))
        assert "run_1" in tensorboard_logger.writers

    def test_log_param_as_text(self, tensorboard_logger):
        """Test that log_param writes parameters as text."""
        tensorboard_logger.create_run("run_1")

        with patch.object(tensorboard_logger.writers["run_1"], 'add_text') as mock_add_text:
            with patch.object(tensorboard_logger.writers["run_1"], 'flush') as mock_flush:
                tensorboard_logger.log_param("run_1", "learning_rate", "1e-3")

                mock_add_text.assert_called_once_with("params/learning_rate", "1e-3", global_step=0)
                mock_flush.assert_called_once()

    def test_log_metric_with_step(self, tensorboard_logger):
        """Test logging metric with explicit step."""
        tensorboard_logger.create_run("run_1")

        with patch.object(tensorboard_logger.writers["run_1"], 'add_scalar') as mock_add_scalar:
            with patch.object(tensorboard_logger.writers["run_1"], 'flush') as mock_flush:
                tensorboard_logger.log_metric("run_1", "loss", 0.5, step=100)

                mock_add_scalar.assert_called_once_with("loss", 0.5, global_step=100)
                mock_flush.assert_called_once()

    def test_log_metric_without_step(self, tensorboard_logger):
        """Test logging metric without step (defaults to 0)."""
        tensorboard_logger.create_run("run_1")

        with patch.object(tensorboard_logger.writers["run_1"], 'add_scalar') as mock_add_scalar:
            tensorboard_logger.log_metric("run_1", "accuracy", 0.95)

            mock_add_scalar.assert_called_once_with("accuracy", 0.95, global_step=0)

    def test_flush_called_after_each_log(self, tensorboard_logger):
        """Test that flush() is called after each log for real-time updates."""
        tensorboard_logger.create_run("run_1")

        with patch.object(tensorboard_logger.writers["run_1"], 'flush') as mock_flush:
            tensorboard_logger.log_metric("run_1", "loss", 0.5, step=1)
            tensorboard_logger.log_metric("run_1", "loss", 0.4, step=2)
            tensorboard_logger.log_metric("run_1", "loss", 0.3, step=3)

            # flush should be called 3 times (once per metric)
            assert mock_flush.call_count == 3

    def test_end_run_closes_writer(self, tensorboard_logger):
        """Test that end_run closes the writer and removes it from registry."""
        tensorboard_logger.create_run("run_1")

        with patch.object(tensorboard_logger.writers["run_1"], 'close') as mock_close:
            tensorboard_logger.end_run("run_1")

            mock_close.assert_called_once()
            assert "run_1" not in tensorboard_logger.writers

    def test_multiple_runs_maintain_separate_writers(self, tensorboard_logger):
        """Test that multiple runs maintain separate SummaryWriters."""
        tensorboard_logger.create_run("run_1")
        tensorboard_logger.create_run("run_2")

        assert "run_1" in tensorboard_logger.writers
        assert "run_2" in tensorboard_logger.writers
        assert tensorboard_logger.writers["run_1"] != tensorboard_logger.writers["run_2"]

    def test_get_run_metrics_returns_empty_dict(self, tensorboard_logger):
        """Test that get_run_metrics returns empty dict (not supported by TensorBoard)."""
        metrics = tensorboard_logger.get_run_metrics("run_1")

        assert metrics == {}

    def test_auto_create_run_on_first_log(self, tensorboard_logger):
        """Test that run is auto-created if it doesn't exist on first log."""
        # Log metric without creating run first
        tensorboard_logger.log_metric("run_1", "loss", 0.5, step=1)

        # Run should be auto-created
        assert "run_1" in tensorboard_logger.writers

    def test_cleanup_on_delete(self, tensorboard_logger):
        """Test that __del__ cleanup closes all writers."""
        tensorboard_logger.create_run("run_1")
        tensorboard_logger.create_run("run_2")

        with patch.object(tensorboard_logger.writers["run_1"], 'close') as mock_close1:
            with patch.object(tensorboard_logger.writers["run_2"], 'close') as mock_close2:
                tensorboard_logger.__del__()

                mock_close1.assert_called_once()
                mock_close2.assert_called_once()


class TestMetricLoggerInterface:
    """Test suite for MetricLogger abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Test that MetricLogger cannot be instantiated directly."""
        with pytest.raises(TypeError):
            MetricLogger()

    def test_all_implementations_have_required_methods(self):
        """Test that all implementations have required abstract methods."""
        required_methods = [
            'create_run',
            'log_param',
            'log_metric',
            'end_run',
            'get_run_metrics',
        ]

        for implementation in [MLflowMetricLogger, TensorBoardMetricLogger]:
            for method in required_methods:
                assert hasattr(implementation, method)
                assert callable(getattr(implementation, method))

    def test_mlflow_logger_conforms_to_interface(self):
        """Test that MLflowMetricLogger conforms to MetricLogger interface."""
        logger = MLflowMetricLogger("http://localhost:8852")
        assert isinstance(logger, MetricLogger)

    def test_tensorboard_logger_conforms_to_interface(self, temp_tensorboard_dir):
        """Test that TensorBoardMetricLogger conforms to MetricLogger interface."""
        logger = TensorBoardMetricLogger(temp_tensorboard_dir)
        assert isinstance(logger, MetricLogger)

class TestCallbacksIntegration:
    """Test suite for integration with callbacks."""

    def test_metric_logging_callback_accepts_metric_logger(self):
        """Test that MetricLoggingCallback accepts metric_logger parameter."""
        from rapidfireai.fit.ml.callbacks import MetricLoggingCallback

        mock_logger = Mock(spec=MetricLogger)
        callback = MetricLoggingCallback(
            metric_logger=mock_logger,
            metric_run_id="run_1",
            completed_steps=0,
            chunk_id=0,
            num_epochs_completed=0
        )

        assert callback.metric_logger == mock_logger

    def test_generation_metrics_callback_accepts_metric_logger(self):
        """Test that GenerationMetricsCallback accepts metric_logger parameter."""
        from rapidfireai.fit.ml.callbacks import GenerationMetricsCallback
        from transformers import AutoTokenizer
        from datasets import Dataset

        mock_logger = Mock(spec=MetricLogger)
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        dataset = Dataset.from_dict({"text": ["test"]})

        callback = GenerationMetricsCallback(
            tokenizer=tokenizer,
            eval_dataset=dataset,
            metric_logger=mock_logger,
            metric_run_id="run_1",
            completed_steps=0
        )

        assert callback.metric_logger == mock_logger

    def test_callback_calls_log_metric(self):
        """Test that callback calls metric_logger.log_metric()."""
        from rapidfireai.fit.ml.callbacks import MetricLoggingCallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        mock_logger = Mock(spec=MetricLogger)
        callback = MetricLoggingCallback(
            metric_logger=mock_logger,
            metric_run_id="run_1",
            completed_steps=0,
            chunk_id=0,
            num_epochs_completed=0
        )

        # Simulate trainer logging
        args = TrainingArguments(output_dir="./output")
        state = TrainerState(global_step=10)
        control = TrainerControl()
        logs = {"loss": 0.5, "learning_rate": 1e-4}

        callback.on_log(args, state, control, logs=logs)

        # Verify log_metric was called for each metric
        assert mock_logger.log_metric.call_count == 4  # 2 metrics + chunk_number + num_epochs_completed

    def test_callback_step_offset_works_correctly(self):
        """Test that callbacks apply completed_steps offset correctly."""
        from rapidfireai.fit.ml.callbacks import MetricLoggingCallback
        from transformers import TrainerState, TrainerControl, TrainingArguments

        mock_logger = Mock(spec=MetricLogger)
        callback = MetricLoggingCallback(
            metric_logger=mock_logger,
            metric_run_id="run_1",
            completed_steps=100,  # Resumed run
            chunk_id=0,
            num_epochs_completed=0
        )

        args = TrainingArguments(output_dir="./output")
        state = TrainerState(global_step=10)
        control = TrainerControl()
        logs = {"loss": 0.5}

        callback.on_log(args, state, control, logs=logs)

        # Verify step includes offset
        mock_logger.log_metric.assert_any_call("run_1", "loss", 0.5, step=110)  # 100 + 10
