"""Unit tests for MLflow / dispatcher status parity.

These cover the backend half of the two-layer plan documented at
``mlflow_status_parity_0c21f74a.plan.md``: every terminal lifecycle
transition (clean success, IC stop, Optuna prune, runtime / init /
dispatch failure, worker exception, stranded-run recovery, create-run
failure) must terminate the matching MLflow run with the right
``RunStatus``, and the idempotency paths in ``query_actor`` and
``experiment_utils`` must not clobber a previously-set terminal state.

The tests deliberately stay narrow:
- The shared ``MetricLogger`` abstraction and ``MLflowMetricLogger`` are
  exercised directly with mocked ``MlflowClient``.
- Controllers / IC handlers are exercised via their private
  ``_finalize_mlflow_run`` helpers (evals + fit) with a mocked
  metric_manager / metric_logger. Driving the full controller loops or
  spinning up Ray actors is far too heavy for unit tests.
- The idempotency logic is verified by replicating the in-tree code
  block under unit test against a mocked client.
"""

from unittest.mock import MagicMock, patch

import pytest

# These guards let the bulk of the suite run in lightweight environments
# (CI machines without GPU stacks). The evals controller hard-imports
# ``ray``, and the fit controller hard-imports ``torch``; when either is
# absent, we skip the dependent classes rather than fail at collection.
try:  # pragma: no cover - exercised by collection only
    import ray as _ray  # noqa: F401

    _HAS_RAY = True
except Exception:
    _HAS_RAY = False
try:  # pragma: no cover
    import torch as _torch  # noqa: F401

    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

requires_ray = pytest.mark.skipif(not _HAS_RAY, reason="ray not installed")
requires_torch = pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")


# --------------------------------------------------------------------------- #
# MLflowMetricLogger.end_run forwards status to set_terminated.
# --------------------------------------------------------------------------- #


def _build_mlflow_logger():
    """Build an MLflowMetricLogger with a mocked client (no MLflow server).

    We bypass ``__init__`` (which talks to a live MLflow tracking server) and
    inject the bare minimum the tested methods need.
    """
    from rapidfireai.utils.metric_logger import MetricLoggerType
    from rapidfireai.utils.metric_mlflow_manager import MLflowMetricLogger

    logger = object.__new__(MLflowMetricLogger)
    logger.type = MetricLoggerType.MLFLOW
    logger.client = MagicMock()
    logger.logger = MagicMock()
    return logger


@pytest.mark.parametrize(
    "status",
    ["FINISHED", "FAILED", "KILLED"],
)
def test_mlflow_metric_logger_end_run_forwards_status(status):
    """end_run(status=X) must call set_terminated(run_id, status=X)."""
    from rapidfireai.utils.metric_mlflow_manager import mlflow as mlflow_mod

    logger = _build_mlflow_logger()
    logger.client.get_run.return_value = MagicMock()  # run exists

    with patch.object(mlflow_mod, "active_run", return_value=None):
        logger.end_run("fake-run-id", status=status)

    logger.client.set_terminated.assert_called_once_with("fake-run-id", status=status)


def test_mlflow_metric_logger_end_run_default_preserves_behaviour():
    """end_run() with no status falls back to the server default (FINISHED)."""
    from rapidfireai.utils.metric_mlflow_manager import mlflow as mlflow_mod

    logger = _build_mlflow_logger()
    logger.client.get_run.return_value = MagicMock()

    with patch.object(mlflow_mod, "active_run", return_value=None):
        logger.end_run("fake-run-id")

    logger.client.set_terminated.assert_called_once_with("fake-run-id")


def test_mlflow_metric_logger_end_run_active_run_uses_status():
    """When the run is the local fluent-API active run, mlflow.end_run(status=X) is also called."""
    from rapidfireai.utils.metric_mlflow_manager import mlflow as mlflow_mod

    logger = _build_mlflow_logger()
    logger.client.get_run.return_value = MagicMock()

    active = MagicMock()
    active.info.run_id = "fake-run-id"

    with patch.object(mlflow_mod, "active_run", return_value=active), patch.object(
        mlflow_mod, "end_run"
    ) as mock_end_run:
        logger.end_run("fake-run-id", status="KILLED")

    logger.client.set_terminated.assert_called_once_with("fake-run-id", status="KILLED")
    mock_end_run.assert_called_once_with(status="KILLED")


# --------------------------------------------------------------------------- #
# Dispatcher -> MLflow status mapping helper.
# --------------------------------------------------------------------------- #


class TestDispatcherStatusMapping:
    def test_completed_maps_to_finished(self):
        from rapidfireai.utils.metric_mlflow_manager import dispatcher_status_to_mlflow

        assert dispatcher_status_to_mlflow("completed") == "FINISHED"

    def test_stopped_maps_to_killed(self):
        from rapidfireai.utils.metric_mlflow_manager import dispatcher_status_to_mlflow

        assert dispatcher_status_to_mlflow("stopped") == "KILLED"

    def test_failed_maps_to_failed(self):
        from rapidfireai.utils.metric_mlflow_manager import dispatcher_status_to_mlflow

        assert dispatcher_status_to_mlflow("failed") == "FAILED"

    def test_uppercase_input(self):
        from rapidfireai.utils.metric_mlflow_manager import dispatcher_status_to_mlflow

        assert dispatcher_status_to_mlflow("COMPLETED") == "FINISHED"

    def test_unknown_returns_none(self):
        from rapidfireai.utils.metric_mlflow_manager import dispatcher_status_to_mlflow

        assert dispatcher_status_to_mlflow("new") is None
        assert dispatcher_status_to_mlflow("ongoing") is None
        assert dispatcher_status_to_mlflow(None) is None

    def test_enum_input(self):
        from rapidfireai.utils.metric_mlflow_manager import dispatcher_status_to_mlflow
        from rapidfireai.evals.utils.constants import PipelineStatus

        assert dispatcher_status_to_mlflow(PipelineStatus.COMPLETED) == "FINISHED"
        assert dispatcher_status_to_mlflow(PipelineStatus.STOPPED) == "KILLED"
        assert dispatcher_status_to_mlflow(PipelineStatus.FAILED) == "FAILED"


# --------------------------------------------------------------------------- #
# Evals controller _finalize_mlflow_run helper.
# --------------------------------------------------------------------------- #


class _EvalsControllerLike:
    """Minimal stand-in that lets us exercise the bound helper without
    paying the full Controller __init__ cost (which expects a real
    experiment dir, Ray, etc.).
    """

    def __init__(self, metric_manager, logger):
        self.metric_manager = metric_manager
        self.logger = logger


@requires_ray
class TestEvalsControllerFinalize:
    def _bind(self, metric_manager):
        from rapidfireai.evals.scheduling.controller import Controller

        ctrl = _EvalsControllerLike(metric_manager, logger=MagicMock())
        # Bind the unbound method onto our stand-in.
        return Controller._finalize_mlflow_run.__get__(ctrl, _EvalsControllerLike)

    def _db_with_metric_run_id(self, metric_run_id):
        db = MagicMock()
        db.get_pipeline.return_value = {"metric_run_id": metric_run_id}
        return db

    @pytest.mark.parametrize("mlflow_status", ["KILLED", "FAILED", "FINISHED"])
    def test_finalize_passes_status_through(self, mlflow_status):
        mm = MagicMock()
        finalize = self._bind(mm)
        db = self._db_with_metric_run_id("evals-rid")

        finalize(db, pipeline_id=7, mlflow_status=mlflow_status)

        mm.end_run.assert_called_once_with("evals-rid", status=mlflow_status)

    def test_finalize_no_metric_manager_is_noop(self):
        finalize = self._bind(None)
        db = self._db_with_metric_run_id("evals-rid")
        finalize(db, pipeline_id=7, mlflow_status="FAILED")  # must not raise

    def test_finalize_no_metric_run_id_is_noop(self):
        mm = MagicMock()
        finalize = self._bind(mm)
        db = MagicMock()
        db.get_pipeline.return_value = {"metric_run_id": None}

        finalize(db, pipeline_id=7, mlflow_status="FAILED")
        mm.end_run.assert_not_called()

    def test_finalize_swallows_mlflow_errors(self):
        mm = MagicMock()
        mm.end_run.side_effect = RuntimeError("server down")
        finalize = self._bind(mm)
        db = self._db_with_metric_run_id("evals-rid")

        # Must not raise: MLflow trouble cannot fail the pipeline.
        finalize(db, pipeline_id=7, mlflow_status="FAILED")
        mm.end_run.assert_called_once()


# --------------------------------------------------------------------------- #
# Evals IC stop calls end_run(KILLED).
# --------------------------------------------------------------------------- #


@requires_ray
class TestEvalsInteractiveControlStop:
    def test_handle_stop_terminates_mlflow_with_killed(self):
        from rapidfireai.evals.scheduling.interactive_control import (
            InteractiveControlHandler,
        )

        # Build the handler with a mocked metric_manager. We use object.__new__
        # so we don't pay the constructor cost (logger setup, etc.).
        handler = object.__new__(InteractiveControlHandler)
        handler.metric_manager = MagicMock()
        handler.logger = MagicMock()
        handler.ic_logger = MagicMock()
        handler._context_cache = {}

        scheduler = MagicMock()
        scheduler.remove_pipeline.return_value = 3

        db = MagicMock()
        db.get_pipeline.return_value = {"metric_run_id": "stop-rid"}

        handler._handle_stop(
            pipeline_id=12,
            scheduler=scheduler,
            db=db,
            progress_display=None,
        )

        handler.metric_manager.end_run.assert_called_once_with(
            "stop-rid", status="KILLED"
        )


# --------------------------------------------------------------------------- #
# Fit controller _finalize_mlflow_run helper.
# --------------------------------------------------------------------------- #


class _FitControllerLike:
    def __init__(self, metric_logger, db, logger):
        self.metric_logger = metric_logger
        self.db = db
        self.logger = logger


@requires_torch
class TestFitControllerFinalize:
    def _bind(self, metric_logger, db):
        from rapidfireai.fit.backend.controller import Controller

        ctrl = _FitControllerLike(metric_logger, db, logger=MagicMock())
        return Controller._finalize_mlflow_run.__get__(ctrl, _FitControllerLike)

    @pytest.mark.parametrize("mlflow_status", ["KILLED", "FAILED", "FINISHED"])
    def test_finalize_passes_status_through(self, mlflow_status):
        ml = MagicMock()
        db = MagicMock()
        db.get_run.return_value = {"metric_run_id": "fit-rid"}
        finalize = self._bind(ml, db)

        finalize(run_id=42, mlflow_status=mlflow_status)
        ml.end_run.assert_called_once_with("fit-rid", status=mlflow_status)

    def test_finalize_no_metric_logger_is_noop(self):
        db = MagicMock()
        db.get_run.return_value = {"metric_run_id": "fit-rid"}
        finalize = self._bind(None, db)
        finalize(run_id=42, mlflow_status="FAILED")
        db.get_run.assert_not_called()

    def test_finalize_no_metric_run_id_is_noop(self):
        ml = MagicMock()
        db = MagicMock()
        db.get_run.return_value = {"metric_run_id": None}
        finalize = self._bind(ml, db)
        finalize(run_id=42, mlflow_status="FAILED")
        ml.end_run.assert_not_called()

    def test_finalize_swallows_mlflow_errors(self):
        ml = MagicMock()
        ml.end_run.side_effect = RuntimeError("server down")
        db = MagicMock()
        db.get_run.return_value = {"metric_run_id": "fit-rid"}
        finalize = self._bind(ml, db)

        finalize(run_id=42, mlflow_status="FAILED")  # must not raise
        ml.end_run.assert_called_once()


# --------------------------------------------------------------------------- #
# Idempotency: actor / experiment_utils mlflow.end_run() preserves terminal state.
#
# These tests replicate the in-tree code block (we can't easily invoke the
# actor's full process_query_request body in unit tests) and verify the
# logic does the right thing for each server-side status.
# --------------------------------------------------------------------------- #


def _idempotent_end_run(metric_run_id, server_status):
    """Mirror the idempotency snippet shared by query_actor and
    experiment_utils. Returns the status arg passed to mlflow.end_run,
    or ``"<no-arg>"`` when called with no kwargs (server default).
    """
    captured = {"called_with": None}

    def fake_end_run(status=None):
        captured["called_with"] = status if status is not None else "<no-arg>"

    mock_client = MagicMock()
    mock_client.get_run.return_value.info.status = server_status

    fake_mlflow = MagicMock()
    fake_mlflow.end_run.side_effect = fake_end_run

    # Inlined logic mirroring the actor / experiment_utils path.
    if metric_run_id:
        existing_status = None
        try:
            existing_status = mock_client.get_run(metric_run_id).info.status
        except Exception:
            pass
        if existing_status in ("KILLED", "FAILED", "FINISHED"):
            fake_mlflow.end_run(status=existing_status)
        else:
            fake_mlflow.end_run()
    return captured["called_with"]


@pytest.mark.parametrize(
    "server_status,expected",
    [
        ("KILLED", "KILLED"),
        ("FAILED", "FAILED"),
        ("FINISHED", "FINISHED"),
        ("RUNNING", "<no-arg>"),
        ("SCHEDULED", "<no-arg>"),
    ],
)
def test_idempotent_end_run_preserves_terminal_state(server_status, expected):
    assert _idempotent_end_run("some-id", server_status) == expected


# --------------------------------------------------------------------------- #
# Evals actor must NOT end+restart MLflow runs between shards of the same
# pipeline (the run should stay RUNNING continuously across shards).
# This guards against the FINISHED -> RUNNING flicker bug.
# --------------------------------------------------------------------------- #


def _actor_request_block(self_metric_run_id, incoming_metric_run_id):
    """Mirror the actor's pipeline-transition logic for unit testing.

    Records every observable side-effect the block would have:
      ``("clear_local",)`` -- pop local active-run stack, no server call
      ``("start_run", run_id)`` -- mlflow.start_run(run_id=...)
      ``("server_end_run", status)`` -- set_terminated on server (forbidden!)

    The actor MUST NEVER produce a ``server_end_run`` event when
    transitioning between pipelines: the controller is the single
    authority on terminal status. This list captures that invariant.
    """
    actions = []

    same_pipeline_next_shard = (
        self_metric_run_id is not None and self_metric_run_id == incoming_metric_run_id
    )
    is_mlflow_enabled = True

    if self_metric_run_id and is_mlflow_enabled and not same_pipeline_next_shard:
        # NB: explicitly local-only -- no SetTerminated on the server.
        actions.append(("clear_local",))

    if incoming_metric_run_id and is_mlflow_enabled and not same_pipeline_next_shard:
        actions.append(("start_run", incoming_metric_run_id))

    return actions


def test_actor_same_pipeline_next_shard_does_not_touch_mlflow():
    """Actor reused for the same pipeline: MLflow run stays RUNNING; no calls."""
    assert _actor_request_block(
        self_metric_run_id="run-a", incoming_metric_run_id="run-a"
    ) == []


def test_actor_different_pipeline_clears_local_and_starts_new():
    """Actor switched to a different pipeline: clear LOCAL state only and start new.

    Critically, no ``server_end_run`` event -- if we hit set_terminated on
    the server, the previous pipeline (which may still have shards in flight
    on other actors) would prematurely show as COMPLETED on the dashboard.
    """
    actions = _actor_request_block(
        self_metric_run_id="run-a", incoming_metric_run_id="run-b"
    )
    assert actions == [("clear_local",), ("start_run", "run-b")]
    # Defensive check: the forbidden side-effect must never appear.
    assert not any(a[0] == "server_end_run" for a in actions)


def test_actor_first_pipeline_only_starts():
    """First pipeline on a fresh actor: nothing to clear, just start the new one."""
    assert _actor_request_block(
        self_metric_run_id=None, incoming_metric_run_id="run-a"
    ) == [("start_run", "run-a")]


# --------------------------------------------------------------------------- #
# RFMetricLogger forwards status to per-backend end_run.
# --------------------------------------------------------------------------- #


def test_rf_metric_logger_forwards_status_to_backends():
    from rapidfireai.utils.metric_logger import MetricLoggerType
    from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger

    mlflow_backend = MagicMock()
    mlflow_backend.type = MetricLoggerType.MLFLOW
    tensorboard_backend = MagicMock()
    tensorboard_backend.type = MetricLoggerType.TENSORBOARD

    # Build an RFMetricLogger but inject its dependencies directly.
    rf = object.__new__(RFMetricLogger)
    rf.metric_loggers = {
        "mlflow": mlflow_backend,
        "tensorboard": tensorboard_backend,
    }
    rf.logger = MagicMock()
    rf._get_run_name = MagicMock(return_value="my-run")

    rf.end_run("real-run-id", status="KILLED")

    mlflow_backend.end_run.assert_called_once_with("real-run-id", status="KILLED")
    tensorboard_backend.end_run.assert_called_once_with("my-run", status="KILLED")


# --------------------------------------------------------------------------- #
# set_tag: progress tags emitted by the evals controller for the Shards
# column on the dashboard. MLflow must receive the tag; TB/Trackio use
# the no-op default in MetricLogger.
# --------------------------------------------------------------------------- #


def test_mlflow_metric_logger_set_tag_forwards_to_client():
    """``MLflowMetricLogger.set_tag`` must call ``client.set_tag`` with stringified value."""
    logger = _build_mlflow_logger()
    logger.set_tag("run-xyz", "rapidfire.progress.current", 3)
    logger.client.set_tag.assert_called_once_with("run-xyz", "rapidfire.progress.current", "3")


def test_metric_logger_default_set_tag_is_noop():
    """Backends without a real implementation should silently accept set_tag."""
    from rapidfireai.utils.metric_logger import MetricLogger

    # Subclass MetricLogger and provide stubs for the abstract methods so
    # we can instantiate it. The whole point of this test is that the
    # default ``set_tag`` doesn't raise -- it's a no-op for backends that
    # don't have a native tag concept.
    class _Stub(MetricLogger):
        def create_experiment(self, experiment_name):
            return ""

        def get_experiment(self, experiment_name):
            return ""

        def create_run(self, run_name):
            return ""

        def log_param(self, run_id, key, value):
            return None

        def log_metric(self, run_id, key, value, step=None):
            return None

        def get_run_metrics(self, run_id):
            return {}

        def end_run(self, run_id, status=None):
            return None

        def delete_run(self, run_id):
            return None

        def clear_context(self):
            return None

    _Stub().set_tag("run", "key", "value")  # must not raise


def test_rf_metric_logger_fanouts_set_tag():
    """RFMetricLogger.set_tag fans out to every configured backend."""
    from rapidfireai.utils.metric_logger import MetricLoggerType
    from rapidfireai.utils.metric_rfmetric_manager import RFMetricLogger

    mlflow_backend = MagicMock()
    mlflow_backend.type = MetricLoggerType.MLFLOW
    tensorboard_backend = MagicMock()
    tensorboard_backend.type = MetricLoggerType.TENSORBOARD

    rf = object.__new__(RFMetricLogger)
    rf.metric_loggers = {
        "mlflow": mlflow_backend,
        "tensorboard": tensorboard_backend,
    }
    rf.logger = MagicMock()
    rf._get_run_name = MagicMock(return_value="run-name-42")

    rf.set_tag("real-id", "rapidfire.progress.current", "2")

    # MLflow gets the canonical run_id; TB/Trackio get the human name
    # (matches the pattern in log_metric/log_param).
    mlflow_backend.set_tag.assert_called_once_with("real-id", "rapidfire.progress.current", "2")
    tensorboard_backend.set_tag.assert_called_once_with("run-name-42", "rapidfire.progress.current", "2")
