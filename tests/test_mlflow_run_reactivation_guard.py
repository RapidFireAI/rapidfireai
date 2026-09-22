"""Terminal-status guard for ``mlflow.start_run(run_id=<existing-run>)``.

``mlflow.start_run(run_id=<terminal-run>)`` reactivates the run, flipping
server status ``KILLED``/``FAILED``/``FINISHED`` back to ``RUNNING``. The query
actor calls it in ``initialize_for_pipeline`` when switching to a (possibly
already-stopped) pipeline; without a guard the dashboard (reads MLflow) flips
``STOPPED -> ONGOING`` while the dispatcher DB (and the notebook) correctly
stay ``STOPPED``.

These tests pin the pure helpers in ``mlflow_utils.py`` that the actor now
gates that call on. They load the module directly from its file path because
the suite-wide conftest mocks ``rapidfireai.evals.utils`` as a MagicMock, so a
normal package import would not reach the real module (same pattern as
``tests/test_nonblocking_actor_init.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_mlflow_utils():
    """Load ``mlflow_utils`` directly from its file path.

    See ``tests/test_nonblocking_actor_init.py`` for the rationale: importing
    via the package path hits the suite-wide ``rapidfireai.evals.utils``
    MagicMock installed by conftest. Loading from the file path with a
    synthetic name bypasses that and exercises the real module. The module's
    top-level ``from rapidfireai.utils.constants import MLflowConfig`` resolves
    to the conftest's MagicMock stub, which is harmless here because the helpers
    under test never touch ``MLflowConfig``.
    """
    src = (
        Path(__file__).resolve().parents[1]
        / "rapidfireai"
        / "evals"
        / "utils"
        / "mlflow_utils.py"
    )
    spec = importlib.util.spec_from_file_location("_mlflow_utils_reactivation_under_test", src)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {src}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    _mlflow_utils = _load_mlflow_utils()
except ImportError as exc:  # pragma: no cover - OSS deps absent
    pytest.skip(f"mlflow_utils unavailable: {exc}", allow_module_level=True)

get_mlflow_run_status = _mlflow_utils.get_mlflow_run_status
is_mlflow_run_terminal = _mlflow_utils.is_mlflow_run_terminal
_MLFLOW_TERMINAL_STATUSES = _mlflow_utils._MLFLOW_TERMINAL_STATUSES


class _FakeRunInfo:
    def __init__(self, status):
        self.status = status


class _FakeRun:
    def __init__(self, status):
        self.info = _FakeRunInfo(status)


class _FakeClient:
    """Minimal stand-in for ``MlflowClient``: returns a run with a fixed
    status, or raises to simulate a transport / lookup failure."""

    def __init__(self, status=None, *, raises=False):
        self._status = status
        self._raises = raises

    def get_run(self, run_id):
        if self._raises:
            raise RuntimeError("simulated MLflow transport error")
        return _FakeRun(self._status) if self._status is not None else None


def test_terminal_statuses_constant_covers_killed_failed_finished():
    assert frozenset({"KILLED", "FAILED", "FINISHED"}) == _MLFLOW_TERMINAL_STATUSES


@pytest.mark.parametrize("status", ["RUNNING", "SCHEDULED"])
def test_non_terminal_statuses_do_not_block_reactivation(status):
    """A still-running (or scheduled) run must be (re)openable -- the guard
    returns False so the actor proceeds with ``start_run`` as before."""
    client = _FakeClient(status=status)
    assert get_mlflow_run_status(client, "run-1") == status
    assert is_mlflow_run_terminal(client, "run-1") is False


@pytest.mark.parametrize("status", ["KILLED", "FAILED", "FINISHED"])
def test_terminal_statuses_block_reactivation(status):
    """The fix: a run the controller already terminated must NOT be
    reactivated by the actor. The guard returns True so the actor skips
    ``start_run`` and leaves the server status untouched."""
    client = _FakeClient(status=status)
    assert get_mlflow_run_status(client, "run-1") == status
    assert is_mlflow_run_terminal(client, "run-1") is True


def test_absent_run_is_not_terminal():
    """``get_run`` returning None (run deleted / unknown) -> status None,
    guard False. Fails open so a misconfigured run id doesn't wedge a
    legitimate pipeline."""
    client = _FakeClient(status=None)
    assert get_mlflow_run_status(client, "missing") is None
    assert is_mlflow_run_terminal(client, "missing") is False


def test_lookup_error_fails_open():
    """A transient MLflow lookup error must NOT make the guard block a
    running pipeline. ``get_mlflow_run_status`` swallows the exception and
    returns None, so ``is_mlflow_run_terminal`` is False (fail open). The
    controller's stopping-pipelines guard still disposes of the shard in the
    genuinely-stopped case, so failing open here is safe."""
    client = _FakeClient(raises=True)
    assert get_mlflow_run_status(client, "run-1") is None
    assert is_mlflow_run_terminal(client, "run-1") is False
