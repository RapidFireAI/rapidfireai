"""This module contains the MLflowManager class which is responsible for managing the MLflow runs."""

import os
import re
import mlflow
from mlflow.tracking import MlflowClient
from typing import Any, Optional
from rapidfireai.utils.metric_logger import MetricLogger, MetricLoggerType
from rapidfireai.utils.ping import ping_server
from rapidfireai.utils.constants import MLflowConfig
from rapidfireai.evals.utils.logger import RFLogger


# Dispatcher pipeline/run status -> MLflow RunStatus mapping.
#
# MLflow's RunInfo.status is a fixed enum (RUNNING, SCHEDULED, FINISHED,
# FAILED, KILLED). We cannot store the dispatcher's vocabulary (COMPLETED,
# STOPPED, ...) verbatim because the MLflow server validates submissions
# against its own enum. We pick the closest MLflow value here and the
# forked frontend (RunViewStatusBox.tsx) relabels them back to the
# dispatcher's words so the dashboard text matches the notebook table.
#
# Both fit-mode (rapidfireai/fit/utils/constants.py::RunStatus) and
# evals-mode (rapidfireai/evals/utils/constants.py::PipelineStatus) use
# the same lowercase string values, so a single mapping covers both.
#
#   COMPLETED -> FINISHED  (clean success)
#   STOPPED   -> KILLED    (covers user IC-stop + Optuna prune)
#   FAILED    -> FAILED    (runtime / init / dispatch errors)
#   DELETED   -> N/A       (handled by delete_run, never via end_run)
DISPATCHER_TO_MLFLOW_STATUS: dict[str, str] = {
    "completed": "FINISHED",
    "stopped": "KILLED",
    "failed": "FAILED",
}


def dispatcher_status_to_mlflow(dispatcher_status: Any) -> Optional[str]:
    """Convert a dispatcher status (enum or string) to its MLflow RunStatus.

    Accepts either a ``PipelineStatus``/``RunStatus`` enum value or a raw
    string. Returns ``None`` when the status has no MLflow counterpart
    (e.g. ``DELETED``, ``NEW``, ``ONGOING``) so callers can decide to skip
    the MLflow termination.
    """
    if dispatcher_status is None:
        return None
    if hasattr(dispatcher_status, "value"):
        key = dispatcher_status.value
    else:
        key = dispatcher_status
    if not isinstance(key, str):
        return None
    return DISPATCHER_TO_MLFLOW_STATUS.get(key.lower())


# MLflow only allows alphanumerics, underscores (_), dashes (-), periods (.),
# colons (:), slashes (/), and spaces in metric and param names. Common
# RAG/IR-style metric names such as ``NDCG@5`` or ``Recall@10`` therefore fail
# MLflow validation and never reach the dashboard. We replace ``@`` with
# ``_at_`` to preserve the human-readable intent (``NDCG@5`` -> ``NDCG_at_5``)
# and fall back to ``_`` for any other disallowed character.
_MLFLOW_NAME_INVALID_RE = re.compile(r"[^\w./\- :]")


def _sanitize_mlflow_name(name: str) -> str:
    """Convert a metric/param name into one MLflow accepts.

    ``@`` is mapped to ``_at_`` so the meaning of names like ``NDCG@5`` is
    preserved (``NDCG_at_5``); any other character outside MLflow's allowed
    set is replaced with ``_``.
    """
    if not isinstance(name, str):
        return name
    # Map ``@`` first to preserve semantic meaning, then catch any other
    # disallowed characters with a generic underscore replacement.
    sanitized = name.replace("@", "_at_")
    return _MLFLOW_NAME_INVALID_RE.sub("_", sanitized)


def clear_local_mlflow_active_run_stack(logger: Any = None) -> None:
    """Pop the current process's MLflow fluent active-run stack WITHOUT
    posting a server-side ``SetTerminated``.

    ``mlflow.end_run()`` (or ``mlflow.end_run(status=None)``) defaults to
    ``FINISHED`` and fires ``SetTerminated(FINISHED)`` on the tracking
    server. That's the right thing to do for a run we know is healthy and
    we are intentionally finishing -- but it is the *wrong* thing to do
    when we can't determine the server-side status (e.g. the
    ``MlflowClient.get_run`` lookup raised). In that case the run may
    already be in a terminal state (``KILLED`` from IC stop, ``FAILED``
    from a worker / controller error) that the fluent default would
    silently clobber, leaving the dashboard with a misleading
    ``COMPLETED`` cell.

    The work here mirrors ``mlflow.end_run`` minus the ``set_terminated``
    call: pop the fluent stack so the next ``mlflow.start_run(...)`` in
    this process succeeds, and tear down the per-run system-metrics
    monitor MLflow may have started (otherwise it leaks).

    Best-effort: MLflow's private fluent symbols (``_active_run_stack``,
    ``_last_active_run_id``, ``run_id_to_system_metrics_monitor``) can
    shift between MLflow versions. If any of them are missing or change
    shape, we log and fall back to no-op rather than raise -- the next
    ``mlflow.start_run()`` will surface a clearer error than we could
    synthesise here, and we would rather expose that than risk marking
    the previous run as ``FINISHED``.

    ``logger`` is optional; when ``None`` we silently swallow internal
    errors (callers that want diagnostics should pass one in).
    """
    try:
        from mlflow.tracking import fluent
        from mlflow.environment_variables import MLFLOW_RUN_ID

        active_stack = fluent._active_run_stack.get()
        if active_stack:
            MLFLOW_RUN_ID.unset()
            run = active_stack.pop()
            fluent._last_active_run_id.set(run.info.run_id)
            monitor = fluent.run_id_to_system_metrics_monitor.pop(
                run.info.run_id, None
            )
            if monitor is not None:
                try:
                    monitor.finish()
                except Exception:
                    pass
    except Exception as e:  # pragma: no cover - defensive against MLflow internals churn
        if logger is not None:
            try:
                logger.warning(
                    f"Could not clear local MLflow active-run stack: {e}"
                )
            except Exception:
                pass


class MLflowMetricLogger(MetricLogger):
    def __init__(self, tracking_uri: str, logger: RFLogger = None, init_kwargs: dict[str, Any] = None):
        """
        Initialize MLflow Manager with tracking URI.

        Args:
            tracking_uri: MLflow tracking server URI
            init_kwargs: Initialization kwargs for MLflow
        """
        self.type = MetricLoggerType.MLFLOW
        self.client = None
        self.logger = logger if logger is not None else RFLogger()
        self.init_kwargs = init_kwargs # Not currently used
        if not ping_server(MLflowConfig.HOST, MLflowConfig.PORT, 2):
            raise ConnectionRefusedError(f"MLflow server not available at {MLflowConfig.URL}. MLflow logging will be disabled")
        else:
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient(tracking_uri=tracking_uri)
        self.experiment_id = None

    def create_experiment(self, experiment_name: str) -> str:
        """Create a new experiment and set it as active."""
        self.experiment_id = self.client.create_experiment(experiment_name)
        # IMPORTANT: Set this as the active experiment in MLflow context
        mlflow.set_experiment(experiment_name)
        return self.experiment_id

    def get_experiment(self, experiment_name: str) -> str:
        """Get existing experiment by name and set it as active."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        self.experiment_id = experiment.experiment_id
        return self.experiment_id

    def create_run(self, run_name: str) -> str:
        """Create a new run and return metric_run_id."""
        if self.experiment_id is None:
            raise ValueError("No experiment set. Call create_experiment() or get_experiment() first.")
        run = self.client.create_run(self.experiment_id, run_name=run_name)
        return run.info.run_id

    def log_param(self, run_id: str, key: str, value: str) -> None:
        """Log parameters to a specific run."""
        safe_key = _sanitize_mlflow_name(key)
        self.client.log_param(run_id, safe_key, value)

    def log_metric(self, run_id: str, key: str, value: float, step: int = None) -> None:
        """Log a metric to a specific run."""
        safe_key = _sanitize_mlflow_name(key)
        self.client.log_metric(run_id, safe_key, value, step=step)

    def get_run_metrics(self, run_id: str) -> dict[str, list[tuple[int, float]]]:
        """
        Get all metrics for a specific run.
        """
        try:
            run = self.client.get_run(run_id)
            if run is None:
                return {}

            run_data = run.data
            metric_dict = {}
            for metric_key in run_data.metrics.keys():
                try:
                    metric_history = self.client.get_metric_history(run_id, metric_key)
                    metric_dict[metric_key] = [(metric.step, metric.value) for metric in metric_history]
                except Exception as e:
                    self.logger.error(f"Error getting metric history for {metric_key}: {e}")
                    continue
            return metric_dict
        except Exception as e:
            self.logger.error(f"Error getting metrics for run {run_id}: {e}")
            return {}

    def end_run(self, run_id: str, status: Optional[str] = None) -> None:
        """End a specific run.

        Args:
            run_id: MLflow run identifier.
            status: Optional MLflow ``RunStatus`` string (``"FINISHED"``,
                ``"FAILED"``, ``"KILLED"``). When ``None``, MLflow's server
                defaults to ``FINISHED``. Use :func:`dispatcher_status_to_mlflow`
                to derive this from a dispatcher pipeline / run status.

        See the ``DISPATCHER_TO_MLFLOW_STATUS`` map at the top of this module
        for the dispatcher -> MLflow status mapping used by callers.
        """
        # Check if run exists before terminating
        run = self.client.get_run(run_id)
        if run is not None:
            # First terminate the run on the server. set_terminated treats
            # status=None as "use server default (FINISHED)", which preserves
            # existing behaviour for callers that haven't been updated.
            if status is not None:
                self.client.set_terminated(run_id, status=status)
            else:
                self.client.set_terminated(run_id)

            # Then clear the local MLflow context if this is the active run.
            # We pass the same status to mlflow.end_run() so the fluent
            # API doesn't POST another SetTerminated(FINISHED) that would
            # clobber our explicit terminal state.
            try:
                current_run = mlflow.active_run()
                # Make sure we end the run on the correct worker
                if current_run and current_run.info.run_id == run_id:
                    if status is not None:
                        mlflow.end_run(status=status)
                    else:
                        mlflow.end_run()
                else:
                    self.logger.warning(f"Run {run_id} is not the active run, no local context to clear")
            except Exception as e:
                self.logger.error(f"Error clearing local MLflow context: {e}")
        else:
            self.logger.warning(f"MLflow run {run_id} not found, cannot terminate")

    def restart_run(self, run_id: str) -> None:
        """Flip an MLflow run that was previously terminated back to ``RUNNING``.

        Counterpart of :meth:`end_run`. The IC stop handler terminates
        the run with ``KILLED`` so the dashboard mirrors the notebook
        table; the IC resume handler calls this to put the run back to
        ``RUNNING`` so the dashboard stops showing ``STOPPED`` for a
        pipeline / fit-run that is actually progressing again.

        We deliberately use ``MlflowClient.update_run`` rather than
        ``set_terminated`` here: ``set_terminated`` is semantically about
        terminal states (its docstring says "Set a run's status to
        terminated" even though MLflow happens to also accept ``RUNNING``
        / ``SCHEDULED``). ``update_run`` is the explicit, future-proof
        way to flip status in either direction.

        Best-effort: any backend error is logged and swallowed because a
        stale dashboard state must not fail the IC operation.
        """
        try:
            run = self.client.get_run(run_id)
        except Exception as e:
            self.logger.warning(
                f"Cannot restart MLflow run {run_id}: get_run failed ({e})"
            )
            return
        if run is None:
            self.logger.warning(
                f"MLflow run {run_id} not found, cannot restart"
            )
            return
        try:
            self.client.update_run(run_id, status="RUNNING")
            self.logger.info(
                f"Restarted MLflow run {run_id} (status -> RUNNING)"
            )
        except Exception as e:
            self.logger.warning(
                f"Failed to restart MLflow run {run_id}: {e}"
            )

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        """Set a (mutable) tag on the MLflow run.

        Used for evolving per-run state (e.g. ``rapidfire.progress.current``)
        that MLflow ``params`` cannot represent because params are
        immutable.
        """
        try:
            self.client.set_tag(run_id, key, str(value))
        except Exception as e:
            self.logger.warning(f"Failed to set MLflow tag {key}={value} on run {run_id}: {e}")

    def delete_run(self, run_id: str) -> None:
        """Delete a specific run."""
        # Check if run exists before deleting
        run = self.client.get_run(run_id)
        if run is not None:
            self.client.delete_run(run_id)
        else:
            raise ValueError(f"Run '{run_id}' not found")

    def clear_context(self) -> None:
        """Clear the MLflow context by ending any active run.

        Idempotent w.r.t. server-side terminal status: if the active run is
        already KILLED / FAILED / FINISHED on the server, we pass that same
        status to ``mlflow.end_run()`` so the fluent API does not POST
        another ``SetTerminated(FINISHED)`` that clobbers the explicit
        terminal state set by the controller / worker / IC handler.

        When the server-side status lookup itself fails (network blip,
        transient MLflow outage, etc.) we deliberately *do not* call
        ``mlflow.end_run()`` -- that would default to ``FINISHED`` and
        clobber any prior ``KILLED`` / ``FAILED`` state we couldn't read.
        Instead, we clear only the local fluent active-run stack via
        :func:`clear_local_mlflow_active_run_stack`, leaving the server's
        view of the run untouched. The next ``mlflow.start_run(...)`` in
        this process will still succeed.
        """
        try:
            current_run = mlflow.active_run()
            if current_run:
                run_id = current_run.info.run_id

                existing_status = None
                get_run_failed = False
                try:
                    existing_status = self.client.get_run(run_id).info.status
                except Exception as get_err:
                    get_run_failed = True
                    self.logger.warning(
                        f"Could not fetch server-side status for MLflow run "
                        f"{run_id} while clearing context: {get_err}. "
                        f"Falling back to local-only stack clear to preserve "
                        f"any prior terminal status."
                    )

                if existing_status in ("KILLED", "FAILED", "FINISHED"):
                    mlflow.end_run(status=existing_status)
                elif get_run_failed:
                    clear_local_mlflow_active_run_stack(self.logger)
                else:
                    # Server says the run is genuinely RUNNING / SCHEDULED;
                    # default end_run() (-> FINISHED) is the correct
                    # terminal state for an explicitly-cleared context.
                    mlflow.end_run()

                self.logger.info(
                    f"MLflow context cleared for run {run_id} "
                    f"(status={existing_status or ('UNKNOWN' if get_run_failed else 'FINISHED')})"
                )
            else:
                self.logger.info("No active MLflow run to clear")
        except Exception as e:
            self.logger.error(f"Error clearing MLflow context: {e}")
