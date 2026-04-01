"""
Conditional MLflow tracing helpers.

All tracing is gated on the ``RF_MLFLOW_ENABLED`` environment variable
(default ``"true"`` on non-Colab machines).  Set it to ``"false"`` to
disable every mlflow call — useful on CPU-only machines where tracing
memory overhead is prohibitive or the MLflow server is not running.
"""

import functools
import os
import socket
import warnings
from contextlib import contextmanager
from typing import Any

import mlflow

from rapidfireai.utils.constants import MLflowConfig


# ---------------------------------------------------------------------------
# Enabled check
# ---------------------------------------------------------------------------

def is_mlflow_enabled() -> bool:
    """Return True if MLflow tracing is enabled (RF_MLFLOW_ENABLED=true)."""
    return os.environ.get("RF_MLFLOW_ENABLED", "true").lower() == "true"


# ---------------------------------------------------------------------------
# Server reachability
# ---------------------------------------------------------------------------

def _is_mlflow_server_reachable() -> bool:
    """Return True if the MLflow server socket is open."""
    try:
        with socket.create_connection((MLflowConfig.HOST, MLflowConfig.PORT), timeout=2):
            return True
    except (OSError, ConnectionRefusedError):
        return False


# ---------------------------------------------------------------------------
# One-shot setup called from QueryProcessingActor.__init__
# ---------------------------------------------------------------------------

def setup_mlflow(experiment_name: str) -> bool:
    """
    Configure MLflow tracking for a Ray actor process.

    - If ``RF_MLFLOW_ENABLED`` is not ``"true"``, does nothing and returns False.
    - If the server is not reachable, emits a ``UserWarning``, sets
      ``RF_MLFLOW_ENABLED=false`` in the current process's environment so that
      all subsequent tracing helpers also become no-ops, and returns False.
    - On any other error during setup, warns and disables in the same way.

    Returns True when setup succeeded.
    """
    if not is_mlflow_enabled():
        return False

    if not _is_mlflow_server_reachable():
        warnings.warn(
            f"RF_MLFLOW_ENABLED=true but the MLflow server is not reachable at "
            f"{MLflowConfig.URL}. MLflow tracing will be disabled for this session. "
            f"Start the server with `rapidfireai start` or set "
            f"RF_MLFLOW_ENABLED=false to suppress this warning.",
            UserWarning,
            stacklevel=2,
        )
        os.environ["RF_MLFLOW_ENABLED"] = "false"
        return False

    try:
        mlflow.langchain.autolog()
        mlflow.openai.autolog()
        mlflow.gemini.autolog()
        mlflow.set_tracking_uri(str(MLflowConfig.URL))
        mlflow.set_experiment(experiment_name)
        return True
    except Exception as exc:
        warnings.warn(
            f"Failed to configure MLflow tracking ({exc}). "
            f"MLflow tracing will be disabled for this session.",
            UserWarning,
            stacklevel=2,
        )
        os.environ["RF_MLFLOW_ENABLED"] = "false"
        return False


# ---------------------------------------------------------------------------
# No-op span — returned when tracing is disabled
# ---------------------------------------------------------------------------

class _NoOpSpan:
    """Drop-in replacement for an MLflow span when tracing is disabled."""

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_inputs(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_outputs(self, *args: Any, **kwargs: Any) -> None:
        pass

    def set_attributes(self, *args: Any, **kwargs: Any) -> None:
        pass


_NOOP_SPAN = _NoOpSpan()


# ---------------------------------------------------------------------------
# Conditional @mlflow.trace decorator
# ---------------------------------------------------------------------------

def mlflow_trace(func=None, *, name: str | None = None, span_type=None, **kwargs):
    """
    Conditional replacement for ``@mlflow.trace``.

    When ``RF_MLFLOW_ENABLED=false`` the original function is called without
    any tracing overhead.  The check happens at *call time*, not at decoration
    time, so toggling the env-var (e.g. via ``setup_mlflow``) takes effect
    immediately without re-importing the module.

    Usage::

        @mlflow_trace(name="my_span", span_type=SpanType.CHAIN)
        def my_method(self, ...): ...

        # direct call form (mirrors mlflow.trace(func=fn, ...))
        traced = mlflow_trace(func=fn, name="...", span_type=SpanType.RETRIEVER)
    """
    def decorator(fn: Any) -> Any:
        trace_kwargs: dict[str, Any] = {}
        if name is not None:
            trace_kwargs["name"] = name
        if span_type is not None:
            trace_kwargs["span_type"] = span_type
        trace_kwargs.update(kwargs)
        _traced = mlflow.trace(fn, **trace_kwargs)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kw: Any) -> Any:
            if is_mlflow_enabled():
                return _traced(*args, **kw)
            return fn(*args, **kw)

        return wrapper

    if func is not None:
        # Called as mlflow_trace(func=fn, name=...) — direct form
        return decorator(func)
    # Called as @mlflow_trace(name=...) — decorator-factory form
    return decorator


# ---------------------------------------------------------------------------
# Conditional mlflow.start_span context manager
# ---------------------------------------------------------------------------

@contextmanager
def mlflow_start_span(name: str, span_type=None, **kwargs):
    """
    Conditional replacement for ``with mlflow.start_span(...) as span:``.

    Yields a real MLflow span when tracing is enabled, or a ``_NoOpSpan``
    that silently ignores all attribute/input/output calls when disabled.
    """
    if is_mlflow_enabled():
        with mlflow.start_span(name=name, span_type=span_type, **kwargs) as span:
            yield span
    else:
        yield _NOOP_SPAN


# ---------------------------------------------------------------------------
# Conditional mlflow.get_current_active_span
# ---------------------------------------------------------------------------

def mlflow_get_current_active_span():
    """
    Return the active MLflow span, or ``None`` when tracing is disabled.

    All call sites already guard with ``if span is not None:``, so returning
    ``None`` when disabled is a safe no-op.
    """
    if is_mlflow_enabled():
        return mlflow.get_current_active_span()
    return None
