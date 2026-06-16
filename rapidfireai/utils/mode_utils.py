"""
Utilities for reading and validating the installed RapidFire mode.

The installed mode ("fit" or "evals") is chosen at install time by
``rapidfireai init`` / ``rapidfireai init --evals`` and persisted to
``$RF_HOME/rf_mode.txt`` (see ``cli.py``). It permanently determines which
dispatcher is started and which database it reads (see ``setup/start.sh``).

Because the dispatcher and IC Ops are bound to this mode, running an experiment
in a different mode than the one installed produces a working-but-uncontrollable
experiment (the IC Ops panel checks the wrong database and stays disabled). These
helpers let ``run_fit()`` / ``run_evals()`` detect that mismatch and block early
with an actionable error.

When ``rf_mode.txt`` is missing, ``setup/start.sh`` falls back to starting the
``"fit"`` dispatcher (``RAPIDFIRE_MODE=$(cat ... || echo "fit")``), so the
effective installed mode in that case is ``"fit"``. The helpers below mirror that
fallback so ``run_fit()`` is not blocked when services are already running in the
default fit mode.
"""

from pathlib import Path

from rapidfireai.utils.constants import RF_HOME

# Mode that ``setup/start.sh`` starts when ``rf_mode.txt`` is absent.
DEFAULT_MODE = "fit"


def get_installed_mode() -> str | None:
    """
    Read the installed RapidFire mode from ``$RF_HOME/rf_mode.txt``.

    Returns:
        The stripped file contents (e.g. ``"fit"`` or ``"evals"``), or ``None``
        if the file is missing or cannot be read.
    """
    mode_file = Path(RF_HOME) / "rf_mode.txt"
    try:
        if mode_file.exists():
            return mode_file.read_text().strip() or None
    except OSError:
        return None
    return None


def assert_mode_matches(required: str, installed: str | None) -> None:
    """
    Ensure the installed mode matches the mode required by the operation.

    A missing mode (``installed is None``) is treated as ``DEFAULT_MODE``,
    matching ``setup/start.sh``, which starts the fit dispatcher when
    ``rf_mode.txt`` is absent. This keeps ``run_fit()`` working when services are
    already running in the default fit mode but no mode file was written.

    Args:
        required: The mode the operation needs ("fit" or "evals").
        installed: The installed mode as returned by ``get_installed_mode()``.

    Raises:
        ValueError: If the effective installed mode does not equal ``required``.
            The message is actionable and tells the user how to fix the mismatch.
    """
    effective = installed if installed is not None else DEFAULT_MODE
    if effective == required:
        return

    init_cmd = "rapidfireai init" if required == "fit" else "rapidfireai init --evals"

    raise ValueError(
        f"RapidFire is installed in '{effective}' mode, but run_{required}() requires "
        f"'{required}' mode. The dispatcher and IC Ops are bound to '{effective}' mode, so "
        f"'{required}' experiments will run but won't be controllable from the dashboard. "
        f"Re-initialize with `{init_cmd}`, then restart services "
        f"(`rapidfireai stop && rapidfireai start`)."
    )
