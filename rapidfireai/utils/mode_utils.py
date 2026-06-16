"""
Utilities for reading and validating the installed RapidFire mode.

The installed mode ("fit" or "evals") is chosen at install time by
``rapidfireai init`` (evals, the default) / ``rapidfireai init --train`` (fit) and
persisted to ``$RF_HOME/rf_mode.txt`` (see ``cli.py``). It permanently determines
which dispatcher is started and which database it reads (see ``setup/start.sh``).

Because the dispatcher and IC Ops are bound to this mode, running an experiment
in a different mode than the one installed produces a working-but-uncontrollable
experiment (the IC Ops panel checks the wrong database and stays disabled). These
helpers let ``Experiment.__init__`` detect that mismatch and block early with an
actionable error.

When ``rf_mode.txt`` is missing or unreadable, ``setup/start.sh`` falls back to
starting the ``"evals"`` dispatcher (``RAPIDFIRE_MODE=$(cat ... || echo "evals")``),
matching the install default, so the effective installed mode in that case is
``"evals"``. The helpers below mirror that fallback so a default evals experiment is
not blocked when services are already running in the default evals mode.

That fallback hinges on ``cat`` *failing*. A file that exists but is empty or
whitespace-only is different: ``cat`` succeeds and yields an empty
``RAPIDFIRE_MODE``, so ``start.sh`` builds a broken ``$RAPIDFIRE_DIR//dispatcher``
path rather than the evals dispatcher. The helpers therefore keep a present-but-blank
file distinct from a missing one (``""`` vs ``None``) and refuse to treat it as the
default, which would otherwise let the guard pass while services run a broken
path — the exact mismatch this guard exists to prevent.
"""

from pathlib import Path

from rapidfireai.utils.constants import RF_HOME

# Mode that ``setup/start.sh`` starts when ``rf_mode.txt`` is absent (matches the
# ``rapidfireai init`` install default). Keep in sync with ``setup/start.sh``.
DEFAULT_MODE = "evals"


def get_installed_mode() -> str | None:
    """
    Read the installed RapidFire mode from ``$RF_HOME/rf_mode.txt``.

    Returns:
        - The stripped file contents (e.g. ``"fit"`` or ``"evals"``) when the file
          exists and is non-empty.
        - ``""`` when the file exists but is empty, whitespace-only, or not valid
          UTF-8. These are kept distinct from ``None`` on purpose: ``setup/start.sh``
          only defaults to the evals dispatcher when ``cat`` *fails* (missing file);
          an existing blank/corrupt file still lets ``cat`` succeed, yielding an empty
          or garbage ``RAPIDFIRE_MODE`` and a broken dispatcher path, so it must not be
          treated as the default.
        - ``None`` when the file is missing or unreadable (OS-level error), mirroring
          the cases where ``cat`` fails and ``start.sh`` falls back to evals.
    """
    mode_file = Path(RF_HOME) / "rf_mode.txt"
    try:
        if mode_file.exists():
            # utf-8-sig consumes a leading BOM (e.g. from a Windows editor) so a value
            # like "﻿evals" still matches "evals" instead of failing the guard.
            return mode_file.read_text(encoding="utf-8-sig").strip()
    except UnicodeDecodeError:
        # Present but not valid UTF-8 (corrupt mode file). Treat like a blank file:
        # cat would still succeed in start.sh and build a broken dispatcher path, so
        # return "" (present-but-invalid) to block with a remedy rather than letting a
        # raw UnicodeDecodeError escape and bypass the mode-mismatch guidance.
        return ""
    except OSError:
        return None
    return None


def assert_mode_matches(required: str, installed: str | None) -> None:
    """
    Ensure the installed mode matches the mode required by the operation.

    A missing mode (``installed is None``) is treated as ``DEFAULT_MODE``,
    matching ``setup/start.sh``, which starts the evals dispatcher when
    ``rf_mode.txt`` is absent or unreadable. This keeps a default evals experiment
    working when services are already running in the default evals mode but no mode
    file was written. A present-but-blank file (``installed == ""``) is *not* treated
    as the default — ``start.sh`` would build a broken dispatcher path for it — so it
    always fails the guard with a remedy.

    Args:
        required: The mode the operation needs ("fit" or "evals").
        installed: The installed mode as returned by ``get_installed_mode()``.

    Raises:
        ValueError: If the effective installed mode does not equal ``required``.
            The message is actionable and tells the user how to fix the mismatch.
    """
    init_cmd = "rapidfireai init --train" if required == "fit" else "rapidfireai init"

    # Blank/whitespace-only or corrupt (non-UTF-8) rf_mode.txt: not a missing file, so
    # it cannot inherit the default. Tell the user the mode file is unreadable and how
    # to recreate it.
    if installed == "":
        raise ValueError(
            f"The installed RapidFire mode file (rf_mode.txt) is empty or unreadable, "
            f"but this experiment is configured for '{required}' mode. Re-initialize "
            f"with `{init_cmd}`, then restart services "
            f"(`rapidfireai stop && rapidfireai start`)."
        )

    effective = installed if installed is not None else DEFAULT_MODE
    if effective == required:
        return

    raise ValueError(
        f"RapidFire is installed in '{effective}' mode, but this experiment is configured "
        f"for '{required}' mode. The dispatcher and IC Ops are bound to '{effective}' mode, "
        f"so '{required}' experiments will run but won't be controllable from the dashboard. "
        f"Re-initialize with `{init_cmd}`, then restart services "
        f"(`rapidfireai stop && rapidfireai start`)."
    )
