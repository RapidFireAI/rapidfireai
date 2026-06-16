"""Quick non-destructive check of the run_fit/run_evals mode guard.

Run:  python test_mode_guard.py
Uses a temp RF_HOME so it never touches your real install.
"""

import os
import tempfile

# Point RF_HOME at a temp dir BEFORE importing rapidfireai (RF_HOME is read at import time).
tmp = tempfile.mkdtemp()
os.environ["RF_HOME"] = tmp
mode_file = os.path.join(tmp, "rf_mode.txt")

from rapidfireai.utils.mode_utils import assert_mode_matches, get_installed_mode


def check(label, installed, required, should_pass):
    if installed is None:
        if os.path.exists(mode_file):
            os.remove(mode_file)
    else:
        with open(mode_file, "w") as f:
            f.write(installed)
    try:
        assert_mode_matches(required, get_installed_mode())
        result = "PASSED (no error)"
        ok = should_pass
    except ValueError as e:
        result = f"BLOCKED -> {e}"
        ok = not should_pass
    print(f"[{'OK ' if ok else 'FAIL'}] {label}: {result}\n")


print(f"Using temp RF_HOME: {tmp}\n")
check("installed=fit,   run_fit   (should allow)", "fit", "fit", should_pass=True)
check("installed=evals, run_fit   (should block)", "evals", "fit", should_pass=False)
check("installed=evals, run_evals (should allow)", "evals", "evals", should_pass=True)
check("installed=fit,   run_evals (should block)", "fit", "evals", should_pass=False)
check("missing file,    run_fit   (should allow)", None, "fit", should_pass=True)
check("missing file,    run_evals (should block)", None, "evals", should_pass=False)
