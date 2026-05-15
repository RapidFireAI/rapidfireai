"""Dataset utilities for agent evaluations.

Provides helpers for common agent benchmark patterns:

- ``expand_for_pass_k``: replicate each task *k* times for pass^k evaluation
- ``run_code_in_sandbox``: execute generated code against test cases (HumanEval+)
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# pass^k expansion
# ---------------------------------------------------------------------------


def expand_for_pass_k(dataset, k: int = 8, task_id_column: str = "task_id"):
    """Replicate each task *k* times for pass^k evaluation.

    Each replica gets a ``trial_id`` column (0 … k−1) and the original
    ``task_id`` for grouping during accumulation.

    Works with both HuggingFace ``Dataset`` objects and plain dicts-of-lists.

    Args:
        dataset: HF Dataset or ``dict[str, list]``.
        k: Number of independent trials per task.
        task_id_column: Column name for task identity.

    Returns:
        Expanded dataset in the same format as the input.
    """
    if hasattr(dataset, "to_dict"):
        data = dataset.to_dict()
    elif isinstance(dataset, dict):
        data = dataset
    else:
        raise TypeError(f"Expected HuggingFace Dataset or dict, got {type(dataset).__name__}")

    if task_id_column not in data:
        raise KeyError(f"Column '{task_id_column}' not found in dataset. Available: {list(data.keys())}")

    expanded: dict[str, list] = {col: [] for col in data}
    expanded["trial_id"] = []

    num_rows = len(data[task_id_column])
    for i in range(num_rows):
        for trial in range(k):
            for col in data:
                expanded[col].append(data[col][i])
            expanded["trial_id"].append(trial)

    # Wrap back into HuggingFace Dataset if the input was one
    if hasattr(dataset, "from_dict"):
        from datasets import Dataset as HFDataset

        return HFDataset.from_dict(expanded)

    return expanded


def accumulate_pass_k(
    metrics: dict[str, list],
    task_id_column: str = "task_id",
    pass_column: str = "passed",
    k: int = 8,
) -> dict[str, dict[str, Any]]:
    """Accumulate per-trial pass/fail into pass^1 … pass^k.

    Expects ``metrics`` to contain per-batch lists for the task ID and
    pass/fail columns. Groups by task, then computes the probability
    of passing at least once in *n* trials for n=1..k.

    Args:
        metrics: Aggregated batch metrics (column name → list of per-batch dicts).
        task_id_column: Column holding the task identity.
        pass_column: Column holding 0/1 pass values.
        k: Maximum k for pass^k computation.

    Returns:
        Dict with ``pass^1`` through ``pass^k`` metrics.
    """
    task_ids_batches = metrics.get(task_id_column, [])
    pass_batches = metrics.get(pass_column, [])

    if not task_ids_batches or not pass_batches:
        return {}

    # Flatten batches: each batch is a dict with a "value" that is a list
    all_task_ids: list = []
    all_passes: list = []
    for tid_batch, pass_batch in zip(task_ids_batches, pass_batches):
        vals = tid_batch if isinstance(tid_batch, list) else tid_batch.get("value", [])
        pvals = pass_batch if isinstance(pass_batch, list) else pass_batch.get("value", [])
        all_task_ids.extend(vals)
        all_passes.extend(pvals)

    # Group by task
    from collections import defaultdict

    task_trials: dict[str, list[int]] = defaultdict(list)
    for tid, p in zip(all_task_ids, all_passes):
        task_trials[tid].append(int(p))

    results: dict[str, dict[str, Any]] = {}
    num_tasks = len(task_trials)

    for n in range(1, k + 1):
        # pass^n = fraction of tasks that pass at least once in n trials
        pass_count = 0
        for trials in task_trials.values():
            available = trials[:n]
            if any(available):
                pass_count += 1
        value = pass_count / num_tasks if num_tasks > 0 else 0.0
        results[f"pass^{n}"] = {"value": round(value, 4), "is_algebraic": True}

    return results


# ---------------------------------------------------------------------------
# Code execution sandbox (HumanEval+)
# ---------------------------------------------------------------------------


def run_code_in_sandbox(
    code: str,
    test_code: str,
    timeout_seconds: float = 30.0,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Execute generated code + test code in an isolated subprocess.

    Writes both snippets to a temp file and runs it. Returns pass/fail,
    stdout, stderr, and whether execution timed out.

    Args:
        code: The generated solution code.
        test_code: The test harness (typically ``check(candidate)`` style).
        timeout_seconds: Maximum wall-clock time for execution.
        python_executable: Python binary path (default: ``sys.executable``).

    Returns:
        ``{"passed": bool, "stdout": str, "stderr": str, "timed_out": bool,
          "return_code": int}``
    """
    python = python_executable or sys.executable

    combined = textwrap.dedent(code) + "\n\n" + textwrap.dedent(test_code)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
    ) as f:
        f.write(combined)
        temp_path = Path(f.name)

    try:
        proc = subprocess.run(
            [python, str(temp_path)],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        return {
            "passed": proc.returncode == 0,
            "stdout": proc.stdout[:2000],
            "stderr": proc.stderr[:2000],
            "timed_out": False,
            "return_code": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout_seconds}s",
            "timed_out": True,
            "return_code": -1,
        }
    finally:
        temp_path.unlink(missing_ok=True)
