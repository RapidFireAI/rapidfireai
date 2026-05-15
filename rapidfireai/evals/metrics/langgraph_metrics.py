"""Generic trajectory-level diagnostic metrics for LangGraph agent evaluations.

These utilities are designed for use as ``compute_metrics_fn`` or inside
``accumulate_metrics_fn`` callbacks when ``capture_trajectory=True``.
"""

from __future__ import annotations

from typing import Any

from rapidfireai.evals.trajectory import AgentTrajectory


# ---------------------------------------------------------------------------
# Per-batch metrics (use inside compute_metrics_fn)
# ---------------------------------------------------------------------------


def compute_trajectory_metrics(batch: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compute trajectory-level diagnostics for a batch.

    Expects ``batch["trajectories"]`` to be a list of
    :class:`AgentTrajectory` and ``batch["agent_output"]`` for
    accuracy (with ``batch["expected_answer"]``).

    Returns metric dicts compatible with RF's ``{"MetricName": {"value": ..., "is_algebraic": True}}`` format.
    """
    trajectories: list[AgentTrajectory] = batch.get("trajectories", [])
    if not trajectories:
        return {}

    n = len(trajectories)

    avg_steps = sum(t.num_steps for t in trajectories) / n
    avg_tool_calls = sum(t.total_tool_calls for t in trajectories) / n
    avg_llm_calls = sum(t.total_llm_calls for t in trajectories) / n
    avg_tokens = sum(t.total_tokens for t in trajectories) / n
    avg_cost = sum(t.estimated_cost_usd for t in trajectories) / n
    avg_time = sum(t.total_time for t in trajectories) / n
    timeout_rate = sum(1 for t in trajectories if t.timed_out) / n
    error_rate = sum(t.error_rate for t in trajectories) / n

    return {
        "avg_steps": {"value": round(avg_steps, 2), "is_algebraic": True},
        "avg_tool_calls": {"value": round(avg_tool_calls, 2), "is_algebraic": True},
        "avg_llm_calls": {"value": round(avg_llm_calls, 2), "is_algebraic": True},
        "avg_tokens": {"value": round(avg_tokens, 1), "is_algebraic": True},
        "avg_cost_usd": {"value": round(avg_cost, 4), "is_algebraic": True},
        "avg_time_seconds": {"value": round(avg_time, 2), "is_algebraic": True},
        "timeout_rate": {"value": round(timeout_rate, 4), "is_algebraic": True},
        "error_rate": {"value": round(error_rate, 4), "is_algebraic": True},
    }


# ---------------------------------------------------------------------------
# Accumulation helper (use as accumulate_metrics_fn)
# ---------------------------------------------------------------------------


def accumulate_trajectory_metrics(
    aggregated_metrics: dict[str, list],
) -> dict[str, dict[str, Any]]:
    """Weighted-average accumulator for trajectory metrics.

    Takes the per-batch metric lists produced by
    :func:`compute_trajectory_metrics` and returns weighted averages
    across all batches.

    Also passes through any non-trajectory metrics (accuracy, etc.)
    using simple averaging.
    """
    _TRAJECTORY_KEYS = {
        "avg_steps",
        "avg_tool_calls",
        "avg_llm_calls",
        "avg_tokens",
        "avg_cost_usd",
        "avg_time_seconds",
        "timeout_rate",
        "error_rate",
    }

    result: dict[str, dict[str, Any]] = {}

    samples_list = aggregated_metrics.get("Samples Processed", [])
    batch_sizes = [
        (s.get("value", 1) if isinstance(s, dict) else s) for s in samples_list
    ]
    total_samples = sum(batch_sizes) if batch_sizes else 1

    for key, batch_values in aggregated_metrics.items():
        if key == "Samples Processed":
            continue

        if not batch_values:
            continue

        values = []
        for bv in batch_values:
            if isinstance(bv, dict):
                values.append(bv.get("value", 0))
            else:
                values.append(bv)

        if key in _TRAJECTORY_KEYS:
            # Weighted average by batch size
            if len(values) == len(batch_sizes):
                weighted = sum(v * w for v, w in zip(values, batch_sizes))
                avg = weighted / total_samples if total_samples > 0 else 0
            else:
                avg = sum(values) / len(values) if values else 0
            result[key] = {"value": round(avg, 4), "is_algebraic": True}
        else:
            # Pass-through: use first batch's metadata if present
            is_algebraic = False
            if batch_values and isinstance(batch_values[0], dict):
                is_algebraic = batch_values[0].get("is_algebraic", False)

            if len(values) == len(batch_sizes) and is_algebraic:
                weighted = sum(v * w for v, w in zip(values, batch_sizes))
                avg = weighted / total_samples if total_samples > 0 else 0
            else:
                avg = sum(values) / len(values) if values else 0

            result[key] = {"value": round(avg, 4) if isinstance(avg, float) else avg, "is_algebraic": is_algebraic}

    return result


# ---------------------------------------------------------------------------
# Failure mode analysis
# ---------------------------------------------------------------------------


def analyze_failure_modes(
    trajectories: list[AgentTrajectory],
    labels: list[bool] | None = None,
) -> dict[str, Any]:
    """Analyze common failure patterns across trajectories.

    Args:
        trajectories: List of agent trajectories.
        labels: Optional pass/fail labels (True = passed) aligned
            with trajectories. If provided, analysis is restricted
            to failed runs.

    Returns:
        Dict with failure mode counts and percentages.
    """
    if labels is not None:
        failed_trajs = [t for t, ok in zip(trajectories, labels) if not ok]
    else:
        failed_trajs = trajectories

    if not failed_trajs:
        return {"total_analyzed": 0}

    n = len(failed_trajs)

    # Categorize failures
    timeout_count = sum(1 for t in failed_trajs if t.timed_out)
    error_count = sum(1 for t in failed_trajs if any(s.error for s in t.steps))

    tool_error_count = 0
    wrong_tool_count = 0
    for t in failed_trajs:
        for step in t.steps:
            for tc in step.tool_calls:
                if tc.get("error"):
                    tool_error_count += 1

    # Most common failing nodes
    from collections import Counter

    failing_nodes = Counter()
    for t in failed_trajs:
        for step in t.steps:
            if step.error:
                failing_nodes[step.node_name] += 1

    return {
        "total_analyzed": n,
        "timeout_fraction": round(timeout_count / n, 3),
        "error_fraction": round(error_count / n, 3),
        "tool_errors": tool_error_count,
        "top_failing_nodes": dict(failing_nodes.most_common(5)),
    }
