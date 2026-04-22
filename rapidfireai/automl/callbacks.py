"""Callback protocols for inter-chunk/shard decision-making during experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class RunDecision:
    """Decision returned by a ChunkCallback after a fit-mode chunk completes."""

    action: Literal["continue", "prune"]
    replacement_config: dict[str, Any] | None = None


@dataclass
class PipelineDecision:
    """Decision returned by a ShardCallback after an evals-mode shard completes."""

    action: Literal["continue", "prune"]
    replacement_config: dict[str, Any] | None = None


class ChunkCallback(Protocol):
    """Protocol for callbacks invoked after each chunk completion in fit mode.

    Implementations receive per-run metrics after every chunk and return a
    decision that the controller uses to prune, continue, or replace runs.
    """

    def register_runs(self, run_id_to_config: dict[int, dict[str, Any]]) -> None:
        """Map newly created DB run IDs to their config dicts.

        Called once by the controller after ``_create_models`` produces initial
        runs, and again whenever replacement configs are created.
        """
        ...

    def on_chunk_complete(
        self,
        run_id: int,
        chunk_id: int,
        metrics: dict[str, Any],
    ) -> RunDecision:
        """Evaluate a run after it finishes a chunk.

        Args:
            run_id: Database run identifier.
            chunk_id: Zero-based index of the completed chunk.
            metrics: Latest metric values for this run (e.g. from MLflow).

        Returns:
            A ``RunDecision`` indicating whether to continue or prune, and
            optionally a replacement config dict.
        """
        ...

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Called after the experiment loop ends.

        Implementations should complete any remaining bookkeeping (e.g. tell an
        Optuna study the final objective values for still-running trials).
        """
        ...


class ShardCallback(Protocol):
    """Protocol for callbacks invoked after each shard completion in evals mode."""

    def register_pipelines(self, pipeline_id_to_config: dict[int, dict[str, Any]]) -> None:
        """Map newly created DB pipeline IDs to their config dicts."""
        ...

    def on_shard_complete(
        self,
        pipeline_id: int,
        shard_id: int,
        metrics: dict[str, Any],
    ) -> PipelineDecision:
        """Evaluate a pipeline after it finishes a shard.

        Args:
            pipeline_id: Database pipeline identifier.
            shard_id: Zero-based index of the completed shard.
            metrics: Cumulative aggregated metrics for this pipeline.

        Returns:
            A ``PipelineDecision`` indicating whether to continue or prune.
        """
        ...

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Called after the experiment loop ends."""
        ...
