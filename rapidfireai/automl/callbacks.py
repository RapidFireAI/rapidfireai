"""Callback protocols for inter-chunk/shard decision-making during experiments.

Classes
-------
RunDecision
    Dataclass returned by ``ChunkCallback.on_chunk_complete`` (fit mode).
PipelineDecision
    Dataclass returned by ``ShardCallback.on_shard_complete`` (evals mode).
ChunkCallback
    Protocol for fit-mode inter-chunk pruning callbacks.
ShardCallback
    Protocol for evals-mode inter-shard pruning callbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass
class RunDecision:
    """Decision returned by a ``ChunkCallback`` after a fit-mode chunk completes.

    Attributes
    ----------
    action : ``"continue"`` or ``"prune"``
    replacement_config : dict or None
        Config-leaf dict for a replacement run, or ``None``.
    """

    action: Literal["continue", "prune"]
    replacement_config: dict[str, Any] | None = None


@dataclass
class PipelineDecision:
    """Decision returned by a ``ShardCallback`` after an evals-mode shard completes.

    Attributes
    ----------
    action : ``"continue"`` or ``"prune"``
    replacement_config : dict or None
        Config-leaf dict for a replacement pipeline, or ``None``.
    """

    action: Literal["continue", "prune"]
    replacement_config: dict[str, Any] | None = None


class ChunkCallback(Protocol):
    """Protocol for callbacks invoked after each chunk in fit mode.

    Call order: ``register_runs`` → ``on_chunk_complete`` (repeated) → ``finalize``.
    """

    def register_runs(self, run_id_to_config: dict[int, dict[str, Any]]) -> None:
        """Map newly created DB run IDs to their config dicts."""
        ...

    def on_chunk_complete(
        self,
        run_id: int,
        chunk_id: int,
        metrics: dict[str, Any],
    ) -> RunDecision:
        """Evaluate a run after it finishes a chunk.

        Parameters
        ----------
        run_id : int
        chunk_id : int
        metrics : dict[str, Any]

        Returns
        -------
        RunDecision
        """
        ...

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Called after the experiment loop ends."""
        ...


class ShardCallback(Protocol):
    """Protocol for callbacks invoked after each shard in evals mode.

    Call order: ``register_pipelines`` → ``on_shard_complete`` (repeated) → ``finalize``.
    """

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

        Parameters
        ----------
        pipeline_id : int
        shard_id : int
        metrics : dict[str, Any]

        Returns
        -------
        PipelineDecision
        """
        ...

    def finalize(self, final_metrics: dict[int, dict[str, Any]]) -> None:
        """Called after the experiment loop ends."""
        ...
