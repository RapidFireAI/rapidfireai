"""
Tests for the evals PipelineScheduler bookkeeping.

These tests document the invariant the controller's main scheduling loop relies
on: an actor is marked busy by `schedule()` and MUST be freed by either
`set_completed_task(actor_id)` (success path) or `remove_pipeline(pipeline_id)`
(failure path). If neither is called, the actor leaks busy state and the
controller's busy-loop wedges indefinitely (see
`run_multi_pipeline_inference` in
`rapidfireai/evals/scheduling/controller.py`).
"""

import pytest

from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler


class TestPipelineSchedulerBookkeeping:
    """Cover the busy/free transitions in PipelineScheduler."""

    def test_schedule_marks_actor_busy(self):
        """schedule() must record actor->pipeline assignment."""
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=2, num_shards=3)
        result = scheduler.schedule()
        actor_id = result["actor_id"]
        pipeline_id = result["pipeline_id"]
        assert actor_id in (0, 1)
        assert pipeline_id in (1, 2)
        assert scheduler.actor_current_pipeline[actor_id] == pipeline_id

    def test_set_completed_task_frees_actor_and_advances_progress(self):
        """Success path: set_completed_task frees actor and counts the shard."""
        scheduler = PipelineScheduler(pipeline_ids=[1], num_actors=1, num_shards=2)
        result = scheduler.schedule()
        actor_id = result["actor_id"]
        assert scheduler.actor_current_pipeline[actor_id] == 1
        assert scheduler.pipeline_shards_completed[1] == 0

        scheduler.set_completed_task(actor_id)

        assert scheduler.actor_current_pipeline[actor_id] == -1
        assert scheduler.pipeline_shards_completed[1] == 1

    def test_remove_pipeline_frees_actor_on_dispatch_failure(self):
        """
        Failure path: when the controller cannot dispatch (e.g. actor init throws,
        a process_batch.remote raises synchronously, or the actor dies before any
        future is created), the controller must call remove_pipeline so the actor
        is freed and the rest of the experiment can proceed.
        """
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=2, num_shards=2)

        first = scheduler.schedule()
        actor_id = first["actor_id"]
        pipeline_id = first["pipeline_id"]
        assert scheduler.actor_current_pipeline[actor_id] == pipeline_id

        # Simulate dispatch failure: controller catches the exception and
        # decides to drop the pipeline. The bookkeeping fix is that the actor
        # gets freed, not left wedged.
        scheduler.remove_pipeline(pipeline_id)

        assert scheduler.actor_current_pipeline[actor_id] == -1
        assert pipeline_id not in scheduler.pipeline_ids

        # The next schedule() must hand out the freed actor (with the surviving
        # pipeline), not return the busy sentinel.
        second = scheduler.schedule()
        assert second["pipeline_id"] != -1
        assert second["actor_id"] != -1

    def test_dispatch_failure_does_not_wedge_remaining_pipelines(self):
        """
        End-to-end bookkeeping replay: 1 actor, 2 pipelines, the first
        dispatch fails. The freed actor must serve the surviving pipeline
        through completion -- the symptom of the bug we are fixing was an
        infinite loop of `{pipeline_id: -1}` returns from schedule().
        """
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)

        first = scheduler.schedule()
        failed_pipeline = first["pipeline_id"]
        actor_id = first["actor_id"]
        # Controller hits an exception and drops the pipeline.
        scheduler.remove_pipeline(failed_pipeline)
        assert scheduler.actor_current_pipeline[actor_id] == -1

        # Now the surviving pipeline should run on the freed actor.
        second = scheduler.schedule()
        survivor = second["pipeline_id"]
        assert survivor != -1
        assert survivor != failed_pipeline
        scheduler.set_completed_task(second["actor_id"])

        # All shards completed -- termination signal.
        terminal = scheduler.schedule()
        assert terminal["pipeline_id"] is None

    def test_all_actors_busy_returns_busy_sentinel(self):
        """When every actor is busy, schedule() must return {-1, -1, -1}."""
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=2, num_shards=4)
        scheduler.schedule()
        scheduler.schedule()
        # Both actors busy now.
        result = scheduler.schedule()
        assert result == {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

    def test_set_completed_task_idempotent_on_free_actor(self):
        """Calling set_completed_task on an already-free actor is a no-op."""
        scheduler = PipelineScheduler(pipeline_ids=[1], num_actors=1, num_shards=1)
        # Actor 0 starts free
        assert scheduler.actor_current_pipeline[0] == -1
        scheduler.set_completed_task(0)
        assert scheduler.actor_current_pipeline[0] == -1
        assert scheduler.pipeline_shards_completed[1] == 0
