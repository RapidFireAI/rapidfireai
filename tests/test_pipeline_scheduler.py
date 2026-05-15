"""Tests for PipelineScheduler bookkeeping.

Invariant: an actor marked busy by schedule() must be freed by either
set_completed_task(actor_id) or remove_pipeline(pipeline_id). If neither
is called, the controller's busy-loop wedges.
"""

from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler


class TestPipelineSchedulerBookkeeping:
    def test_schedule_marks_actor_busy(self):
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=2, num_shards=3)
        result = scheduler.schedule()
        actor_id = result["actor_id"]
        pipeline_id = result["pipeline_id"]
        assert actor_id in (0, 1)
        assert pipeline_id in (1, 2)
        assert scheduler.actor_current_pipeline[actor_id] == pipeline_id

    def test_set_completed_task_frees_actor_and_advances_progress(self):
        scheduler = PipelineScheduler(pipeline_ids=[1], num_actors=1, num_shards=2)
        result = scheduler.schedule()
        actor_id = result["actor_id"]
        assert scheduler.actor_current_pipeline[actor_id] == 1
        assert scheduler.pipeline_shards_completed[1] == 0

        scheduler.set_completed_task(actor_id)

        assert scheduler.actor_current_pipeline[actor_id] == -1
        assert scheduler.pipeline_shards_completed[1] == 1

    def test_remove_pipeline_frees_actor_on_dispatch_failure(self):
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=2, num_shards=2)

        first = scheduler.schedule()
        actor_id = first["actor_id"]
        pipeline_id = first["pipeline_id"]
        assert scheduler.actor_current_pipeline[actor_id] == pipeline_id

        scheduler.remove_pipeline(pipeline_id)

        assert scheduler.actor_current_pipeline[actor_id] == -1
        assert pipeline_id not in scheduler.pipeline_ids

        second = scheduler.schedule()
        assert second["pipeline_id"] != -1
        assert second["actor_id"] != -1

    def test_dispatch_failure_does_not_wedge_remaining_pipelines(self):
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)

        first = scheduler.schedule()
        failed_pipeline = first["pipeline_id"]
        actor_id = first["actor_id"]
        scheduler.remove_pipeline(failed_pipeline)
        assert scheduler.actor_current_pipeline[actor_id] == -1

        second = scheduler.schedule()
        survivor = second["pipeline_id"]
        assert survivor != -1
        assert survivor != failed_pipeline
        scheduler.set_completed_task(second["actor_id"])

        terminal = scheduler.schedule()
        assert terminal["pipeline_id"] is None

    def test_all_actors_busy_returns_busy_sentinel(self):
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=2, num_shards=4)
        scheduler.schedule()
        scheduler.schedule()
        result = scheduler.schedule()
        assert result == {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

    def test_set_completed_task_idempotent_on_free_actor(self):
        scheduler = PipelineScheduler(pipeline_ids=[1], num_actors=1, num_shards=1)
        assert scheduler.actor_current_pipeline[0] == -1
        scheduler.set_completed_task(0)
        assert scheduler.actor_current_pipeline[0] == -1
        assert scheduler.pipeline_shards_completed[1] == 0

    def test_actor_leaks_busy_when_neither_completion_nor_removal_called(self):
        """Regression: if dispatch fails and the controller forgets remove_pipeline,
        schedule() returns the busy sentinel indefinitely. This is the wedge the
        controller fix in run_multi_pipeline_inference prevents."""
        scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)
        scheduler.schedule()

        for _ in range(20):
            result = scheduler.schedule()
            assert result == {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}
