"""
Pipeline Scheduler for Multi-Pipeline Inference.

Schedules pipelines to actors with fair round-robin scheduling using generations.
Ensures no pipeline is scheduled twice before all pipelines are scheduled once per generation.
"""


class PipelineScheduler:
    """
    Scheduler for assigning pipelines to actors with fair round-robin scheduling.

    Maintains generation-based fairness: no pipeline processes a second shard before
    all active pipelines have processed their current shard.
    """

    def __init__(self, pipeline_ids: list[int], num_actors: int, num_shards: int) -> None:
        """
        Initialize the pipeline scheduler.

        Args:
            pipeline_ids: List of pipeline IDs to schedule (1-indexed or any int)
            num_actors: Number of query processing actors available
            num_shards: Total number of shards in the dataset

        Note:
            - pipeline_ids: user-defined IDs (can be any int)
            - actor_ids: 0-indexed (0, 1, 2, ..., num_actors-1)
            - shard_ids: 0-indexed (0, 1, 2, ..., num_shards-1)
        """
        self.num_actors = num_actors
        self.num_shards = num_shards
        self.pipeline_ids = list(pipeline_ids)

        # Track which actor is running which pipeline (-1 means free)
        self.actor_current_pipeline = dict.fromkeys(range(num_actors), -1)

        # Track progress: how many shards each pipeline has completed
        self.pipeline_shards_completed = dict.fromkeys(pipeline_ids, 0)

        # Generation tracking for fair round-robin
        # Generation increments when all active pipelines have been scheduled once
        self.current_generation = 0
        self.pipelines_scheduled_in_generation = set()

    def add_pipeline(self, pipeline_id: int, shards_completed: int = 0) -> None:
        """
        Add a new pipeline to the scheduler (for dynamic pipeline addition).

        Args:
            pipeline_id: ID of the pipeline to add
            shards_completed: Number of shards already completed (default: 0)
        """
        if pipeline_id not in self.pipeline_ids:
            self.pipeline_ids.append(pipeline_id)

        self.pipeline_shards_completed[pipeline_id] = shards_completed

        # New pipeline starts in current generation
        # (it will be scheduled fairly with others)

    def set_completed_task(self, actor_id: int) -> None:
        """
        Mark a task as completed, freeing up the actor and updating pipeline progress.

        Args:
            actor_id: ID of the actor that completed the task
        """
        pipeline_id = self.actor_current_pipeline[actor_id]

        if pipeline_id != -1:
            # Free up the actor
            self.actor_current_pipeline[actor_id] = -1

            # Increment pipeline progress
            self.pipeline_shards_completed[pipeline_id] += 1

    def remove_pipeline(self, pipeline_id: int) -> int:
        """
        Remove a pipeline from the scheduler (for errors or user deletion).

        Args:
            pipeline_id: ID of the pipeline to remove

        Returns:
            Number of shards completed by this pipeline before removal
        """
        if pipeline_id not in self.pipeline_ids:
            return 0

        # Get progress before removing
        progress = self.pipeline_shards_completed.get(pipeline_id, 0)

        # Free up actor if this pipeline is running
        for actor_id in range(self.num_actors):
            if self.actor_current_pipeline[actor_id] == pipeline_id:
                self.actor_current_pipeline[actor_id] = -1

        # Remove from tracking structures
        self.pipeline_shards_completed.pop(pipeline_id, None)
        self.pipelines_scheduled_in_generation.discard(pipeline_id)

        if pipeline_id in self.pipeline_ids:
            self.pipeline_ids.remove(pipeline_id)

        return progress

    def schedule(self) -> dict[str, int | None]:
        """
        Schedule a single task with fair round-robin across pipelines.

        Scheduling rules:
        1. Fair round-robin: Use generation-based fairness
        2. No pipeline scheduled twice before all scheduled once (per generation)
        3. Pipelines process shards sequentially: 0, 1, 2, ..., num_shards-1

        Returns:
            Dictionary with keys:
            - If scheduling possible: {pipeline_id: int, actor_id: int, shard_id: int}
            - If all pipelines completed: {pipeline_id: None, actor_id: None, shard_id: None}
            - If all actors busy: {pipeline_id: -1, actor_id: -1, shard_id: -1}
        """
        # Check if all actors are busy
        available_actors = [
            actor_id for actor_id in range(self.num_actors) if self.actor_current_pipeline[actor_id] == -1
        ]
        if not available_actors:
            return {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

        # Check if all pipelines have completed all shards (termination)
        if all(self.pipeline_shards_completed[pid] >= self.num_shards for pid in self.pipeline_ids):
            return {"pipeline_id": None, "actor_id": None, "shard_id": None}

        # Get busy pipelines (currently being processed)
        busy_pipelines = {pid for pid in self.actor_current_pipeline.values() if pid != -1}

        # Get available pipelines (not busy, not completed)
        available_pipelines = [
            pid
            for pid in self.pipeline_ids
            if self.pipeline_shards_completed[pid] < self.num_shards and pid not in busy_pipelines
        ]

        # If no available pipelines, return busy state
        if not available_pipelines:
            return {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

        # Generation-based fair scheduling
        # Check if all active pipelines have been scheduled in this generation
        active_pipelines = [pid for pid in self.pipeline_ids if self.pipeline_shards_completed[pid] < self.num_shards]

        if len(self.pipelines_scheduled_in_generation) >= len(active_pipelines):
            # Start new generation
            self.current_generation += 1
            self.pipelines_scheduled_in_generation = set()

        # Filter available pipelines to those not yet scheduled in this generation
        unscheduled_in_generation = [
            pid for pid in available_pipelines if pid not in self.pipelines_scheduled_in_generation
        ]

        # If all available pipelines were scheduled, allow re-scheduling
        # (can happen if some pipelines are busy)
        if not unscheduled_in_generation:
            unscheduled_in_generation = available_pipelines

        # Select pipeline: prioritize least progress, then lowest pipeline_id for tie-breaking
        pipeline_id = min(unscheduled_in_generation, key=lambda pid: (self.pipeline_shards_completed[pid], pid))

        # Select first available actor
        actor_id = available_actors[0]

        # Next shard for this pipeline
        shard_id = self.pipeline_shards_completed[pipeline_id]

        # Update state
        self.actor_current_pipeline[actor_id] = pipeline_id
        self.pipelines_scheduled_in_generation.add(pipeline_id)

        return {"pipeline_id": pipeline_id, "actor_id": actor_id, "shard_id": shard_id}

    def get_status(self) -> dict:
        """
        Get current scheduler status for debugging and monitoring.

        Returns:
            Dictionary with scheduler state including:
            - active_pipelines: Number of pipelines not yet completed
            - busy_actors: Number of actors currently processing
            - completed_pipelines: Number of pipelines that finished all shards
            - current_generation: Current generation number
            - actor_assignments: Which actor is running which pipeline
            - pipeline_progress: Progress for each pipeline (shards_completed/num_shards)
        """
        completed_pipelines = [
            pid for pid in self.pipeline_ids if self.pipeline_shards_completed[pid] >= self.num_shards
        ]

        return {
            "num_actors": self.num_actors,
            "num_shards": self.num_shards,
            "active_pipelines": len(
                [pid for pid in self.pipeline_ids if self.pipeline_shards_completed[pid] < self.num_shards]
            ),
            "busy_actors": len([aid for aid in range(self.num_actors) if self.actor_current_pipeline[aid] != -1]),
            "completed_pipelines": len(completed_pipelines),
            "current_generation": self.current_generation,
            "pipelines_in_generation": len(self.pipelines_scheduled_in_generation),
            "actor_assignments": {
                actor_id: self.actor_current_pipeline[actor_id]
                for actor_id in range(self.num_actors)
                if self.actor_current_pipeline[actor_id] != -1
            },
            "pipeline_progress": {
                pid: f"{self.pipeline_shards_completed[pid]}/{self.num_shards}" for pid in self.pipeline_ids
            },
        }


# Export for external use
__all__ = ["PipelineScheduler"]
