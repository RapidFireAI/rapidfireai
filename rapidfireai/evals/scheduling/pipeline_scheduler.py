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

        # Pipelines that have been stopped/deleted but still have an in-flight
        # shard running on some actor. The actor must NOT be freed in the
        # scheduler's view until the in-flight shard completes (otherwise the
        # scheduler reassigns the still-busy actor to a new pipeline, double-
        # booking it and orphaning the in-flight futures). The actor is freed
        # naturally via set_completed_task when the in-flight shard finishes.
        self.stopping_pipelines: set[int] = set()

    def add_pipeline(self, pipeline_id: int, shards_completed: int = 0) -> None:
        """
        Add a new pipeline to the scheduler (for dynamic pipeline addition).

        Also used by resume to re-add a previously stopped pipeline. In that
        case the pipeline may still be in ``stopping_pipelines`` (set by
        ``remove_pipeline(in_flight=True)`` when the stop raced with an
        in-flight shard). If we leave the marker in place, the in-flight
        shard's later ``set_completed_task`` would treat the pipeline as still
        stopping -- popping ``pipeline_shards_completed[pipeline_id]`` while
        the id remains in ``pipeline_ids`` -- and the next ``schedule()`` would
        raise ``KeyError``. Clear the marker here so the in-flight completion
        increments progress normally.

        Args:
            pipeline_id: ID of the pipeline to add
            shards_completed: Number of shards already completed (default: 0)
        """
        if pipeline_id not in self.pipeline_ids:
            self.pipeline_ids.append(pipeline_id)

        self.pipeline_shards_completed[pipeline_id] = shards_completed

        # Clear any residual stopping marker so a still-in-flight shard from
        # a prior stop completes as a normal (progress-incrementing) task.
        self.stopping_pipelines.discard(pipeline_id)

        # New pipeline starts in current generation
        # (it will be scheduled fairly with others)

    def set_completed_task(self, actor_id: int) -> bool:
        """
        Mark a task as completed, freeing up the actor and updating pipeline progress.

        Args:
            actor_id: ID of the actor that completed the task

        Returns:
            ``True`` if the completed task belonged to a stopping pipeline (one
            that was removed while still having an in-flight shard). In that
            case the actor is freed but pipeline progress is NOT incremented
            (the pipeline is being stopped, not advanced). ``False`` for a
            normal completion, where progress IS incremented.
        """
        pipeline_id = self.actor_current_pipeline[actor_id]

        if pipeline_id != -1:
            # Free up the actor
            self.actor_current_pipeline[actor_id] = -1

            if pipeline_id in self.stopping_pipelines:
                # The pipeline was removed (stop/delete) while this shard was
                # in flight. Do NOT increment progress -- the pipeline is no
                # longer being scheduled -- and clean up its residual state.
                self.stopping_pipelines.discard(pipeline_id)
                self.pipeline_shards_completed.pop(pipeline_id, None)
                return True

            # Normal completion: increment pipeline progress
            self.pipeline_shards_completed[pipeline_id] += 1
            return False

        return False

    def remove_pipeline(self, pipeline_id: int, in_flight: bool = False) -> int:
        """
        Remove a pipeline from the scheduler (for errors or user deletion/stop).

        Args:
            pipeline_id: ID of the pipeline to remove
            in_flight: ``True`` if this pipeline currently has an actor
                processing an in-flight shard. In that case the actor is NOT
                freed here -- it stays marked busy with this pipeline so the
                scheduler does not reassign it. The actor is freed naturally
                when the in-flight shard completes via ``set_completed_task``,
                which detects the pipeline in ``stopping_pipelines`` and skips
                the progress increment. ``False`` (default) frees any actor
                mapped to this pipeline immediately (pre-existing behavior for
                pipelines with no in-flight shard).

        Returns:
            Number of shards completed by this pipeline before removal
        """
        if pipeline_id not in self.pipeline_ids:
            # Already removed (e.g. stop issued twice). Still record a stopping
            # marker if requested so the in-flight completion is handled.
            if in_flight:
                self.stopping_pipelines.add(pipeline_id)
            return self.pipeline_shards_completed.get(pipeline_id, 0)

        # Get progress before removing
        progress = self.pipeline_shards_completed.get(pipeline_id, 0)

        if in_flight:
            # Do NOT free the actor -- it is still processing the in-flight
            # shard. Mark the pipeline as stopping so set_completed_task knows
            # to free the actor (without incrementing progress) when the shard
            # finishes. Keep pipeline_shards_completed until then so a stray
            # set_completed_task cannot KeyError on it.
            self.stopping_pipelines.add(pipeline_id)
        else:
            # No in-flight shard: free any actor mapped to this pipeline
            # (normally already -1) immediately.
            for actor_id in range(self.num_actors):
                if self.actor_current_pipeline[actor_id] == pipeline_id:
                    self.actor_current_pipeline[actor_id] = -1
            self.pipeline_shards_completed.pop(pipeline_id, None)

        # Remove from scheduling eligibility in both cases
        self.pipelines_scheduled_in_generation.discard(pipeline_id)

        if pipeline_id in self.pipeline_ids:
            self.pipeline_ids.remove(pipeline_id)

        return progress

    def rollback_last_schedule(self) -> None:
        """
        Roll back the scheduler state mutations made by the most recent
        successful ``schedule()`` call.

        The controller's dispatch safety net rejects a schedule when the
        returned actor is already in ``active_tasks`` (a scheduler /
        ``active_tasks`` divergence, e.g. a stop raced with completion).
        ``schedule()`` has *already* mutated state by that point -- it
        assigned the actor to the new pipeline and bumped generation
        bookkeeping -- so continuing without rolling back leaves the
        scheduler believing the new pipeline owns the actor while
        ``active_tasks`` still tracks the old in-flight shard. The old
        shard's later ``set_completed_task`` would then read the *new*
        pipeline from ``actor_current_pipeline`` and credit progress to it.
        This method restores the snapshot captured at the top of
        ``schedule()`` so the next ``schedule()`` re-derives the same
        decision from clean state.

        Safe to call when no schedule has been made yet, or when the last
        ``schedule()`` returned early without mutating (no snapshot is
        recorded in those cases).
        """
        snap = getattr(self, "_last_schedule_rollback", None)
        if not snap:
            return
        self.current_generation = snap["prev_generation"]
        self.pipelines_scheduled_in_generation = snap["prev_scheduled_in_generation"]
        actor_id = snap["actor_id"]
        if actor_id is not None:
            # The actor was available (== -1) before schedule() assigned it.
            self.actor_current_pipeline[actor_id] = -1
        self._last_schedule_rollback = None

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
        # Invalidate any leftover snapshot from a prior call; the early-return
        # paths below must leave nothing rollback-able.
        self._last_schedule_rollback = None
        if not available_actors:
            return {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

        # Check if all pipelines have completed all shards (termination).
        # Do NOT terminate while ``stopping_pipelines`` still holds busy actors
        # draining in-flight shards for stopped/deleted pipelines. Returning
        # termination here would make the controller ``break`` and kill those
        # actors mid-drain, skipping the in-flight shard's aggregation, progress
        # update, and STOPPED->COMPLETED flip. Signal "no work to schedule"
        # (-1) instead so the controller keeps looping and waits on the
        # in-flight futures via ``ray.wait``; ``set_completed_task`` clears the
        # stopping marker when the drain finishes, and the next ``schedule()``
        # then returns true termination.
        if all(self.pipeline_shards_completed[pid] >= self.num_shards for pid in self.pipeline_ids):
            if not self.stopping_pipelines:
                return {"pipeline_id": None, "actor_id": None, "shard_id": None}
            return {"pipeline_id": -1, "actor_id": -1, "shard_id": -1}

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

        # Snapshot state for rollback BEFORE any mutation. The controller's
        # dispatch safety net may reject this schedule (returned actor already
        # in active_tasks); rollback_last_schedule() restores this snapshot so
        # the scheduler doesn't keep a phantom actor assignment + generation
        # bump. Without rollback, a later set_completed_task for the old
        # in-flight shard would read the new pipeline from
        # actor_current_pipeline and credit progress to the wrong pipeline.
        self._last_schedule_rollback = {
            "actor_id": None,  # filled in after actor selection below
            "prev_generation": self.current_generation,
            "prev_scheduled_in_generation": set(self.pipelines_scheduled_in_generation),
        }

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
        # Record the chosen actor in the rollback snapshot so
        # rollback_last_schedule() can reset actor_current_pipeline[actor_id].
        self._last_schedule_rollback["actor_id"] = actor_id

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
            "stopping_pipelines": sorted(self.stopping_pipelines),
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
