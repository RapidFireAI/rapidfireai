import pytest

from rapidfireai.fit.backend.scheduler import Scheduler


class TestSchedulerFairRoundRobin:
    """Test suite for Scheduler fair round-robin behavior."""

    def test_basic_round_robin_single_worker(self):
        """Test that scheduler uses round-robin with a single worker."""
        run_ids = [1, 2, 3, 4]
        num_workers = 1
        num_chunks = 4

        scheduler = Scheduler(run_ids, num_workers, num_chunks)

        # Track which runs get scheduled
        scheduled_runs = []

        # Schedule first 4 tasks (should be runs 1, 2, 3, 4 in order)
        for _ in range(4):
            schedule = scheduler.schedule()
            assert schedule["run_id"] is not None and schedule["run_id"] != -1
            scheduled_runs.append(schedule["run_id"])
            # Simulate completion
            scheduler.set_completed_task(schedule["worker_id"])

        # Each run should be scheduled exactly once in the first round
        assert sorted(scheduled_runs) == [1, 2, 3, 4], (
            f"Expected each run scheduled once, got: {scheduled_runs}"
        )

    def test_no_run_scheduled_twice_before_all_scheduled_once(self):
        """Test that no run gets scheduled twice before all runs are scheduled once."""
        run_ids = [1, 2, 3, 4]
        num_workers = 1
        num_chunks = 4

        scheduler = Scheduler(run_ids, num_workers, num_chunks)

        # Track scheduling order
        scheduled_runs = []

        # Run through all chunks for all runs (16 total schedules)
        for _ in range(len(run_ids) * num_chunks):
            schedule = scheduler.schedule()
            if schedule["run_id"] is None:
                break
            scheduled_runs.append(schedule["run_id"])
            scheduler.set_completed_task(schedule["worker_id"])

        # Check round-robin property: in each group of 4, each run appears once
        for round_num in range(num_chunks):
            start_idx = round_num * len(run_ids)
            end_idx = start_idx + len(run_ids)
            round_runs = scheduled_runs[start_idx:end_idx]

            assert sorted(round_runs) == sorted(run_ids), (
                f"Round {round_num}: Expected {sorted(run_ids)}, got {sorted(round_runs)}. "
                f"Full schedule: {scheduled_runs}"
            )

    def test_fair_scheduling_after_epoch_reset(self):
        """
        Test that runs remain fair after epoch reset.

        This is the bug scenario: When Run 1 completes its epoch and gets reset to 0,
        it should NOT be scheduled before Runs 2, 3, 4 complete their current chunks.
        """
        run_ids = [1, 2, 3, 4]
        num_workers = 1
        num_chunks = 4

        scheduler = Scheduler(run_ids, num_workers, num_chunks)

        # Schedule all runs through their first 3 chunks (round-robin)
        for chunk_round in range(3):
            for _ in range(len(run_ids)):
                schedule = scheduler.schedule()
                assert schedule["run_id"] is not None
                scheduler.set_completed_task(schedule["worker_id"])

        # State: all runs have 3 chunks visited
        assert all(scheduler.run_visited_num_chunks[r] == 3 for r in run_ids)

        # Now Run 1 completes its 4th chunk (epoch complete)
        schedule = scheduler.schedule()
        assert schedule["run_id"] == 1  # Run 1 should be selected (tie-breaker: lowest id)
        scheduler.set_completed_task(schedule["worker_id"])

        # Run 1 has now completed epoch (4 chunks). Simulate reset.
        scheduler.reset_run(1)

        # State: Run 1 has 0, Runs 2, 3, 4 have 3
        assert scheduler.run_visited_num_chunks[1] == 0
        assert all(scheduler.run_visited_num_chunks[r] == 3 for r in [2, 3, 4])

        # BUG: The next schedule should be Run 2, 3, or 4 (they need to complete their epoch)
        # But the current implementation will pick Run 1 (because 0 < 3)
        schedule = scheduler.schedule()

        # This assertion will FAIL with the current buggy implementation
        # It should be Run 2 (next in line to complete its 4th chunk)
        # But it will be Run 1 (because it has 0 chunks visited)
        print(f"After reset: run_visited_chunks = {scheduler.run_visited_num_chunks}")
        print(f"Next scheduled run: {schedule['run_id']}")

        # Expected: Run 2 should be scheduled (to complete its epoch)
        # Actual (buggy): Run 1 will be scheduled (because 0 < 3)
        assert schedule["run_id"] in [2, 3, 4], (
            f"After epoch reset, runs that haven't completed their epoch should be scheduled first. "
            f"Got run_id={schedule['run_id']}, expected one of [2, 3, 4]. "
            f"run_visited_chunks={scheduler.run_visited_num_chunks}"
        )

    def test_fair_scheduling_multiple_epochs(self):
        """Test fair round-robin behavior across multiple epochs."""
        run_ids = [1, 2, 3, 4]
        num_workers = 1
        num_chunks = 4
        num_epochs = 3

        scheduler = Scheduler(run_ids, num_workers, num_chunks)

        all_schedules = []

        for epoch in range(num_epochs):
            epoch_schedules = []

            # Each epoch: all runs should complete all chunks in round-robin fashion
            for chunk_round in range(num_chunks):
                round_schedules = []
                for _ in range(len(run_ids)):
                    schedule = scheduler.schedule()
                    if schedule["run_id"] is None:
                        break
                    round_schedules.append(schedule["run_id"])
                    scheduler.set_completed_task(schedule["worker_id"])

                # Verify round-robin within each chunk round
                assert sorted(round_schedules) == sorted(run_ids), (
                    f"Epoch {epoch}, Chunk round {chunk_round}: "
                    f"Expected {sorted(run_ids)}, got {sorted(round_schedules)}"
                )
                epoch_schedules.extend(round_schedules)

            # Reset all runs for next epoch (simulating controller behavior)
            for run_id in run_ids:
                scheduler.reset_run(run_id)

            all_schedules.append(epoch_schedules)

        # Verify each epoch had fair scheduling
        for epoch, epoch_schedules in enumerate(all_schedules):
            run_counts = {r: epoch_schedules.count(r) for r in run_ids}
            assert all(c == num_chunks for c in run_counts.values()), (
                f"Epoch {epoch}: Each run should be scheduled {num_chunks} times, got {run_counts}"
            )

    def test_multi_worker_fair_scheduling(self):
        """Test fair round-robin with multiple workers."""
        run_ids = [1, 2, 3, 4]
        num_workers = 2
        num_chunks = 4

        scheduler = Scheduler(run_ids, num_workers, num_chunks)

        scheduled_runs = []

        # With 2 workers, runs can execute in parallel
        # But each run should still get equal opportunities
        while True:
            schedule = scheduler.schedule()
            if schedule["run_id"] is None:
                break
            if schedule["run_id"] == -1:
                # All workers busy, simulate one completion
                for w in range(num_workers):
                    if scheduler.worker_running_current_run[w] != -1:
                        scheduled_runs.append(scheduler.worker_running_current_run[w])
                        scheduler.set_completed_task(w)
                        break
            else:
                # Task scheduled but not yet complete
                pass

        # Complete any remaining tasks
        for w in range(num_workers):
            if scheduler.worker_running_current_run[w] != -1:
                scheduled_runs.append(scheduler.worker_running_current_run[w])
                scheduler.set_completed_task(w)

        # Each run should be scheduled exactly num_chunks times
        run_counts = {r: scheduled_runs.count(r) for r in run_ids}
        assert all(c == num_chunks for c in run_counts.values()), (
            f"Each run should be scheduled {num_chunks} times, got {run_counts}"
        )

    def test_epoch_reset_starvation(self):
        """
        Regression test: Verify that epoch reset doesn't cause run starvation.

        Scenario: Run 1 finishes epoch while Runs 2, 3, 4 are at chunk 3.
        After reset, Runs 2, 3, 4 should finish their chunks before Run 1 starts epoch 2.
        """
        run_ids = [1, 2, 3, 4]
        num_workers = 1
        num_chunks = 4

        scheduler = Scheduler(run_ids, num_workers, num_chunks)

        # Manually set up the state after Run 1 completes epoch and is reset:
        # Run 1: completed 1 epoch, now at 0 chunks in epoch 2
        # Runs 2, 3, 4: still in epoch 1, have 3 chunks visited (need 1 more)
        scheduler.run_visited_num_chunks = {1: 0, 2: 3, 3: 3, 4: 3}
        scheduler.run_epochs_completed = {1: 1, 2: 0, 3: 0, 4: 0}

        # Track next 3 schedules
        next_schedules = []
        for _ in range(3):
            schedule = scheduler.schedule()
            next_schedules.append(schedule["run_id"])
            scheduler.set_completed_task(schedule["worker_id"])

        print(f"Chunks: {scheduler.run_visited_num_chunks}")
        print(f"Epochs: {scheduler.run_epochs_completed}")
        print(f"Next 3 schedules: {next_schedules}")

        # Runs 2, 3, 4 should be scheduled to complete their epochs
        # NOT Run 1 which just started a new epoch
        assert next_schedules == [2, 3, 4], (
            f"Runs 2, 3, 4 should complete their epoch before Run 1 starts new epoch. "
            f"Got: {next_schedules}"
        )


class TestSchedulerBasicOperations:
    """Test basic scheduler operations."""

    def test_add_run(self):
        """Test adding a new run to the scheduler."""
        scheduler = Scheduler([1, 2], num_workers=1, num_chunks=4)

        assert 3 not in scheduler.run_ids
        scheduler.add_run(3, 0)
        assert 3 in scheduler.run_ids
        assert scheduler.run_visited_num_chunks[3] == 0

    def test_remove_run(self):
        """Test removing a run from the scheduler."""
        scheduler = Scheduler([1, 2, 3], num_workers=1, num_chunks=4)

        progress = scheduler.remove_run(2)
        assert 2 not in scheduler.run_ids
        assert 2 not in scheduler.run_visited_num_chunks

    def test_reset_run(self):
        """Test resetting a run's progress."""
        scheduler = Scheduler([1, 2], num_workers=1, num_chunks=4)
        scheduler.run_visited_num_chunks[1] = 3

        scheduler.reset_run(1)
        assert scheduler.run_visited_num_chunks[1] == 0

    def test_schedule_returns_none_when_all_complete(self):
        """Test that schedule returns None when all runs complete all chunks."""
        scheduler = Scheduler([1], num_workers=1, num_chunks=2)

        # Complete all chunks
        for _ in range(2):
            schedule = scheduler.schedule()
            scheduler.set_completed_task(schedule["worker_id"])

        # Next schedule should indicate completion
        schedule = scheduler.schedule()
        assert schedule["run_id"] is None

    def test_schedule_returns_busy_when_worker_busy(self):
        """Test that schedule returns -1 when all workers are busy."""
        scheduler = Scheduler([1, 2], num_workers=1, num_chunks=4)

        # Schedule first task (worker now busy)
        schedule1 = scheduler.schedule()
        assert schedule1["run_id"] == 1

        # Try to schedule another (should return busy)
        schedule2 = scheduler.schedule()
        assert schedule2["run_id"] == -1

