#!/usr/bin/env python3
"""
Comprehensive pytest tests for the Scheduler module.
Tests the Monte Carlo-based scheduler for multi-worker distributed training.
"""

from unittest.mock import patch

import pytest

from rapidfireai.backend.scheduler import Scheduler


class TestSchedulerInitialization:
    """Test scheduler initialization and basic setup"""

    def test_basic_initialization(self):
        """Test basic scheduler initialization with runs_info format"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0, "start_chunk_id": 0},
            {"run_id": 2, "req_workers": 2, "estimated_runtime": 15.0, "start_chunk_id": 0},
            {"run_id": 3, "req_workers": 1, "estimated_runtime": 8.0, "start_chunk_id": 0},
        ]

        scheduler = Scheduler(runs_info, num_workers=4, num_chunks=10, num_simulations=50)

        assert scheduler.n_workers == 4
        assert scheduler.n_chunks == 10
        assert scheduler.num_simulations == 50
        assert scheduler.n_runs == 3
        assert scheduler.run_ids == [1, 2, 3]

        # Check run parameters
        assert scheduler.run_req_workers[1] == 1
        assert scheduler.run_req_workers[2] == 2
        assert scheduler.run_req_workers[3] == 1
        assert scheduler.run_estimated_runtime[1] == 10.0
        assert scheduler.run_estimated_runtime[2] == 15.0
        assert scheduler.run_estimated_runtime[3] == 8.0

    def test_initialization_with_defaults(self):
        """Test initialization with default values"""
        runs_info = [{"run_id": 1}]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        assert scheduler.num_simulations == 100  # Default value
        assert scheduler.run_req_workers[1] == 1  # Default value
        assert scheduler.run_estimated_runtime[1] == 1.0  # Default value
        assert scheduler.run_start_chunk_id[1] == 0  # Default value

    def test_initialization_empty_runs(self):
        """Test initialization with empty runs list"""
        scheduler = Scheduler([], num_workers=3, num_chunks=5)

        assert scheduler.n_runs == 0
        assert scheduler.run_ids == []
        assert scheduler.n_workers == 3
        assert scheduler.n_chunks == 5

    def test_initial_state(self):
        """Test that initial state is properly set"""
        runs_info = [{"run_id": 1}, {"run_id": 2}]
        scheduler = Scheduler(runs_info, num_workers=3, num_chunks=4)

        # All workers should be idle
        assert all(worker == -1 for worker in scheduler.state.worker_running_current_run.values())

        # All runs should have 0 progress
        assert all(progress == 0 for progress in scheduler.state.run_visited_num_chunks.values())

        # No runs should have assigned workers
        assert all(len(workers) == 0 for workers in scheduler.state.run_assigned_workers.values())

        # No completion times set
        assert len(scheduler.state.run_completion_time) == 0


class TestRunManagement:
    """Test run management operations"""

    def test_add_run_basic(self):
        """Test adding a new run to the scheduler"""
        scheduler = Scheduler([], num_workers=2, num_chunks=5)

        run_info = {"run_id": 1, "req_workers": 2, "estimated_runtime": 12.0, "start_chunk_id": 0}

        scheduler.add_run(run_info, run_visited_num_chunks=0)

        assert 1 in scheduler.run_ids
        assert scheduler.n_runs == 1
        assert scheduler.run_req_workers[1] == 2
        assert scheduler.run_estimated_runtime[1] == 12.0
        assert scheduler.state.run_visited_num_chunks[1] == 0
        assert len(scheduler.state.run_assigned_workers[1]) == 0

    def test_add_run_with_progress(self):
        """Test adding a run with existing progress"""
        scheduler = Scheduler([], num_workers=2, num_chunks=10)

        run_info = {"run_id": 1, "req_workers": 1, "estimated_runtime": 5.0}
        scheduler.add_run(run_info, run_visited_num_chunks=3)

        assert scheduler.state.run_visited_num_chunks[1] == 3

    def test_add_existing_run(self):
        """Test adding a run that already exists"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 5.0}]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        # Add same run with different parameters
        run_info = {"run_id": 1, "req_workers": 2, "estimated_runtime": 8.0}
        scheduler.add_run(run_info, run_visited_num_chunks=2)

        # Should update parameters but not add duplicate
        assert scheduler.n_runs == 1
        assert scheduler.run_req_workers[1] == 2
        assert scheduler.run_estimated_runtime[1] == 8.0
        assert scheduler.state.run_visited_num_chunks[1] == 2

    def test_remove_run(self):
        """Test removing a run from the scheduler"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 5.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        # Set some progress
        scheduler.state.run_visited_num_chunks[1] = 3

        # Remove run 1
        progress = scheduler.remove_run(1)

        assert progress == 3
        assert 1 not in scheduler.run_ids
        assert scheduler.n_runs == 1
        assert 1 not in scheduler.state.run_visited_num_chunks
        assert 1 not in scheduler.run_req_workers
        assert 1 not in scheduler.run_estimated_runtime
        assert 1 not in scheduler.state.run_assigned_workers

    def test_remove_nonexistent_run(self):
        """Test removing a run that doesn't exist"""
        scheduler = Scheduler([{"run_id": 1}], num_workers=2, num_chunks=5)

        progress = scheduler.remove_run(999)
        assert progress == 0
        assert scheduler.n_runs == 1

    def test_reset_run(self):
        """Test resetting a run to start from beginning"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 5.0}]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        # Set some progress and assign workers
        scheduler.state.run_visited_num_chunks[1] = 3
        scheduler.state.worker_running_current_run[0] = 1
        scheduler.state.run_assigned_workers[1].add(0)
        scheduler.state.run_completion_time[1] = 10.0

        # Reset the run
        scheduler.reset_run(1)

        assert scheduler.state.run_visited_num_chunks[1] == 0
        assert scheduler.state.worker_running_current_run[0] == -1
        assert len(scheduler.state.run_assigned_workers[1]) == 0

    def test_reset_nonexistent_run(self):
        """Test resetting a run that doesn't exist"""
        scheduler = Scheduler([{"run_id": 1}], num_workers=2, num_chunks=5)

        # Should not raise error
        scheduler.reset_run(999)
        assert scheduler.n_runs == 1


class TestTaskCompletion:
    """Test task completion functionality"""

    def test_set_completed_task(self):
        """Test marking a task as completed"""
        runs_info = [{"run_id": 1, "req_workers": 2, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=3, num_chunks=5)

        # Assign workers to the run
        scheduler.state.worker_running_current_run[0] = 1
        scheduler.state.worker_running_current_run[1] = 1
        scheduler.state.run_assigned_workers[1].add(0)
        scheduler.state.run_assigned_workers[1].add(1)
        scheduler.state.run_completion_time[1] = 15.0

        # Complete the task
        scheduler.set_completed_task(1)

        # Workers should be freed
        assert scheduler.state.worker_running_current_run[0] == -1
        assert scheduler.state.worker_running_current_run[1] == -1
        assert len(scheduler.state.run_assigned_workers[1]) == 0
        assert 1 not in scheduler.state.run_completion_time
        assert scheduler.state.run_visited_num_chunks[1] == 1

    def test_set_completed_task_nonexistent(self):
        """Test completing a task for a run that doesn't exist"""
        scheduler = Scheduler([], num_workers=2, num_chunks=5)

        # Should not raise error
        scheduler.set_completed_task(999)
        assert scheduler.n_runs == 0


class TestScheduling:
    """Test the main scheduling functionality"""

    def test_schedule_basic(self):
        """Test basic scheduling with available workers and runs"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5, num_simulations=10)

        # Mock the simulation to return a predictable result
        with patch.object(scheduler, "_simulate_random_schedule") as mock_sim:
            mock_sim.return_value = (
                20.0,
                [{"run_id": 1, "worker_ids": (0,), "chunk_id": 0, "start_time": 0.0, "end_time": 10.0}],
            )

            result = scheduler.schedule()

            assert result is not None
            assert result["run_id"] == 1
            assert result["worker_ids"] == (0,)
            assert result["chunk_id"] == 0
            assert result["is_last_chunk"] is False

            # Check that workers are assigned
            assert scheduler.state.worker_running_current_run[0] == 1
            assert 0 in scheduler.state.run_assigned_workers[1]

    def test_schedule_all_workers_busy(self):
        """Test scheduling when all workers are busy"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=5)

        # Make all workers busy
        scheduler.state.worker_running_current_run[0] = 1

        result = scheduler.schedule()

        assert result["run_id"] == -1
        assert result["worker_ids"] is None
        assert result["chunk_id"] == -1
        assert result["is_last_chunk"] is None

    def test_schedule_all_runs_completed(self):
        """Test scheduling when all runs have completed all chunks"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=2)

        # Mark run as completed
        scheduler.state.run_visited_num_chunks[1] = 2

        result = scheduler.schedule()

        assert result["run_id"] is None
        assert result["worker_ids"] is None
        assert result["chunk_id"] is None
        assert result["is_last_chunk"] is None

    def test_schedule_no_schedulable_runs(self):
        """Test scheduling when no runs can be scheduled"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=5)

        # Make run busy (assigned workers)
        scheduler.state.run_assigned_workers[1].add(0)

        result = scheduler.schedule()

        assert result["run_id"] == -1
        assert result["worker_ids"] is None
        assert result["chunk_id"] == -1
        assert result["is_last_chunk"] is None

    def test_schedule_multi_worker_run(self):
        """Test scheduling a run that requires multiple workers"""
        runs_info = [{"run_id": 1, "req_workers": 3, "estimated_runtime": 15.0}]
        scheduler = Scheduler(runs_info, num_workers=4, num_chunks=5, num_simulations=10)

        with patch.object(scheduler, "_simulate_random_schedule") as mock_sim:
            mock_sim.return_value = (
                15.0,
                [{"run_id": 1, "worker_ids": (0, 1, 2), "chunk_id": 0, "start_time": 0.0, "end_time": 15.0}],
            )

            result = scheduler.schedule()

            assert result["run_id"] == 1
            assert result["worker_ids"] == (0, 1, 2)
            assert result["chunk_id"] == 0

            # Check that all workers are assigned
            for worker_id in [0, 1, 2]:
                assert scheduler.state.worker_running_current_run[worker_id] == 1
                assert worker_id in scheduler.state.run_assigned_workers[1]

    def test_insufficient_workers_validation(self):
        """Test that ValueError is raised when runs require more workers than available"""
        # Test in initialization
        runs_info = [{"run_id": 1, "req_workers": 3, "estimated_runtime": 10.0}]

        with pytest.raises(ValueError, match="Run 1 requires 3 workers but only 2 workers are available"):
            Scheduler(runs_info, num_workers=2, num_chunks=5)

        # Test in add_run
        scheduler = Scheduler([], num_workers=2, num_chunks=5)

        with pytest.raises(ValueError, match="Run 1 requires 3 workers but only 2 workers are available"):
            scheduler.add_run({"run_id": 1, "req_workers": 3, "estimated_runtime": 10.0})

    def test_schedule_last_chunk_detection(self):
        """Test that is_last_chunk is correctly detected"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=3)

        # Set run to be on last chunk
        scheduler.state.run_visited_num_chunks[1] = 2  # 0-indexed, so chunk 2 is last

        with patch.object(scheduler, "_simulate_random_schedule") as mock_sim:
            mock_sim.return_value = (
                10.0,
                [{"run_id": 1, "worker_ids": (0,), "chunk_id": 2, "start_time": 0.0, "end_time": 10.0}],
            )

            result = scheduler.schedule()

            assert result["is_last_chunk"] is True


class TestMonteCarloSimulation:
    """Test Monte Carlo simulation functionality"""

    def test_simulate_random_schedule_basic(self):
        """Test basic simulation functionality"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=2, num_simulations=5)

        makespan, schedule_sequence = scheduler._simulate_random_schedule(scheduler.state, seed=42)

        assert isinstance(makespan, float)
        assert makespan > 0
        assert isinstance(schedule_sequence, list)

        # Check that all runs complete all chunks
        completed_chunks = {}
        for task in schedule_sequence:
            run_id = task["run_id"]
            completed_chunks[run_id] = completed_chunks.get(run_id, 0) + 1

        # Each run should complete at least n_chunks (allowing for some flexibility)
        for run_id in scheduler.run_ids:
            assert completed_chunks.get(run_id, 0) >= scheduler.n_chunks

    def test_simulate_random_schedule_with_seed(self):
        """Test that simulation is deterministic with same seed"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=2, num_simulations=5)

        # Run simulation twice with same seed
        makespan1, seq1 = scheduler._simulate_random_schedule(scheduler.state, seed=123)
        makespan2, seq2 = scheduler._simulate_random_schedule(scheduler.state, seed=123)

        assert makespan1 == makespan2
        assert seq1 == seq2

    def test_simulate_random_schedule_different_seeds(self):
        """Test that different seeds produce different results"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=2, num_simulations=5)

        makespan1, seq1 = scheduler._simulate_random_schedule(scheduler.state, seed=123)
        makespan2, seq2 = scheduler._simulate_random_schedule(scheduler.state, seed=456)

        # Results might be the same by chance, but likely different
        # We just test that the method doesn't crash with different seeds
        assert isinstance(makespan1, float)
        assert isinstance(makespan2, float)
        assert isinstance(seq1, list)
        assert isinstance(seq2, list)

    def test_simulate_random_schedule_state_preservation(self):
        """Test that simulation doesn't modify the original state"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=2, num_simulations=5)

        # Save original state
        original_workers = scheduler.state.worker_running_current_run.copy()
        original_progress = scheduler.state.run_visited_num_chunks.copy()
        original_assignments = scheduler.state.run_assigned_workers.copy()

        # Run simulation
        scheduler._simulate_random_schedule(scheduler.state, seed=42)

        # Check that state is unchanged
        assert scheduler.state.worker_running_current_run == original_workers
        assert scheduler.state.run_visited_num_chunks == original_progress
        assert scheduler.state.run_assigned_workers == original_assignments

    def test_simulate_random_schedule_empty_runs(self):
        """Test simulation with no runs"""
        scheduler = Scheduler([], num_workers=2, num_chunks=5, num_simulations=5)

        makespan, schedule_sequence = scheduler._simulate_random_schedule(scheduler.state, seed=42)

        assert makespan == 0.0
        assert schedule_sequence == []


class TestHelperMethods:
    """Test helper methods"""

    def test_get_available_workers(self):
        """Test getting available workers"""
        scheduler = Scheduler([], num_workers=3, num_chunks=5)

        # All workers should be available initially
        available = scheduler._get_available_workers(scheduler.state)
        assert set(available) == {0, 1, 2}

        # Make some workers busy
        scheduler.state.worker_running_current_run[0] = 1
        scheduler.state.worker_running_current_run[2] = 2

        available = scheduler._get_available_workers(scheduler.state)
        assert available == [1]

    def test_get_schedulable_runs(self):
        """Test getting schedulable runs"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        # All runs should be schedulable initially
        schedulable = scheduler._get_schedulable_runs(scheduler.state)
        assert set(schedulable) == {1, 2}

        # Make one run busy
        scheduler.state.run_assigned_workers[1].add(0)

        schedulable = scheduler._get_schedulable_runs(scheduler.state)
        assert schedulable == [2]

    def test_get_next_chunk_id(self):
        """Test getting next chunk ID for a run"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0, "start_chunk_id": 2}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=5)

        # Test with different progress levels
        assert scheduler._get_next_chunk_id(scheduler.state, 1) == 2  # 0 + 2 = 2
        scheduler.state.run_visited_num_chunks[1] = 1
        assert scheduler._get_next_chunk_id(scheduler.state, 1) == 3  # 1 + 2 = 3
        scheduler.state.run_visited_num_chunks[1] = 3
        assert scheduler._get_next_chunk_id(scheduler.state, 1) == 0  # (3 + 2) % 5 = 0

    def test_scheduler_state_copy(self):
        """Test SchedulerState copying functionality"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        # Modify state
        scheduler.state.worker_running_current_run[0] = 1
        scheduler.state.run_visited_num_chunks[1] = 2
        scheduler.state.run_assigned_workers[1].add(0)
        scheduler.state.run_completion_time[1] = 15.0

        # Copy state
        state_copy = scheduler.state.copy()

        # Modify original state further
        scheduler.state.worker_running_current_run[1] = 1
        scheduler.state.run_visited_num_chunks[1] = 3

        # Check that copy is independent
        assert state_copy.worker_running_current_run[0] == 1
        assert state_copy.worker_running_current_run[1] == -1
        assert state_copy.run_visited_num_chunks[1] == 2
        assert 0 in state_copy.run_assigned_workers[1]
        assert state_copy.run_completion_time[1] == 15.0


class TestStatusReporting:
    """Test status reporting functionality"""

    def test_get_status_basic(self):
        """Test basic status reporting"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 2, "estimated_runtime": 15.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=3, num_chunks=5)

        status = scheduler.get_status()

        assert "active_runs" in status
        assert "busy_workers" in status
        assert "completed_runs" in status
        assert "run_progress" in status
        assert "run_workers" in status
        assert "current_chunks" in status

        assert status["active_runs"] == 2
        assert status["busy_workers"] == 0
        assert status["completed_runs"] == 0

    def test_get_status_with_progress(self):
        """Test status reporting with some progress"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5)

        # Set some progress
        scheduler.state.run_visited_num_chunks[1] = 5  # Completed
        scheduler.state.run_visited_num_chunks[2] = 2  # In progress
        scheduler.state.worker_running_current_run[0] = 2
        scheduler.state.run_assigned_workers[2].add(0)

        status = scheduler.get_status()

        assert status["active_runs"] == 1  # Only run 2
        assert status["busy_workers"] == 1  # Worker 0
        assert status["completed_runs"] == 1  # Run 1
        assert status["run_progress"] == {1: "5/5", 2: "2/5"}
        assert status["run_workers"] == {1: "0/1", 2: "1/1"}


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def test_zero_workers(self):
        """Test scheduler with zero workers raises exception when runs require workers"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]

        with pytest.raises(ValueError, match="Run 1 requires 1 workers but only 0 workers are available"):
            Scheduler(runs_info, num_workers=0, num_chunks=5)

        # Test with zero workers and no runs (should work)
        scheduler = Scheduler([], num_workers=0, num_chunks=5)
        result = scheduler.schedule()
        assert result["run_id"] is None
        assert result["worker_ids"] is None
        assert result["chunk_id"] is None

    def test_zero_chunks(self):
        """Test scheduler with zero chunks"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=0)

        result = scheduler.schedule()
        assert result["run_id"] is None
        assert result["worker_ids"] is None
        assert result["chunk_id"] is None

    def test_empty_runs(self):
        """Test scheduler with no runs"""
        scheduler = Scheduler([], num_workers=2, num_chunks=5)

        result = scheduler.schedule()
        assert result["run_id"] is None
        assert result["worker_ids"] is None
        assert result["chunk_id"] is None

    def test_negative_runtime(self):
        """Test scheduler with negative runtime"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": -5.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=5)

        # Should not crash, but behavior is undefined
        result = scheduler.schedule()
        assert result is not None

    def test_zero_runtime(self):
        """Test scheduler with zero runtime"""
        runs_info = [{"run_id": 1, "req_workers": 1, "estimated_runtime": 0.0}]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=5)

        result = scheduler.schedule()
        assert result is not None

    def test_very_large_worker_requirement(self):
        """Test scheduler with worker requirement larger than available workers raises exception"""
        runs_info = [{"run_id": 1, "req_workers": 100, "estimated_runtime": 10.0}]

        with pytest.raises(ValueError, match="Run 1 requires 100 workers but only 2 workers are available"):
            Scheduler(runs_info, num_workers=2, num_chunks=5)


class TestIntegrationScenarios:
    """Test complex integration scenarios"""

    def test_large_scale_scheduling(self):
        """Test scheduling 10 runs on 3 workers with 4 chunks each, using different worker requirements"""
        # Create 10 runs with different worker requirements and runtimes
        runs_info = []
        for i in range(1, 11):
            # Vary worker requirements: 1, 2, 3, 1, 2, 3, 1, 2, 3, 1
            req_workers = ((i - 1) % 3) + 1
            runs_info.append(
                {
                    "run_id": i,
                    "req_workers": req_workers,
                    "estimated_runtime": 5.0 + (i % 3) * 2.0,  # Varying runtimes: 5.0, 7.0, 9.0
                    "start_chunk_id": 0,  # i % 4,  # Different starting chunks
                }
            )

        scheduler = Scheduler(
            runs_info, num_workers=3, num_chunks=4, num_simulations=10
        )  # Reduced simulations for faster testing

        # Track all assignments and progress
        all_assignments = []
        max_iterations = 200  # Prevent infinite loop
        iteration = 0
        runs_in_progress = set()  # Track which runs are currently being processed

        print("\n=== Starting Large Scale Scheduling Test ===")
        print(f"Runs: {len(runs_info)}, Workers: {scheduler.n_workers}, Chunks per run: {scheduler.n_chunks}")
        print(f"Worker requirements by run: {[run['req_workers'] for run in runs_info]}")
        print(f"Start chunks by run: {[run['start_chunk_id'] for run in runs_info]}")
        print()

        # Simulate complete training
        while iteration < max_iterations:
            result = scheduler.schedule()

            if result["run_id"] is None:  # All runs completed
                print(f"\n✓ All runs completed after {iteration} iterations")
                break

            elif result["run_id"] == -1:  # No workers available
                # Complete one of the runs in progress
                if runs_in_progress:
                    # In a real scenario, we'd complete the run that finishes first
                    # For testing, we'll complete the first one in the set
                    run_to_complete = next(iter(runs_in_progress))
                    scheduler.set_completed_task(run_to_complete)
                    runs_in_progress.remove(run_to_complete)
                    print(f"  [{iteration}] Completed Run {run_to_complete}, freed workers")
                else:
                    print(f"ERROR: No workers available but no runs in progress at iteration {iteration}")
                    break

            else:
                # Successfully scheduled a run
                run_id = result["run_id"]
                worker_ids = result["worker_ids"]
                chunk_id = result["chunk_id"]

                all_assignments.append(result)
                runs_in_progress.add(run_id)

                print(f"  [{iteration}] Scheduled Run {run_id} on Workers {worker_ids} for Chunk {chunk_id}")

                # Simulate some runs completing (every 3rd iteration, complete some runs)
                if iteration % 3 == 2 and runs_in_progress:
                    # Complete 1-2 runs randomly
                    num_to_complete = min(len(runs_in_progress), 2)
                    runs_to_complete = list(runs_in_progress)[:num_to_complete]

                    for run_id in runs_to_complete:
                        scheduler.set_completed_task(run_id)
                        runs_in_progress.remove(run_id)
                        print(f"    → Completed Run {run_id}")

            iteration += 1

            # Print progress every 20 iterations
            if iteration % 20 == 0:
                progress = {run_id: scheduler.state.run_visited_num_chunks[run_id] for run_id in scheduler.run_ids}
                print(f"\n  Progress at iteration {iteration}: {progress}")
                print(f"  Runs in progress: {runs_in_progress}\n")

        # Complete any remaining runs
        while runs_in_progress:
            run_id = runs_in_progress.pop()
            scheduler.set_completed_task(run_id)
            print(f"  Final cleanup: Completed Run {run_id}")

        print("\n=== Test Results ===")
        print(f"Total iterations: {iteration}")
        print(f"Total assignments made: {len(all_assignments)}")
        print("Final progress by run:")
        for run_id in sorted(scheduler.run_ids):
            progress = scheduler.state.run_visited_num_chunks[run_id]
            expected = scheduler.n_chunks
            status = "✓" if progress == expected else "✗"
            print(f"  Run {run_id}: {progress}/{expected} chunks {status}")

        # Verification assertions
        print("\n=== Verification ===")

        # 1. Verify that all runs completed all chunks
        all_completed = True
        for run_id in scheduler.run_ids:
            chunks_done = scheduler.state.run_visited_num_chunks[run_id]
            if chunks_done != scheduler.n_chunks:
                print(f"✗ Run {run_id} only completed {chunks_done}/{scheduler.n_chunks} chunks")
                all_completed = False

        if all_completed:
            print("✓ All runs completed all chunks")

        # 2. Verify that all workers are idle
        all_idle = True
        for worker_id in range(scheduler.n_workers):
            if scheduler.state.worker_running_current_run[worker_id] != -1:
                print(
                    f"✗ Worker {worker_id} is still assigned to run {scheduler.state.worker_running_current_run[worker_id]}"
                )
                all_idle = False

        if all_idle:
            print("✓ All workers are idle")

        # 3. Verify minimum number of assignments (10 runs * 4 chunks = 40 minimum)
        min_assignments = len(scheduler.run_ids) * scheduler.n_chunks
        if len(all_assignments) >= min_assignments:
            print(f"✓ Made at least {min_assignments} assignments (actual: {len(all_assignments)})")
        else:
            print(f"✗ Expected at least {min_assignments} assignments, got {len(all_assignments)}")

        # 4. Verify all runs were scheduled
        scheduled_runs = set(assignment["run_id"] for assignment in all_assignments)
        if len(scheduled_runs) == len(scheduler.run_ids):
            print(f"✓ All {len(scheduler.run_ids)} runs were scheduled")
        else:
            print(f"✗ Only {len(scheduled_runs)}/{len(scheduler.run_ids)} runs were scheduled")
            missing = set(scheduler.run_ids) - scheduled_runs
            if missing:
                print(f"  Missing runs: {missing}")

        # 5. Verify worker requirements were respected
        violations = []
        for assignment in all_assignments:
            run_id = assignment["run_id"]
            worker_ids = assignment["worker_ids"]
            expected_workers = scheduler.run_req_workers[run_id]
            if len(worker_ids) != expected_workers:
                violations.append(f"Run {run_id}: expected {expected_workers} workers, got {len(worker_ids)}")

        if not violations:
            print("✓ All worker requirements were respected")
        else:
            print(f"✗ Found {len(violations)} worker requirement violations:")
            for v in violations[:5]:  # Show first 5 violations
                print(f"  {v}")

        # 6. Verify chunk assignments
        chunk_counts = {run_id: {} for run_id in scheduler.run_ids}
        for assignment in all_assignments:
            run_id = assignment["run_id"]
            chunk_id = assignment["chunk_id"]
            chunk_counts[run_id][chunk_id] = chunk_counts[run_id].get(chunk_id, 0) + 1

        chunk_issues = []
        for run_id, chunks in chunk_counts.items():
            for chunk_id in range(scheduler.n_chunks):
                if chunk_id not in chunks:
                    chunk_issues.append(f"Run {run_id} never processed chunk {chunk_id}")
                elif chunks[chunk_id] > 1:
                    chunk_issues.append(f"Run {run_id} processed chunk {chunk_id} {chunks[chunk_id]} times")

        if not chunk_issues:
            print("✓ Each run processed each chunk exactly once")
        else:
            print(f"✗ Found {len(chunk_issues)} chunk assignment issues:")
            for issue in chunk_issues[:5]:  # Show first 5 issues
                print(f"  {issue}")

        # Summary
        print("\n=== Summary ===")
        success = (
            all_completed
            and all_idle
            and len(all_assignments) >= min_assignments
            and len(scheduled_runs) == len(scheduler.run_ids)
            and not violations
            and not chunk_issues
        )

        if success:
            print("✅ All tests passed!")
        else:
            print("❌ Some tests failed - see details above")

        # Convert to assertions for pytest
        assert all_completed, "Not all runs completed all chunks"
        assert all_idle, "Not all workers are idle"
        assert len(all_assignments) >= min_assignments, (
            f"Expected at least {min_assignments} assignments, got {len(all_assignments)}"
        )
        assert len(scheduled_runs) == len(scheduler.run_ids), (
            f"Expected all {len(scheduler.run_ids)} runs to be scheduled, but only {len(scheduled_runs)} were scheduled"
        )
        assert not violations, f"Found worker requirement violations: {violations[:5]}"
        assert not chunk_issues, f"Found chunk assignment issues: {chunk_issues[:5]}"

        print("Large scale scheduling test completed successfully!")

    def test_simple_scheduling_without_monte_carlo(self):
        """Test basic scheduling without Monte Carlo simulation to verify core logic"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
            {"run_id": 3, "req_workers": 1, "estimated_runtime": 12.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=3, num_simulations=0)  # Force fallback logic

        # Track assignments
        assignments = []

        # Test basic scheduling without Monte Carlo
        for i in range(10):  # Try 10 assignments
            result = scheduler.schedule()

            if result["run_id"] is None:  # All completed
                break
            elif result["run_id"] == -1:  # No workers available
                # Complete any running tasks
                for worker_id in range(scheduler.n_workers):
                    if scheduler.state.worker_running_current_run[worker_id] != -1:
                        run_id = scheduler.state.worker_running_current_run[worker_id]
                        scheduler.set_completed_task(run_id)
                        break
                continue

            assignments.append(result)
            scheduler.set_completed_task(result["run_id"])

        print(f"Simple test assignments: {len(assignments)}")
        print(f"Final progress: {scheduler.state.run_visited_num_chunks}")

        # Verify that all runs got some progress
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] > 0, f"Run {run_id} made no progress"

        # Verify that all runs completed all chunks
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] == scheduler.n_chunks

    def test_complete_training_cycle(self):
        """Test a complete training cycle from start to finish"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 15.0},
        ]
        scheduler = Scheduler(
            runs_info, num_workers=2, num_chunks=2, num_simulations=5
        )  # Reduced chunks for faster test

        # Track all assignments
        all_assignments = []
        max_iterations = 10  # Prevent infinite loop
        iteration = 0

        # Simulate complete training
        while iteration < max_iterations:
            result = scheduler.schedule()

            if result["run_id"] is None:  # All runs completed
                break
            elif result["run_id"] == -1:  # No workers available
                # In a real scenario, we'd wait for workers to complete
                # For testing, we'll simulate completion
                for worker_id in range(scheduler.n_workers):
                    if scheduler.state.worker_running_current_run[worker_id] != -1:
                        run_id = scheduler.state.worker_running_current_run[worker_id]
                        scheduler.set_completed_task(run_id)
                        break
                iteration += 1
                continue

            all_assignments.append(result)

            # Simulate task completion
            scheduler.set_completed_task(result["run_id"])
            iteration += 1

        # Verify that all runs completed all chunks
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] == scheduler.n_chunks

        # Verify that all workers are idle
        for worker_id in range(scheduler.n_workers):
            assert scheduler.state.worker_running_current_run[worker_id] == -1

    def test_dynamic_run_management(self):
        """Test adding and removing runs during execution"""
        scheduler = Scheduler([], num_workers=2, num_chunks=5, num_simulations=5)

        # Add runs dynamically
        run1_info = {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0}
        scheduler.add_run(run1_info)

        # Schedule and complete first run
        result = scheduler.schedule()
        assert result["run_id"] == 1
        scheduler.set_completed_task(1)

        # Add second run
        run2_info = {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0}
        scheduler.add_run(run2_info)

        # Schedule second run
        result = scheduler.schedule()
        assert result["run_id"] == 2
        scheduler.set_completed_task(2)

        # Remove first run
        progress = scheduler.remove_run(1)
        assert progress == 1

        # Add third run
        run3_info = {"run_id": 3, "req_workers": 1, "estimated_runtime": 12.0}
        scheduler.add_run(run3_info)

        # Schedule third run
        result = scheduler.schedule()
        assert result["run_id"] == 3

    def test_worker_failure_simulation(self):
        """Test handling worker failures by removing runs"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 8.0},
            {"run_id": 3, "req_workers": 1, "estimated_runtime": 12.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=2, num_chunks=5, num_simulations=5)

        # Start some runs
        result1 = scheduler.schedule()
        result2 = scheduler.schedule()

        # Simulate failure of one run
        failed_run_id = result1["run_id"]
        progress = scheduler.remove_run(failed_run_id)

        # Complete the other run
        scheduler.set_completed_task(result2["run_id"])

        # Should be able to schedule remaining runs
        result3 = scheduler.schedule()
        assert result3["run_id"] != failed_run_id
        assert result3["run_id"] in scheduler.run_ids

    def test_load_balancing_verification(self):
        """Test that the scheduler balances load across runs"""
        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 10.0},
            {"run_id": 3, "req_workers": 1, "estimated_runtime": 10.0},
        ]
        scheduler = Scheduler(runs_info, num_workers=1, num_chunks=10, num_simulations=20)

        # Track which runs get scheduled
        run_schedule_counts = {1: 0, 2: 0, 3: 0}

        # Simulate many scheduling decisions
        for _ in range(30):  # More than total chunks to ensure we see all runs
            result = scheduler.schedule()

            if result["run_id"] is None:  # All completed
                break
            elif result["run_id"] == -1:  # No workers available
                continue

            run_schedule_counts[result["run_id"]] += 1
            scheduler.set_completed_task(result["run_id"])

        # All runs should have been scheduled roughly equally
        # (allowing for some randomness in Monte Carlo)
        for run_id in [1, 2, 3]:
            assert run_schedule_counts[run_id] > 0

    def test_high_worker_utilization_scenario(self):
        """Test scenario with high worker utilization - many runs requiring multiple workers"""
        print("\n=== High Worker Utilization Test ===")

        # Create runs that heavily utilize workers
        runs_info = []
        for i in range(1, 8):  # 7 runs
            req_workers = 2 if i % 2 == 0 else 1  # Alternating 1 and 2 workers
            runs_info.append(
                {
                    "run_id": i,
                    "req_workers": req_workers,
                    "estimated_runtime": 3.0 + (i % 3) * 1.0,  # 3.0, 4.0, 5.0
                    "start_chunk_id": 0,
                }
            )

        scheduler = Scheduler(runs_info, num_workers=4, num_chunks=3, num_simulations=15)

        all_assignments = []
        max_iterations = 100
        iteration = 0
        runs_in_progress = set()

        print(f"Runs: {len(runs_info)}, Workers: {scheduler.n_workers}")
        print(f"Worker requirements: {[run['req_workers'] for run in runs_info]}")

        while iteration < max_iterations:
            result = scheduler.schedule()

            if result["run_id"] is None:
                print(f"✓ All runs completed after {iteration} iterations")
                break
            elif result["run_id"] == -1:
                if runs_in_progress:
                    run_to_complete = next(iter(runs_in_progress))
                    scheduler.set_completed_task(run_to_complete)
                    runs_in_progress.remove(run_to_complete)
                else:
                    break
            else:
                all_assignments.append(result)
                runs_in_progress.add(result["run_id"])

                # Complete some runs periodically
                if iteration % 4 == 3 and runs_in_progress:
                    num_to_complete = min(len(runs_in_progress), 2)
                    runs_to_complete = list(runs_in_progress)[:num_to_complete]
                    for run_id in runs_to_complete:
                        scheduler.set_completed_task(run_id)
                        runs_in_progress.remove(run_id)

            iteration += 1

        # Complete remaining runs
        while runs_in_progress:
            run_id = runs_in_progress.pop()
            scheduler.set_completed_task(run_id)

        # Verify all runs completed
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] == scheduler.n_chunks

        print(f"✓ High utilization test completed: {len(all_assignments)} assignments")

    def test_mixed_worker_requirements_scenario(self):
        """Test scenario with mixed worker requirements - some runs need 1, 2, or 3 workers"""
        print("\n=== Mixed Worker Requirements Test ===")

        runs_info = [
            {"run_id": 1, "req_workers": 1, "estimated_runtime": 4.0, "start_chunk_id": 0},
            {"run_id": 2, "req_workers": 2, "estimated_runtime": 6.0, "start_chunk_id": 1},
            {"run_id": 3, "req_workers": 3, "estimated_runtime": 8.0, "start_chunk_id": 2},
            {"run_id": 4, "req_workers": 1, "estimated_runtime": 3.0, "start_chunk_id": 0},
            {"run_id": 5, "req_workers": 2, "estimated_runtime": 5.0, "start_chunk_id": 1},
            {"run_id": 6, "req_workers": 1, "estimated_runtime": 4.0, "start_chunk_id": 2},
        ]

        scheduler = Scheduler(runs_info, num_workers=4, num_chunks=4, num_simulations=20)

        all_assignments = []
        max_iterations = 150
        iteration = 0
        runs_in_progress = set()

        print(f"Runs: {len(runs_info)}, Workers: {scheduler.n_workers}")
        print(f"Worker requirements: {[run['req_workers'] for run in runs_info]}")

        while iteration < max_iterations:
            result = scheduler.schedule()

            if result["run_id"] is None:
                print(f"✓ All runs completed after {iteration} iterations")
                break
            elif result["run_id"] == -1:
                if runs_in_progress:
                    run_to_complete = next(iter(runs_in_progress))
                    scheduler.set_completed_task(run_to_complete)
                    runs_in_progress.remove(run_to_complete)
                else:
                    break
            else:
                all_assignments.append(result)
                runs_in_progress.add(result["run_id"])

                # Complete runs at different rates based on worker requirements
                if iteration % 5 == 4 and runs_in_progress:
                    # Complete 1-3 runs
                    num_to_complete = min(len(runs_in_progress), 3)
                    runs_to_complete = list(runs_in_progress)[:num_to_complete]
                    for run_id in runs_to_complete:
                        scheduler.set_completed_task(run_id)
                        runs_in_progress.remove(run_id)

            iteration += 1

        # Complete remaining runs
        while runs_in_progress:
            run_id = runs_in_progress.pop()
            scheduler.set_completed_task(run_id)

        # Verify all runs completed
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] == scheduler.n_chunks

        # Verify worker requirements were respected
        for assignment in all_assignments:
            run_id = assignment["run_id"]
            worker_ids = assignment["worker_ids"]
            expected_workers = scheduler.run_req_workers[run_id]
            assert len(worker_ids) == expected_workers

        print(f"✓ Mixed requirements test completed: {len(all_assignments)} assignments")

    def test_stress_test_scenario(self):
        """Stress test with many runs and high chunk counts"""
        print("\n=== Stress Test ===")

        # Create many runs with varying requirements
        runs_info = []
        for i in range(1, 16):  # 15 runs
            req_workers = ((i - 1) % 4) + 1  # 1, 2, 3, 4, 1, 2, 3, 4, ...
            runs_info.append(
                {
                    "run_id": i,
                    "req_workers": req_workers,
                    "estimated_runtime": 2.0 + (i % 4) * 0.5,  # 2.0, 2.5, 3.0, 3.5
                    "start_chunk_id": i % 3,
                }
            )

        scheduler = Scheduler(runs_info, num_workers=6, num_chunks=6, num_simulations=25)

        all_assignments = []
        max_iterations = 300
        iteration = 0
        runs_in_progress = set()

        print(f"Runs: {len(runs_info)}, Workers: {scheduler.n_workers}, Chunks: {scheduler.n_chunks}")
        print(f"Worker requirements: {[run['req_workers'] for run in runs_info]}")

        while iteration < max_iterations:
            result = scheduler.schedule()

            if result["run_id"] is None:
                print(f"✓ All runs completed after {iteration} iterations")
                break
            elif result["run_id"] == -1:
                if runs_in_progress:
                    run_to_complete = next(iter(runs_in_progress))
                    scheduler.set_completed_task(run_to_complete)
                    runs_in_progress.remove(run_to_complete)
                else:
                    break
            else:
                all_assignments.append(result)
                runs_in_progress.add(result["run_id"])

                # Complete runs more frequently in stress test
                if iteration % 3 == 2 and runs_in_progress:
                    num_to_complete = min(len(runs_in_progress), 3)
                    runs_to_complete = list(runs_in_progress)[:num_to_complete]
                    for run_id in runs_to_complete:
                        scheduler.set_completed_task(run_id)
                        runs_in_progress.remove(run_id)

            iteration += 1

        # Complete remaining runs
        while runs_in_progress:
            run_id = runs_in_progress.pop()
            scheduler.set_completed_task(run_id)

        # Verify all runs completed
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] == scheduler.n_chunks

        # Verify minimum assignments (15 runs * 6 chunks = 90 minimum)
        min_assignments = len(scheduler.run_ids) * scheduler.n_chunks
        assert len(all_assignments) >= min_assignments

        print(f"✓ Stress test completed: {len(all_assignments)} assignments")

    def test_edge_case_worker_allocation(self):
        """Test edge cases in worker allocation"""
        print("\n=== Edge Case Worker Allocation Test ===")

        # Test case where total worker requirements exactly match available workers
        runs_info = [
            {"run_id": 1, "req_workers": 2, "estimated_runtime": 5.0, "start_chunk_id": 0},
            {"run_id": 2, "req_workers": 1, "estimated_runtime": 3.0, "start_chunk_id": 0},
            {"run_id": 3, "req_workers": 1, "estimated_runtime": 4.0, "start_chunk_id": 0},
        ]

        scheduler = Scheduler(runs_info, num_workers=4, num_chunks=3, num_simulations=10)

        all_assignments = []
        max_iterations = 50
        iteration = 0
        runs_in_progress = set()

        print(f"Runs: {len(runs_info)}, Workers: {scheduler.n_workers}")
        print(f"Worker requirements: {[run['req_workers'] for run in runs_info]}")

        while iteration < max_iterations:
            result = scheduler.schedule()

            if result["run_id"] is None:
                print(f"✓ All runs completed after {iteration} iterations")
                break
            elif result["run_id"] == -1:
                if runs_in_progress:
                    run_to_complete = next(iter(runs_in_progress))
                    scheduler.set_completed_task(run_to_complete)
                    runs_in_progress.remove(run_to_complete)
                else:
                    break
            else:
                all_assignments.append(result)
                runs_in_progress.add(result["run_id"])

                # Complete runs after short intervals
                if iteration % 2 == 1 and runs_in_progress:
                    run_to_complete = next(iter(runs_in_progress))
                    scheduler.set_completed_task(run_to_complete)
                    runs_in_progress.remove(run_to_complete)

            iteration += 1

        # Complete remaining runs
        while runs_in_progress:
            run_id = runs_in_progress.pop()
            scheduler.set_completed_task(run_id)

        # Verify all runs completed
        for run_id in scheduler.run_ids:
            assert scheduler.state.run_visited_num_chunks[run_id] == scheduler.n_chunks

        print(f"✓ Edge case test completed: {len(all_assignments)} assignments")


if __name__ == "__main__":
    pytest.main([__file__])
