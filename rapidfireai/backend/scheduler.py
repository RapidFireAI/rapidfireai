"""This module contains the Scheduler class which is responsible for scheduling runs on workers to train on a chunk."""

import copy
import random

from rapidfireai.utils.constants import NUM_MONTE_CARLO_SIMULATIONS


class SchedulerState:
    """A lightweight state for simulation that can be easily copied and modified."""

    def __init__(self):
        self.worker_running_current_run: dict[int, int] = {}
        self.run_visited_num_chunks: dict[int, int] = {}
        self.run_assigned_workers: dict[int, set[int]] = {}
        self.run_completion_time: dict[int, float] = {}
        self.current_time: float = 0.0

    def copy(self):
        """Create a deep copy of this state."""
        new_state = SchedulerState()
        new_state.worker_running_current_run = copy.deepcopy(self.worker_running_current_run)
        new_state.run_visited_num_chunks = copy.deepcopy(self.run_visited_num_chunks)
        new_state.run_assigned_workers = copy.deepcopy(self.run_assigned_workers)
        new_state.run_completion_time = copy.deepcopy(self.run_completion_time)
        new_state.current_time = self.current_time
        return new_state


class Scheduler:
    """This class is responsible for scheduling runs on to workers to train on a chunk"""

    def __init__(self, runs_info: list[dict], num_workers: int, num_chunks: int, num_simulations: int = 100) -> None:
        self.n_workers: int = num_workers
        self.n_chunks: int = num_chunks
        self.num_simulations: int = num_simulations

        # create data structures
        self.worker_running_current_run: dict[int, int] = dict.fromkeys(range(self.n_workers), -1)
        self.run_visited_num_chunks: dict[int, int] = dict.fromkeys(self.run_ids, 0)

        # Store run parameters
        self.run_req_workers: dict[int, int] = {}
        self.run_estimated_runtime: dict[int, float] = {}
        self.run_start_chunk_id: dict[int, int] = {}

        for run in runs_info:
            run_id = run["run_id"]
            req_workers = run.get("req_workers", 1)
            if req_workers > self.n_workers:
                raise ValueError(
                    f"Run {run_id} requires {req_workers} workers but only {self.n_workers} workers are available"
                )
            self.run_req_workers[run_id] = req_workers
            self.run_estimated_runtime[run_id] = run.get("estimated_runtime", 1.0)
            self.run_start_chunk_id[run_id] = run.get("start_chunk_id", 0)

        # Initialize actual state - NO completion times needed for actual state!
        self.state = SchedulerState()
        self.state.worker_running_current_run = dict.fromkeys(range(self.n_workers), -1)
        self.state.run_visited_num_chunks = dict.fromkeys(self.run_ids, 0)
        self.state.run_assigned_workers = {run_id: set() for run_id in self.run_ids}

    def reset_run(self, run_id: int) -> None:
        """Reset the scheduler for a specific run (used at epoch boundaries)"""
        if run_id not in self.run_ids:
            return

        # Reset progress for this run
        self.state.run_visited_num_chunks[run_id] = 0

    def add_run(self, run_id: int, run_visited_num_chunks: int) -> None:
        """Add a new run to the scheduler."""
        run_id = run_info["run_id"]

        # Add to run_ids if not already present
        if run_id not in self.run_ids:
            self.run_ids.append(run_id)
            self.n_runs = len(self.run_ids)
            self.state.run_assigned_workers[run_id] = set()

        self.run_visited_num_chunks[run_id] = run_visited_num_chunks

        # Set the progress
        self.state.run_visited_num_chunks[run_id] = run_visited_num_chunks

    def remove_run(self, run_id: int) -> int:
        """Remove a run from the scheduler and return its progress."""
        if run_id not in self.run_ids:
            return 0

        # Get the progress before removing
        progress = self.state.run_visited_num_chunks.get(run_id, 0)

        # Free all workers assigned to this run
        workers_to_free = list(self.state.run_assigned_workers.get(run_id, set()))
        for worker_id in workers_to_free:
            if self.state.worker_running_current_run[worker_id] == run_id:
                self.state.worker_running_current_run[worker_id] = -1

        # Remove from all data structures
        self.run_visited_num_chunks.pop(run_id, None)

        # Remove from run_ids
        if run_id in self.run_ids:
            self.run_ids.remove(run_id)
            self.n_runs = len(self.run_ids)

        return progress

    def set_completed_task(self, run_id: int, state: SchedulerState = None) -> None:
        """Set a task as completed for a given run."""
        if state is None:
            state = self.state

        if run_id not in self.run_ids:
            return

        # Free all workers assigned to this run
        for worker_id in list(state.run_assigned_workers[run_id]):
            if state.worker_running_current_run[worker_id] == run_id:
                state.worker_running_current_run[worker_id] = -1

        # Clear the run's assigned workers
        state.run_assigned_workers[run_id].clear()

        # Increment progress for this run
        state.run_visited_num_chunks[run_id] += 1

        # Clear completion time (only relevant in simulation)
        if run_id in state.run_completion_time:
            del state.run_completion_time[run_id]

    def _get_available_workers(self, sim_state: SchedulerState) -> list[int]:
        """Get list of available workers in given state."""
        available_workers = []
        for w in range(self.n_workers):
            if sim_state.worker_running_current_run[w] == -1:
                available_workers.append(w)
        return available_workers

    def _get_schedulable_runs(self, sim_state: SchedulerState) -> list[int]:
        """Get runs that can be scheduled"""
        schedulable = []

        # Find the minimum chunk progress among all incomplete runs
        min_chunks_visited = float("inf")
        for run_id in self.run_ids:
            if sim_state.run_visited_num_chunks[run_id] < self.n_chunks:
                min_chunks_visited = min(min_chunks_visited, sim_state.run_visited_num_chunks[run_id])

        # Only allow scheduling runs that are at the minimum level
        for run_id in self.run_ids:
            # Skip completed runs
            if sim_state.run_visited_num_chunks[run_id] >= self.n_chunks:
                continue
            # Skip currently running runs
            if len(sim_state.run_assigned_workers[run_id]) > 0:
                continue
            # Only include runs at the minimum chunk level
            if sim_state.run_visited_num_chunks[run_id] == min_chunks_visited:
                schedulable.append(run_id)

        return schedulable

    def _get_next_chunk_id(self, sim_state: SchedulerState, run_id: int) -> int:
        """Get the next chunk ID for a run in given state."""
        return (sim_state.run_visited_num_chunks[run_id] + self.run_start_chunk_id[run_id]) % self.n_chunks

    def _simulate_random_schedule(self, start_state: SchedulerState, seed: int = None) -> tuple[float, list[dict]]:
        """Run a simulation from a given state."""
        if seed is not None:
            random.seed(seed)

        # Work with a copy - simulation DOES care about time
        sim_state = start_state.copy()
        schedule_sequence = []
        max_iterations = NUM_MONTE_CARLO_SIMULATIONS

        iteration = 0
        while iteration < max_iterations:
            iteration += 1

            # Check termination
            if all(sim_state.run_visited_num_chunks[run_id] >= self.n_chunks for run_id in self.run_ids):
                break

            # Complete any runs that have finished at current time
            runs_to_complete = [
                run_id for run_id, t in sim_state.run_completion_time.items() if t <= sim_state.current_time
            ]
            for run_id in runs_to_complete:
                self.set_completed_task(run_id, sim_state)

            # Get available workers and schedulable runs
            available_workers = self._get_available_workers(sim_state)
            schedulable_runs = self._get_schedulable_runs(sim_state)

            # If we can't schedule anything, advance time
            if not available_workers or not schedulable_runs:
                if sim_state.run_completion_time:
                    # Advance to the earliest completion time
                    sim_state.current_time = min(sim_state.run_completion_time.values())
                    continue
                else:
                    # No runs in progress and can't schedule anything - we're done
                    break

            # Try to schedule runs randomly
            shuffled_runs = random.sample(schedulable_runs, len(schedulable_runs))
            scheduled_any = False

            for run_id in shuffled_runs:
                req_workers = self.run_req_workers[run_id]
                available_workers = self._get_available_workers(sim_state)  # Refresh

                if req_workers <= len(available_workers):
                    chunk_id = self._get_next_chunk_id(sim_state, run_id)
                    assigned_workers = available_workers[:req_workers]
                    runtime = self.run_estimated_runtime[run_id]

                    # Update simulation state
                    for worker_id in assigned_workers:
                        sim_state.worker_running_current_run[worker_id] = run_id
                        sim_state.run_assigned_workers[run_id].add(worker_id)

                    sim_state.run_completion_time[run_id] = sim_state.current_time + runtime

                    # Record this scheduling decision
                    schedule_sequence.append(
                        {
                            "run_id": run_id,
                            "worker_ids": tuple(assigned_workers),
                            "chunk_id": chunk_id,
                            "start_time": sim_state.current_time,
                            "end_time": sim_state.current_time + runtime,
                        }
                    )
                    scheduled_any = True
                    # Continue to try scheduling more runs in this time slot

            # If we scheduled something but there are still workers/runs available,
            # stay at current time to try scheduling more
            # Otherwise, if nothing was scheduled but we have runs in progress, advance time
            if not scheduled_any and sim_state.run_completion_time:
                sim_state.current_time = min(sim_state.run_completion_time.values())

        # Calculate makespan
        makespan = max(task["end_time"] for task in schedule_sequence) if schedule_sequence else 0.0

        return makespan, schedule_sequence

    def schedule(self) -> dict[str, int | tuple | bool | None] | None:
        """
        Schedule tasks using Monte Carlo simulation to find the best schedule.
        The actual scheduler doesn't care about time - that's managed externally.
        """
        # Check if all runs have seen all chunks first
        if all(self.state.run_visited_num_chunks[run_id] >= self.n_chunks for run_id in self.run_ids):
            return {"run_id": None, "worker_ids": None, "chunk_id": None, "is_last_chunk": None}

        # Check basic conditions using actual state
        available_workers = self._get_available_workers(self.state)
        if not available_workers:
            return {"run_id": -1, "worker_ids": None, "chunk_id": -1, "is_last_chunk": None}

        schedulable_runs = self._get_schedulable_runs(self.state)
        if not schedulable_runs:
            return {"run_id": -1, "worker_ids": None, "chunk_id": -1, "is_last_chunk": None}

        # Get busy runs and available runs
        busy_runs = {run_id for run_id in self.worker_running_current_run.values() if run_id != -1}
        available_runs = [
            run_id
            for run_id in self.run_ids
            if self.run_visited_num_chunks[run_id] < self.n_chunks and run_id not in busy_runs
        ]

        for sim in range(self.num_simulations):
            makespan, schedule_sequence = self._simulate_random_schedule(self.state, seed=sim)

            if schedule_sequence and makespan < best_makespan:
                best_makespan = makespan
                # Find the first action that hasn't been done yet
                # Since we're simulating from current state, the first action should be valid
                for action in schedule_sequence:
                    # Check if this action is still valid in current state
                    run_id = action["run_id"]
                    if run_id in schedulable_runs and len(self.state.run_assigned_workers[run_id]) == 0:
                        best_first_action = action
                        break

        run_id = min(available_runs, key=lambda run_id: (self.run_visited_num_chunks[run_id], run_id))
        worker_id = available_workers[0]  # Pick first available worker
        chunk_id = self.run_visited_num_chunks[run_id] % self.n_chunks  # Next chunk in sequence starting from 0
        is_last_chunk = chunk_id == self.n_chunks - 1

        if req_workers <= len(available_workers):
            chunk_id = self._get_next_chunk_id(self.state, run_id)
            best_first_action = {
                "run_id": run_id,
                "worker_ids": tuple(available_workers[:req_workers]),
                "chunk_id": chunk_id,
            }

        if best_first_action:
            # Apply the best action to actual state
            run_id = best_first_action["run_id"]
            # Make sure we use currently available workers, not the ones from simulation
            req_workers = self.run_req_workers[run_id]
            current_available = self._get_available_workers(self.state)

            if req_workers <= len(current_available):
                worker_ids = tuple(current_available[:req_workers])
                chunk_id = self._get_next_chunk_id(self.state, run_id)

                # Update actual state
                for worker_id in worker_ids:
                    self.state.worker_running_current_run[worker_id] = run_id
                    self.state.run_assigned_workers[run_id].add(worker_id)

                is_last_chunk = chunk_id == self.n_chunks - 1

                return {
                    "run_id": run_id,
                    "worker_ids": worker_ids,
                    "chunk_id": chunk_id,
                    "is_last_chunk": is_last_chunk,
                }

        return {"run_id": -1, "worker_ids": None, "chunk_id": -1, "is_last_chunk": None}

    def get_status(self) -> dict:
        """Get current scheduler status for debugging."""
        completed_runs = [
            run_id for run_id in self.run_ids if self.state.run_visited_num_chunks[run_id] == self.n_chunks
        ]

        current_chunks = {}
        for run_id in self.run_ids:
            if len(self.state.run_assigned_workers[run_id]) > 0:
                current_chunks[run_id] = self._get_next_chunk_id(self.state, run_id)
            else:
                current_chunks[run_id] = "None"

        return {
            "active_runs": len([r for r in self.run_ids if self.state.run_visited_num_chunks[r] < self.n_chunks]),
            "busy_workers": len([w for w in range(self.n_workers) if self.state.worker_running_current_run[w] != -1]),
            "completed_runs": len(completed_runs),
            "run_progress": {r: f"{self.state.run_visited_num_chunks[r]}/{self.n_chunks}" for r in self.run_ids},
            "current_chunks": current_chunks,
            "run_workers": {
                r: f"{len(self.state.run_assigned_workers[r])}/{self.run_req_workers[r]}" for r in self.run_ids
            },
        }
