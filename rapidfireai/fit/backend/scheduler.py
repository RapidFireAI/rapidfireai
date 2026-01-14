"""This module contains the Scheduler class which is responsible for scheduling runs on workers to train on a chunk."""


class Scheduler:
    """This class is responsible for scheduling runs on to workers to train on a chunk"""

    def __init__(self, run_ids: list[int], num_workers: int, num_chunks: int) -> None:
        # run_ids are 1 indexed
        # worker_ids are 0 indexed
        # chunk_ids are 0 indexed

        self.n_runs: int = len(run_ids)
        self.n_workers: int = num_workers
        self.n_chunks: int = num_chunks
        self.run_ids: list[int] = run_ids

        # create data structures
        self.worker_running_current_run: dict[int, int] = dict.fromkeys(range(self.n_workers), -1)
        self.run_visited_num_chunks: dict[int, int] = dict.fromkeys(self.run_ids, 0)
        # Track epochs completed per run for fair scheduling across epoch boundaries
        self.run_epochs_completed: dict[int, int] = dict.fromkeys(self.run_ids, 0)

        # add runs to scheduler
        for run_id in run_ids:
            self.add_run(run_id, 0)

    def reset_run(self, run_id: int) -> None:
        """Reset the scheduler for a specific run (used at epoch boundaries)"""
        if run_id in self.run_ids:
            # Increment epoch counter before resetting chunk progress
            # This ensures fair scheduling: runs that completed more epochs have lower priority
            self.run_epochs_completed[run_id] = self.run_epochs_completed.get(run_id, 0) + 1

            # Reset chunk progress for this run
            self.run_visited_num_chunks[run_id] = 0

            # If this run is currently assigned to a worker, free the worker
            for worker_id in range(self.n_workers):
                if self.worker_running_current_run[worker_id] == run_id:
                    self.worker_running_current_run[worker_id] = -1

    def add_run(self, run_id: int, run_visited_num_chunks: int, run_epochs_completed: int = 0) -> None:
        """Add a new run to the scheduler."""
        if run_id not in self.run_ids:
            self.run_ids.append(run_id)
            self.n_runs = len(self.run_ids)

        self.run_visited_num_chunks[run_id] = run_visited_num_chunks
        self.run_epochs_completed[run_id] = run_epochs_completed

    def set_completed_task(self, worker_id: int) -> None:
        """Set a task as completed."""
        run_id = self.worker_running_current_run[worker_id]

        if run_id != -1:
            self.worker_running_current_run[worker_id] = -1
            self.run_visited_num_chunks[run_id] += 1

    def remove_run(self, run_id: int) -> int:
        """Remove a run from the scheduler and return its progress."""
        if run_id not in self.run_ids:
            return 0

        # Get the progress before removing
        progress = self.run_visited_num_chunks.get(run_id, 0)

        # Clean up worker assignment
        for worker_id in range(self.n_workers):
            if self.worker_running_current_run[worker_id] == run_id:
                self.worker_running_current_run[worker_id] = -1

        # Remove from all data structures
        self.run_visited_num_chunks.pop(run_id, None)
        self.run_epochs_completed.pop(run_id, None)

        if run_id in self.run_ids:
            self.run_ids.remove(run_id)
            self.n_runs = len(self.run_ids)

        return progress

    def schedule(self) -> dict[str, int | bool | None] | None:
        """
        Schedule a single task based on constraints and preferences.
        Returns {run_id: <>, worker_id: <>, chunk_id: <>, is_last_chunk: <>} if a schedule is possible.
        Returns {run_id: None, worker_id: None, chunk_id: None, is_last_chunk: None} if all runs have seen all chunks.
        Returns {run_id: -1, worker_id: -1, chunk_id: -1, is_last_chunk: None} if all workers are busy or no runs are available.
        """
        # First check if all workers are busy (most common condition)
        available_workers = [
            worker_id for worker_id in range(self.n_workers) if self.worker_running_current_run[worker_id] == -1
        ]
        if not available_workers:
            return {"run_id": -1, "worker_id": -1, "chunk_id": -1, "is_last_chunk": None}

        # Next check if all runs have seen all chunks (termination condition)
        if all(self.run_visited_num_chunks[run_id] >= self.n_chunks for run_id in self.run_ids):
            return {"run_id": None, "worker_id": None, "chunk_id": None, "is_last_chunk": None}

        # Get busy runs and available runs
        busy_runs = {run_id for run_id in self.worker_running_current_run.values() if run_id != -1}
        available_runs = [
            run_id
            for run_id in self.run_ids
            if self.run_visited_num_chunks[run_id] < self.n_chunks and run_id not in busy_runs
        ]

        # If no available runs, return busy state
        if not available_runs:
            return {"run_id": -1, "worker_id": -1, "chunk_id": -1, "is_last_chunk": None}

        # Find the run with least progress using epoch-aware priority:
        # 1. First: fewest epochs completed (ensures runs complete current epoch before others start new ones)
        # 2. Then: fewest chunks in current epoch (fair round-robin within an epoch)
        # 3. Finally: lowest run_id for tie-breaking
        # NOTE: newly inserted clones will have 0 epochs, so they get priority

        run_id = min(available_runs, key=lambda r: (
            self.run_epochs_completed.get(r, 0),
            self.run_visited_num_chunks[r],
            r
        ))
        worker_id = available_workers[0]  # Pick first available worker
        chunk_id = self.run_visited_num_chunks[run_id] % self.n_chunks  # Next chunk in sequence starting from 0
        is_last_chunk = chunk_id == self.n_chunks - 1

        # Update internal state immediately
        self.worker_running_current_run[worker_id] = run_id

        return {"run_id": run_id, "worker_id": worker_id, "chunk_id": chunk_id, "is_last_chunk": is_last_chunk}

    def get_status(self) -> dict:
        """Get current scheduler status for debugging."""
        completed_runs = [run_id for run_id in self.run_ids if self.run_visited_num_chunks[run_id] == self.n_chunks]

        return {
            "active_runs": len([r for r in self.run_ids if self.run_visited_num_chunks[r] < self.n_chunks]),
            "busy_workers": len([w for w in range(self.n_workers) if self.worker_running_current_run[w] != -1]),
            "completed_runs": len(completed_runs),
            "worker_assignments": {
                w: self.worker_running_current_run[w]
                for w in range(self.n_workers)
                if self.worker_running_current_run[w] != -1
            },
            "run_progress": {
                r: f"epoch {self.run_epochs_completed.get(r, 0)}, chunk {self.run_visited_num_chunks[r]}/{self.n_chunks}"
                for r in self.run_ids
            },
        }
