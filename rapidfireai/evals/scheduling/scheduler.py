from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import ray


class SchedulingStrategy(ABC):
    """Abstract base class for scheduling strategies."""

    @abstractmethod
    def select_actor(self, actors: list[Any], task_index: int, task_data: Any) -> int:
        """
        Select which actor should handle the task.

        Args:
            actors: List of available actors
            task_index: Index of the current task
            task_data: Data for the current task

        Returns:
            Index of the selected actor
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the scheduling strategy."""
        pass


class RoundRobinStrategy(SchedulingStrategy):
    """Round-robin scheduling strategy - cycles through actors sequentially."""

    def select_actor(self, actors: list[Any], task_index: int, task_data: Any) -> int:
        """Select actor using round-robin (modulo) assignment."""
        return task_index % len(actors)

    def get_name(self) -> str:
        return "round_robin"


class LoadBalancedStrategy(SchedulingStrategy):
    """Load-balanced scheduling strategy - assigns to least busy actor."""

    def __init__(self):
        self.actor_loads = {}

    def select_actor(self, actors: list[Any], task_index: int, task_data: Any) -> int:
        """Select actor with minimum current load."""
        # Initialize loads if first time
        if not self.actor_loads:
            self.actor_loads = dict.fromkeys(range(len(actors)), 0)

        # Find actor with minimum load
        min_load_actor = min(self.actor_loads.keys(), key=lambda x: self.actor_loads[x])

        # Increment load for selected actor
        self.actor_loads[min_load_actor] += 1

        return min_load_actor

    def get_name(self) -> str:
        return "load_balanced"


class Scheduler:
    """
    Handles job scheduling and distribution across actors with pluggable strategies.
    Supports various scheduling algorithms for optimal load distribution.
    """

    def __init__(self, strategy: str = "round_robin"):
        """
        Initialize the scheduler with a specific strategy.

        Args:
            strategy: Scheduling strategy ('round_robin', 'load_balanced')
        """
        self.strategy = self._create_strategy(strategy)
        self.submitted_jobs = 0
        self.actor_assignments = {}

    def _create_strategy(self, strategy_name: str) -> SchedulingStrategy:
        """Create and return the appropriate scheduling strategy."""
        strategies = {
            "round_robin": RoundRobinStrategy(),
            "load_balanced": LoadBalancedStrategy(),
        }

        if strategy_name not in strategies:
            raise ValueError(f"Unknown scheduling strategy: {strategy_name}. Available: {list(strategies.keys())}")

        return strategies[strategy_name]

    def submit_jobs(
        self,
        actors: list[Any],
        tasks: list[Any],
        task_method: str = "process_batch",
        preprocess_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        compute_metrics_fn: Callable | None = None,
    ) -> list[ray.ObjectRef]:
        """
        Submit jobs to actors using the configured scheduling strategy.

        Args:
            actors: List of Ray actors to distribute work to
            tasks: List of tasks/data chunks to process
            task_method: Method name to call on actors (default: "process_batch")
            preprocess_fn: Optional preprocessing function
            postprocess_fn: Optional postprocessing function
            compute_metrics_fn: Optional metrics computation function
            config: Optional configuration dictionary
        Returns:
            List of Ray futures for submitted jobs
        """
        futures = []
        self.actor_assignments = {i: [] for i in range(len(actors))}

        for i, task_data in enumerate(tasks):
            # Use strategy to select which actor should handle this task
            actor_idx = self.strategy.select_actor(actors, i, task_data)

            # Get the method to call on the selected actor
            actor_method = getattr(actors[actor_idx], task_method)

            # Submit the job
            future = actor_method.remote(task_data, preprocess_fn, postprocess_fn, compute_metrics_fn)

            futures.append(future)
            self.actor_assignments[actor_idx].append(i)

        self.submitted_jobs = len(futures)

        # Print scheduling summary
        # self._print_scheduling_summary(actors, tasks)

        return futures

    def _print_scheduling_summary(self, actors: list[Any], tasks: list[Any]) -> None:
        """Print a summary of the scheduling results."""
        print(f"Submitted {len(tasks)} batches to {len(actors)} actors using {self.strategy.get_name()} scheduling")

        # Show distribution across actors
        if len(actors) <= 10:  # Only show detailed breakdown for reasonable number of actors
            for actor_idx, task_indices in self.actor_assignments.items():
                print(f"  Actor {actor_idx}: {len(task_indices)} tasks")

    def get_scheduling_stats(self) -> dict[str, Any]:
        """
        Get statistics about the current scheduling state.

        Returns:
            Dictionary with scheduling statistics
        """
        if not self.actor_assignments:
            return {"status": "no_jobs_submitted"}

        task_counts = [len(tasks) for tasks in self.actor_assignments.values()]

        return {
            "strategy": self.strategy.get_name(),
            "total_jobs": self.submitted_jobs,
            "num_actors": len(self.actor_assignments),
            "tasks_per_actor": {
                "min": min(task_counts),
                "max": max(task_counts),
                "avg": sum(task_counts) / len(task_counts),
                "distribution": task_counts,
            },
            "load_balance_ratio": min(task_counts) / max(task_counts) if max(task_counts) > 0 else 1.0,
        }

    def change_strategy(self, strategy_name: str) -> None:
        """
        Change the scheduling strategy.

        Args:
            strategy_name: New strategy to use
        """
        old_strategy = self.strategy.get_name()
        self.strategy = self._create_strategy(strategy_name)
        print(f"Scheduling strategy changed from {old_strategy} to {strategy_name}")

    def get_available_strategies(self) -> list[str]:
        """Get list of available scheduling strategies."""
        return ["round_robin", "load_balanced"]
