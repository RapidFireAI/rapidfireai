"""Base classes and configurations for AutoML algorithms."""

from abc import ABC, abstractmethod
from typing import Any

from rapidfireai.evals.automl.datatypes import List


class AutoMLAlgorithm(ABC):
    """Base class for AutoML algorithms."""

    def __init__(self, configs=None, num_runs: int = 1):
        """Initialize AutoML algorithm with configurations and trainer type."""
        try:
            self.configs = self._normalize_configs(configs)
            self.num_runs = num_runs

            # self._validate_configs()
        except Exception as e:
            raise Exception(f"Error initializing {self.__class__.__name__}: {e}") from e

    def _normalize_configs(self, configs):
        """Normalize configs to list format."""
        if isinstance(configs, List):
            return configs.values
        elif isinstance(configs, list):
            return configs
        return [configs] if configs else []

    @abstractmethod
    def get_runs(self, seed: int) -> list[dict[str, Any]]:
        """Generate hyperparameter combinations for different training configurations."""
        if not isinstance(seed, int) or seed < 0:
            raise Exception("seed must be a non-negative integer")
