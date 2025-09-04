"""Base classes and configurations for AutoML algorithms."""

from abc import ABC, abstractmethod
from typing import Any

from rapidfireai.automl.datatypes import List
from rapidfireai.automl.model_config import RFModelConfig
from rapidfireai.utils.exceptions import AutoMLException


class AutoMLAlgorithm(ABC):
    """Base class for AutoML algorithms."""

    VALID_TRAINER_TYPES = {"SFT", "DPO", "GRPO"}

    def __init__(self, configs=None, create_model_fn=None, trainer_type: str = "SFT", num_runs: int = 1):
        """Initialize AutoML algorithm with configurations and trainer type."""
        try:
            self.configs = self._normalize_configs(configs)
            self.trainer_type = trainer_type.upper()
            self.num_runs = num_runs

            if self.trainer_type not in self.VALID_TRAINER_TYPES:
                raise AutoMLException(f"trainer_type must be one of {self.VALID_TRAINER_TYPES}")

            self._validate_configs()
        except Exception as e:
            raise AutoMLException(f"Error initializing {self.__class__.__name__}: {e}") from e

    def _normalize_configs(self, configs):
        """Normalize configs to list format."""
        if isinstance(configs, List):
            return configs.values
        elif isinstance(configs, list):
            return configs
        return [configs] if configs else []

    def _validate_configs(self):
        """Validate all configs are RFModelConfig instances."""
        for config in self.configs:
            if not isinstance(config, RFModelConfig):
                raise AutoMLException(f"All configs must be RFModelConfig instances, got {type(config)}")

    @abstractmethod
    def get_runs(self, seed: int) -> list[dict[str, Any]]:
        """Generate hyperparameter combinations for different training configurations."""
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")
