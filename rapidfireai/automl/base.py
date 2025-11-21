"""Base classes and configurations for AutoML algorithms."""

from abc import ABC, abstractmethod
from typing import Any

from rapidfireai.automl.datatypes import List
from rapidfireai.fit.utils.exceptions import AutoMLException


class AutoMLAlgorithm(ABC):
    """Base class for AutoML algorithms."""

    VALID_TRAINER_TYPES = {"SFT", "DPO", "GRPO"}

    def __init__(self, configs=None, create_model_fn=None, trainer_type: str | None = None, num_runs: int = 1):
        """
        Initialize AutoML algorithm with configurations and trainer type.
        
        Args:
            configs: List of configurations (RFModelConfig for fit mode, dict for evals mode)
            create_model_fn: Optional function to create models (legacy parameter)
            trainer_type: Trainer type ("SFT", "DPO", "GRPO") for fit mode, None for evals mode
            num_runs: Number of runs for random search
        
        Mode detection:
            - If trainer_type is provided: fit mode (requires RFModelConfig instances)
            - If trainer_type is None: evals mode (requires dict instances)
        """
        try:
            self.configs = self._normalize_configs(configs)
            self.num_runs = num_runs
            
            # Detect mode based on trainer_type
            if trainer_type is not None:
                self.mode = "fit"
                self.trainer_type = trainer_type.upper()
                if self.trainer_type not in self.VALID_TRAINER_TYPES:
                    raise AutoMLException(f"trainer_type must be one of {self.VALID_TRAINER_TYPES}")
            else:
                self.mode = "evals"
                self.trainer_type = None

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
        """Validate configs based on mode."""
        if not self.configs:
            return
            
        # Import here to avoid circular imports
        from rapidfireai.automl.model_config import RFModelConfig
        
        if self.mode == "fit":
            # Fit mode: must have RFModelConfig instances
            for config in self.configs:
                if not isinstance(config, RFModelConfig):
                    raise AutoMLException(
                        f"Fit mode requires RFModelConfig instances, but got {type(config)}. "
                        f"If you want evals mode, set trainer_type=None."
                    )
        else:
            # Evals mode: must have dict instances
            for config in self.configs:
                if not isinstance(config, dict):
                    raise AutoMLException(
                        f"Evals mode requires dict instances, but got {type(config)}. "
                        f"If you want fit mode, provide a trainer_type."
                    )

    @abstractmethod
    def get_runs(self, seed: int) -> list[dict[str, Any]]:
        """Generate hyperparameter combinations for different training configurations."""
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")
