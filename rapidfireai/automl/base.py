"""Base class for AutoML search algorithms.

Classes
-------
AutoMLAlgorithm
    Abstract base subclassed by ``RFGridSearch``, ``RFRandomSearch``,
    and ``RFOptuna``.
"""

from abc import ABC, abstractmethod
from typing import Any

from rapidfireai.automl.datatypes import List
from rapidfireai.fit.utils.exceptions import AutoMLException


class AutoMLAlgorithm(ABC):
    """Abstract base class for AutoML search strategies.

    Parameters
    ----------
    configs :
        Config templates (``RFModelConfig`` for fit, dicts for evals).
        Accepts a list, a ``List([...])`` wrapper, or a single object.
    create_model_fn :
        Legacy parameter (unused).
    trainer_type : str or None
        ``"SFT"`` / ``"DPO"`` / ``"GRPO"`` for fit mode, ``None`` for evals.
    num_runs : int
        Number of samples (used by ``RFRandomSearch``).

    Attributes
    ----------
    configs : list
    mode : str
        ``"fit"`` or ``"evals"``.
    trainer_type : str or None
    num_runs : int

    Methods
    -------
    get_runs(seed) -> list[dict]
        Return concrete config-leaf dicts.
    get_callback(**kwargs) -> ChunkCallback | ShardCallback | None
        Return an optional inter-step pruning callback.
    """

    VALID_TRAINER_TYPES = {"SFT", "DPO", "GRPO"}

    def __init__(
        self,
        configs=None,
        create_model_fn=None,
        trainer_type: str | None = None,
        num_runs: int = 1,
    ):
        try:
            self.configs = self._normalize_configs(configs)
            self.num_runs = num_runs

            # Detect mode based on trainer_type
            if trainer_type is not None:
                self.mode = "fit"
                self.trainer_type = trainer_type.upper()
                if self.trainer_type not in self.VALID_TRAINER_TYPES:
                    raise AutoMLException(
                        f"trainer_type must be one of {self.VALID_TRAINER_TYPES}"
                    )
            else:
                self.mode = "evals"
                self.trainer_type = None

            self._validate_configs()
        except Exception as e:
            raise AutoMLException(
                f"Error initializing {self.__class__.__name__}: {e}"
            ) from e

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

    def get_callback(self, **kwargs):
        """Return an optional callback for inter-chunk/shard pruning decisions.

        Returns
        -------
        ChunkCallback or ShardCallback or None
        """
        return None

    @abstractmethod
    def get_runs(self, seed: int) -> list[dict[str, Any]]:
        """Return concrete config-leaf dicts for the controller.

        Parameters
        ----------
        seed : int
            Non-negative random seed.

        Returns
        -------
        list[dict[str, Any]]
        """
        if not isinstance(seed, int) or seed < 0:
            raise AutoMLException("seed must be a non-negative integer")
