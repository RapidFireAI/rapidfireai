"""
This module contains the DatasetLocators class which encapsulates the filepaths for the datasets.
"""

from pathlib import Path

from rapidfireai.fit.utils.exceptions import DataPathException
from rapidfireai.utils.os_utils import mkdir_p


class DataPath:
    """Class to set the data paths for ML"""

    @classmethod
    def initialize(cls, experiment_name: str, experiments_path: str) -> None:
        """Create directories for the ML paths"""

        try:
            # set standard paths
            cls.experiments_path: Path = Path(experiments_path) / f"{experiment_name}"
            cls.user_code_path: Path = cls.experiments_path / "code"
            cls.mlflow_path: Path = cls.experiments_path / "mlflow"

            # create directories
            mkdir_p(cls.experiments_path, notify=False)
            mkdir_p(cls.user_code_path, notify=False)
            mkdir_p(cls.mlflow_path, notify=False)
        except (PermissionError, OSError) as e:
            raise DataPathException(f"Failed to create required DataPaths directories: {e}") from e

    @classmethod
    def base_run_path(cls, run_id: str | int) -> Path:
        """Return the work directory path"""
        return cls.experiments_path / "runs" / f"{run_id}"

    @classmethod
    def dataset_path(cls) -> Path:
        """Return the dataset path"""
        return cls.experiments_path / "datasets.dill"

    @classmethod
    def work_dir_path(cls, base_run_path: Path) -> Path:
        """Return the work directory path"""
        return base_run_path / "work_dir"

    @classmethod
    def initial_checkpoint_path(cls, base_run_path: Path) -> Path:
        """Return the initial checkpoint path"""
        return base_run_path / "checkpoints" / "initial_checkpoint"

    @classmethod
    def final_checkpoint_path(cls, base_run_path: Path) -> Path:
        """Return the final checkpoint path"""
        return base_run_path / "checkpoints" / "final_checkpoint"

    @classmethod
    def intermediate_checkpoint_path(cls, base_run_path: Path) -> Path:
        """Return the intermediate checkpoint path"""
        return base_run_path / "checkpoints" / "intermediate_checkpoints"

    @classmethod
    def intermediate_checkpoint_for_step(
        cls, base_run_path: Path, completed_steps: int
    ) -> Path:
        """Return a per-step subfolder under ``intermediate_checkpoints/``.

        Each chunk save lands in its own ``checkpoint-<step>`` folder (mirrors
        HF Trainer's convention) so users get one snapshot per chunk instead of
        a single overwritten folder. The legacy ``checkpoint`` name is kept as
        a fallback for the rare case ``completed_steps`` is 0 (e.g. an initial
        intermediate save before any training step has run).
        """
        if completed_steps and completed_steps > 0:
            return (
                cls.intermediate_checkpoint_path(base_run_path)
                / f"checkpoint-{completed_steps}"
            )
        return cls.intermediate_checkpoint_path(base_run_path) / "checkpoint"

    @classmethod
    def latest_intermediate_checkpoint(cls, base_run_path: Path) -> Path | None:
        """Return the most recent ``checkpoint-<step>`` folder for this run.

        Falls back to the legacy ``checkpoint`` folder (without a step suffix)
        for backwards compatibility with runs created before per-step folders
        were introduced. Returns ``None`` if no intermediate checkpoint exists.
        """
        root = cls.intermediate_checkpoint_path(base_run_path)
        if not root.exists():
            return None
        best_step = -1
        best_path: Path | None = None
        for child in root.iterdir():
            if not child.is_dir():
                continue
            name = child.name
            if name.startswith("checkpoint-"):
                try:
                    step = int(name.split("-", 1)[1])
                except ValueError:
                    continue
                if step > best_step:
                    best_step = step
                    best_path = child
            elif name == "checkpoint" and best_path is None:
                # legacy single-folder layout; only used if no step folder exists
                best_path = child
        return best_path

    @classmethod
    def val_metrics_path(cls, base_run_path: Path) -> Path:
        """Return the validation metrics path"""
        return cls.work_dir_path(base_run_path) / "val_metrics.csv"

    @classmethod
    def ref_model_path(cls, base_run_path: Path) -> Path:
        """Return the reference model path for DPO training"""
        return base_run_path / "ref_model"
