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
    def val_metrics_path(cls, base_run_path: Path) -> Path:
        """Return the validation metrics path"""
        return cls.work_dir_path(base_run_path) / "val_metrics.csv"

    @classmethod
    def ref_model_path(cls, base_run_path: Path) -> Path:
        """Return the reference model path for DPO training"""
        return base_run_path / "ref_model"
