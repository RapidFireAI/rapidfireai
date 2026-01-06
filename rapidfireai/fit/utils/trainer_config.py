"""This module contains the TrainerConfig class which is responsible for configuring the trainer."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TrainerConfig:
    """Trainer configuration"""

    worker_id: int
    run_id: int
    metric_run_id: str
    config_leaf: dict[str, Any]
    total_steps: int
    completed_steps: int
    create_model_fn: Callable
    train_dataset: torch.utils.data.Dataset
    eval_dataset: torch.utils.data.Dataset | None
    warm_started_from: int | None
    cloned_from: int | None
    num_epochs_completed: int
