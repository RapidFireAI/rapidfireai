# CLAUDE.md - AutoML

This file provides guidance for working with the AutoML search algorithms in RapidFire AI.

## Overview

The automl module provides search algorithms for hyperparameter tuning and configuration exploration. Instead of manually creating runs one-by-one, users can specify search spaces and let RapidFire generate multiple configurations automatically.

## Files

### base.py
**Purpose**: Abstract base class for all AutoML algorithms

**Key Responsibilities**:
- Defines common interface for AutoML algorithms
- Validates trainer types (SFT, DPO, GRPO)
- Normalizes config inputs (list vs single config)
- Enforces that all configs are `RFModelConfig` instances

**Key Methods**:
- `get_runs(seed)`: Abstract method that subclasses implement to generate run configurations
- `_validate_configs()`: Ensures all configs are RFModelConfig instances
- `_normalize_configs()`: Converts various input formats to list of configs

**Usage Pattern**:
```python
class MySearchAlgorithm(AutoMLAlgorithm):
    def get_runs(self, seed):
        # Generate list of config dicts
        return [config_dict_1, config_dict_2, ...]
```

### grid_search.py
**Purpose**: Exhaustive grid search over hyperparameter combinations

**Key Responsibilities**:
- Generates all possible combinations from parameter grid
- Uses itertools.product for Cartesian product
- Supports nested parameter spaces via `RFModelConfig`

**Key Methods**:
- `__init__(configs, trainer_type, num_runs)`: Takes list of `RFModelConfig` with parameter lists
- `get_runs(seed)`: Generates all combinations (seed unused for deterministic search)

**Usage Example**:
```python
from rapidfireai.automl import GridSearch, RFModelConfig

config = RFModelConfig(
    training_args={
        'learning_rate': [1e-4, 1e-5, 1e-6],
        'per_device_train_batch_size': [8, 16],
        'num_train_epochs': [3],
    },
    peft_params={
        'r': [8, 16],
        'lora_alpha': [32],
    }
)

grid_search = GridSearch(configs=[config], trainer_type="SFT")
# Generates 3 * 2 * 1 * 2 * 1 = 12 runs
```

**Notes**:
- Number of runs = product of all parameter list lengths
- Can explode quickly with many parameters
- Deterministic (same configs every time)

### random_search.py
**Purpose**: Random sampling from hyperparameter distributions

**Key Responsibilities**:
- Randomly samples from parameter distributions
- Supports discrete (list) and continuous (distribution) parameters
- Uses seed for reproducibility
- Limits number of samples via `num_runs`

**Key Methods**:
- `__init__(configs, trainer_type, num_runs)`: Takes configs with distributions and sample count
- `get_runs(seed)`: Generates `num_runs` random samples using seed
- `_sample_from_config()`: Samples single config from distributions

**Usage Example**:
```python
from rapidfireai.automl import RandomSearch, RFModelConfig, distributions

config = RFModelConfig(
    training_args={
        'learning_rate': distributions.loguniform(1e-6, 1e-3),
        'per_device_train_batch_size': [8, 16, 32],  # discrete
        'num_train_epochs': [3],
    }
)

random_search = RandomSearch(configs=[config], trainer_type="SFT", num_runs=10)
# Generates 10 randomly sampled runs
```

**Distribution Types** (from `datatypes.py`):
- `uniform(low, high)`: Uniform distribution
- `loguniform(low, high)`: Log-uniform (good for learning rates)
- `randint(low, high)`: Random integer
- Lists: Uniform random choice from list

### model_config.py
**Purpose**: Configuration container for model and training parameters

**Key Responsibilities**:
- Wraps all parameters needed to create a trainer
- Supports trainer_type, training_args, peft_params, additional_kwargs
- Used by AutoML algorithms to define search spaces
- Validates parameter structure

**Key Attributes**:
- `trainer_type`: "SFT", "DPO", or "GRPO"
- `training_args`: Dict of HuggingFace TrainingArguments
- `peft_params`: Dict of PEFT/LoRA config (optional)
- `additional_kwargs`: Extra kwargs for trainer (e.g., compute_metrics, formatting_func)

**Usage**:
```python
config = RFModelConfig(
    trainer_type="SFT",
    training_args={
        'learning_rate': 1e-5,
        'num_train_epochs': 3,
    },
    peft_params={
        'r': 8,
        'lora_alpha': 32,
        'target_modules': ['q_proj', 'v_proj'],
    },
    additional_kwargs={
        'compute_metrics': my_metrics_fn,
    }
)
```

### datatypes.py
**Purpose**: Type definitions and distribution classes for parameter sampling

**Key Classes**:
- `List`: Wrapper for list of values (discrete choice)
- `Distribution`: Base class for continuous distributions
- `uniform`, `loguniform`, `randint`: Specific distribution implementations

**Usage**:
```python
from rapidfireai.automl.datatypes import uniform, loguniform

lr = loguniform(1e-6, 1e-3)  # Log-uniform between 1e-6 and 1e-3
batch_size = List([8, 16, 32])  # Discrete choice
```

## Key Concepts

### Parameter Space Definition
Two ways to specify parameter ranges:

1. **Grid Search**: Lists of discrete values
```python
'learning_rate': [1e-4, 1e-5, 1e-6]  # Try all three
```

2. **Random Search**: Distributions or lists
```python
'learning_rate': loguniform(1e-6, 1e-3)  # Sample from distribution
'batch_size': [8, 16, 32]  # Random choice from list
```

### config_leaf Format
The output of `get_runs()` is a list of `config_leaf` dicts:
```python
config_leaf = {
    'trainer_type': 'SFT',
    'training_args': {
        'learning_rate': 1e-5,
        'per_device_train_batch_size': 8,
        # ... other args
    },
    'peft_params': {
        'r': 8,
        'lora_alpha': 32,
        # ... other peft args
    },
    'additional_kwargs': {
        'compute_metrics': fn,
    }
}
```

This dict is stored in the database as `config_leaf` column and passed to `create_trainer_instance()`.

### Seed Handling
- Seed passed to `get_runs(seed)` by Controller
- Used for reproducibility in RandomSearch
- GridSearch ignores seed (deterministic)
- Same seed = same random samples

## Common Patterns

### Using with Experiment
```python
from rapidfireai import Experiment
from rapidfireai.automl import GridSearch, RFModelConfig

config = RFModelConfig(...)
grid_search = GridSearch(configs=[config], trainer_type="SFT")

exp = Experiment("my_experiment")
exp.run_fit(
    param_config=grid_search,  # Pass AutoML algorithm
    create_model_fn=my_model_fn,
    train_dataset=train_data,
    eval_dataset=eval_data,
    num_chunks=8,
    seed=42
)
```

### Multiple Config Spaces
```python
# Search over two different model architectures
config1 = RFModelConfig(...)  # Model A params
config2 = RFModelConfig(...)  # Model B params

grid_search = GridSearch(configs=[config1, config2], trainer_type="SFT")
# Will search over both config spaces
```

### Hybrid Search
```python
# Some params grid, some random
config = RFModelConfig(
    training_args={
        'learning_rate': loguniform(1e-6, 1e-3),  # Random
        'num_train_epochs': [1, 3, 5],  # Grid (if using RandomSearch, random choice)
    }
)

random_search = RandomSearch(configs=[config], num_runs=20)
```

## Adding New Search Algorithms

1. **Create new file** (e.g., `bayesian_optimization.py`)

2. **Subclass AutoMLAlgorithm**:
```python
from rapidfireai.automl.base import AutoMLAlgorithm

class BayesianOptimization(AutoMLAlgorithm):
    def __init__(self, configs, trainer_type="SFT", num_runs=10):
        super().__init__(configs, trainer_type, num_runs)
        # Additional initialization

    def get_runs(self, seed):
        # Implement search logic
        configs = []
        for i in range(self.num_runs):
            # Sample config based on previous results
            config = self._sample_config()
            configs.append(config)
        return configs
```

3. **Update `__init__.py`**:
```python
from .bayesian_optimization import BayesianOptimization

__all__ = ['AutoMLAlgorithm', 'GridSearch', 'RandomSearch', 'BayesianOptimization', ...]
```

4. **Test**:
```python
bo = BayesianOptimization(configs=[config], num_runs=10)
runs = bo.get_runs(seed=42)
assert len(runs) == 10
```

## Integration with Controller

1. User passes AutoML algorithm to `Experiment.run_fit(param_config=grid_search)`
2. Controller detects it's an AutoML instance (not plain dict)
3. Controller calls `get_runs(seed)` to generate configs
4. Controller creates runs in DB for each config
5. Workers train all configs concurrently (chunk-based)

Flow:
```
User → Experiment.run_fit(param_config=AutoMLAlgorithm)
     → Controller._create_models(param_config, ...)
     → get_runs(seed) → [config1, config2, ...]
     → db.create_run() for each config
     → Scheduler assigns to workers
```

## Testing

Manual testing:
```python
from rapidfireai.automl import GridSearch, RFModelConfig

config = RFModelConfig(
    training_args={'learning_rate': [1e-4, 1e-5]},
)
grid = GridSearch(configs=[config])
runs = grid.get_runs(seed=42)
print(len(runs))  # Should be 2
print(runs[0])    # First config
```

Integration testing via tutorial notebooks with AutoML examples.
