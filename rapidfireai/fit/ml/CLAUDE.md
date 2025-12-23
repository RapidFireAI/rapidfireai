# CLAUDE.md - ML

This file provides guidance for working with the ML training components of RapidFire AI.

## Overview

The ml module contains the training execution logic that wraps HuggingFace Transformers and TRL trainers. It handles trainer instantiation, checkpoint management, callbacks, and integration with RapidFire's chunk-based training system.

## Files

### trainer.py
**Purpose**: Creates and configures TRL trainer instances (SFT, DPO, GRPO)

**Key Responsibilities**:
- Instantiates appropriate trainer type based on config (SFTTrainer, DPOTrainer, GRPOTrainer)
- Loads model from checkpoint (shared memory or disk)
- Configures training arguments (batch size, learning rate, gradient accumulation, etc.)
- Sets up callbacks (MLflow logging, generation metrics, log level control)
- Handles PEFT (LoRA) configuration if specified
- Manages reference model for DPO training
- Restores trainer state (optimizer, scheduler) for resumed runs

**Key Functions**:
- `create_trainer_instance()`: Main entry point, returns configured trainer
- `_configure_training_args()`: Merges user args with RapidFire overrides
- `_create_trainer_config_object()`: Creates SFTConfig/DPOConfig/GRPOConfig
- `_setup_reference_model()`: Loads reference model for DPO
- `_prepare_trainer_kwargs()`: Builds kwargs dict for trainer constructor
- `_setup_callbacks()`: Initializes callbacks (MLflow, generation metrics, log level)
- `_create_trainer()`: Actually instantiates the trainer object
- `_restore_trainer_state()`: Restores optimizer/scheduler state for resumed runs

**Trainer Types**:
- **SFT** (Supervised Fine-Tuning): Standard next-token prediction
- **DPO** (Direct Preference Optimization): Preference-based training with reference model
- **GRPO** (Group Relative Policy Optimization): Advanced RL-based training

**Training Args Overrides**:
RapidFire overrides certain args to ensure chunk-based training works:
- `output_dir`: Set to experiment-specific path
- `logging_dir`: Set to experiment-specific tensorboard path
- `save_strategy`: "no" (checkpoints managed by RapidFire)
- `evaluation_strategy`: "no" or "epoch" (custom eval via callbacks)
- `max_steps`: Calculated based on chunk size
- `logging_steps`: Set to chunk-sized batches

**PEFT Integration**:
If `config_leaf['peft_params']` is provided:
- Wraps model with PEFT adapter (LoRA)
- Loads/saves adapter weights separately from base model
- Uses `get_peft_model_state_dict()` and `set_peft_model_state_dict()`

**Patterns**:
- Expects `TrainerConfig` object with all necessary info (run_id, worker_id, config_leaf, etc.)
- Returns tuple of (trainer, status_string)
- Handles both fresh runs and resumed runs (chunk_id > 0)
- Uses `USE_SHARED_MEMORY` flag to decide checkpoint loading strategy

### checkpoint_utils.py
**Purpose**: Checkpoint loading, saving, and restoration

**Key Responsibilities**:
- Save/load model checkpoints to/from shared memory
- Save/load model checkpoints to/from disk
- Restore trainer state (optimizer, scheduler, RNG) for resumed runs
- Handle PEFT adapter checkpoints separately from base models
- Move tensors between CPU and GPU for memory efficiency

**Key Functions - Shared Memory**:
- `save_model_to_shared_memory()`: Store model weights in SHM
- `save_checkpoint_to_shared_memory()`: Store model + optimizer state in SHM
- `load_checkpoint_from_shared_memory()`: Load model from SHM registry
- `restore_trainer_from_shared_memory()`: Restore trainer state from SHM

**Key Functions - Disk**:
- `save_checkpoint_to_disk()`: Save checkpoint to experiment directory
- `load_checkpoint_from_disk()`: Load checkpoint from disk
- `restore_trainer_from_disk()`: Restore trainer state from disk checkpoint
- `load_or_create_ref_model()`: Load reference model for DPO (always from disk)

**Key Functions - Utilities**:
- `move_tensors_to_cpu()`: Move all tensors in dict to CPU (for SHM storage)
- `move_tensors_to_device()`: Move all tensors to specified device
- `ensure_gradient_compatibility()`: Fix gradient dtype mismatches
- `_get_checkpoint_path()`: Generate checkpoint filename path

**Checkpoint Structure**:
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainer.optimizer.state_dict(),
    'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
    'rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state(),
    'epoch': current_epoch,
    'global_step': trainer.state.global_step,
}
```

**Shared Memory Registry**:
- Key format: `f"{run_id}_model"`, `f"{run_id}_checkpoint"`
- Stores model on CPU to avoid GPU memory issues
- Uses `SharedMemoryManager` for access coordination

**Disk Checkpoint Paths**:
- Pattern: `{experiment_path}/runs/run_{run_id}/checkpoints/checkpoint_chunk_{chunk_id}.pt`
- Saved after each chunk completion
- Used when SHM disabled or checkpoint too large

**PEFT Handling**:
- PEFT adapters saved separately: `checkpoint['adapter_state_dict']`
- Base model not saved (only adapters)
- Load base model fresh, then apply saved adapter

### callbacks.py
**Purpose**: Custom Transformers callbacks for RapidFire integration

**Key Responsibilities**:
- Log training metrics to MLflow
- Compute generation-based metrics during evaluation
- Control log verbosity during training

**Callbacks**:

**GenerationMetricsCallback**:
- Generates text during evaluation to compute quality metrics
- Uses user-provided `compute_metrics` function
- Logs generation metrics to MLflow (e.g., BLEU, ROUGE, custom metrics)
- Batches generation for efficiency
- Supports custom generation configs (temperature, top_p, max_tokens)

**MetricLoggingCallback**:
- Logs training metrics (loss, learning rate, grad norm) to Metric Logger
- Handles step offset for resumed runs (continued step numbering)
- Filters out None values and non-numeric metrics
- Logs at appropriate intervals based on `logging_steps`

**LogLevelCallback**:
- Temporarily reduces log verbosity during training
- Prevents console spam from Transformers
- Restores original log level after training
- Uses `transformers.logging.set_verbosity()`

**Usage Pattern**:
```python
callbacks = [
    MetricLoggingCallback(metric_manager, metric_run_id, completed_steps),
    GenerationMetricsCallback(tokenizer, eval_dataset, generation_config, compute_metrics),
    LogLevelCallback(),
]
trainer = SFTTrainer(..., callbacks=callbacks)
```

## Key Concepts

### Chunk-Based Training
- Each chunk is a separate training session with max_steps calculated for that chunk
- Trainer created fresh for each chunk (avoids state leakage)
- Checkpoint saved after chunk completion
- Next chunk loads checkpoint and continues

### Trainer State Restoration
When resuming from checkpoint:
1. Load model state dict
2. Restore optimizer state dict
3. Restore scheduler state dict
4. Restore RNG states (CPU and CUDA)
5. Set trainer.state.global_step to continue step numbering
6. Metrics continue from previous chunk

### PEFT (LoRA) Support
- User specifies `peft_params` in config with LoRA config (r, alpha, dropout, target_modules)
- Model wrapped with `get_peft_model(model, lora_config)`
- Only adapter weights saved/loaded (base model stays frozen)
- Reduces checkpoint size and training memory

### DPO Reference Model
- DPO requires reference model for KL divergence penalty
- Reference model loaded from disk (never updated during training)
- Moved to same device as main model
- Shared across all chunks (not checkpointed)

## Common Patterns

### Adding New Trainer Type
1. Import trainer class from TRL (e.g., `from trl import NewTrainer`)
2. Add config class import (e.g., `from trl import NewConfig`)
3. Add trainer type to `_create_trainer_config_object()`
4. Add trainer type to `_create_trainer()` instantiation logic
5. Handle any special args in `_prepare_trainer_kwargs()`
6. Update AutoML base class `VALID_TRAINER_TYPES` in `automl/base.py`

### Adding Custom Metrics
User provides `compute_metrics` function in config:
```python
def compute_metrics(predictions, references):
    # Custom metric computation
    return {'custom_metric': score}

config = {
    'additional_kwargs': {
        'compute_metrics': compute_metrics
    }
}
```

Integrated via `GenerationMetricsCallback`.

### Debugging Training Issues
- Check `trainer.state.log_history` for metrics
- Inspect checkpoint files on disk (torch.load)
- Add logging in `_restore_trainer_state()` to verify state restoration
- Check MLflow UI for metric continuity across chunks
- Verify `global_step` increments correctly across chunks

### Memory Optimization
- Use PEFT to reduce memory footprint
- Enable gradient checkpointing: `training_args['gradient_checkpointing'] = True`
- Reduce batch size or increase gradient accumulation
- Use bfloat16 or float16: `training_args['bf16'] = True`
- Disable shared memory if checkpoints too large: `USE_SHARED_MEMORY = False`

## Integration with Backend

1. **Worker calls `create_trainer_instance()`**:
   - Passes `TrainerConfig` with run details
   - Gets back configured trainer

2. **Worker calls `trainer.train()`**:
   - Trains for `max_steps` (one chunk worth)
   - Callbacks log to MLflow

3. **Worker saves checkpoint**:
   - Calls `save_checkpoint_to_shared_memory()` or `save_checkpoint_to_disk()`
   - Stores optimizer/scheduler state for next chunk

4. **Next chunk loads checkpoint**:
   - Worker calls `create_trainer_instance()` with chunk_id > 0
   - `trainer.py` detects resumed run and calls restoration functions
   - Training continues from exact state

## Testing

Manual testing:
```python
# Test checkpoint save/load
from rapidfireai.ml.checkpoint_utils import save_checkpoint_to_disk, load_checkpoint_from_disk

save_checkpoint_to_disk(trainer, run_id=1, chunk_id=0, epoch=0)
model, tokenizer = load_checkpoint_from_disk(trainer_config, is_peft=False)
```

Integration testing via tutorial notebooks.
