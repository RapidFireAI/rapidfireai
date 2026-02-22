# RapidFire AI Project

## What Is This
RapidFire AI (`rapidfireai`) is a Python package for hyperparallelized LLM experimentation — running multiple configs concurrently, with real-time IC Ops (stop/resume/clone/delete runs mid-flight).

Supports two workflows:
- **RAG / Context Engineering** — evals mode, uses `run_evals()`
- **Fine-Tuning / Post-Training** — fit mode (SFT, DPO, GRPO), uses `run_fit()`

## Environment Requirements
- Python **3.12+** (required — verify before creating venv)
- `pip install rapidfireai`
- After install: `pip uninstall -y hf-xet` (known upstream bug)
- Ports used: **8850** (jupyter), **8851** (dispatcher), **8852** (mlflow), **8853** (frontend)

## Server Lifecycle
```bash
rapidfireai init          # Run ONCE per venv (or when switching GPUs)
rapidfireai init --evals  # RAG/evals variant
rapidfireai start         # Start services — leave terminal running
rapidfireai stop          # Graceful stop (NEVER kill -9)
rapidfireai doctor        # Diagnostics
```

## Core Mental Model
```
Experiment (named, unique)
  └── Config Group (grid/random search over knobs)
        └── Run (one leaf config = one run)
              └── Chunks/Shards (data split for concurrent execution)
```

## Canonical Code Pattern

### Fit (SFT/DPO/GRPO)
```python
from rapidfireai import Experiment
from rapidfireai.automl import RFModelConfig, RFLoraConfig, RFSFTConfig, List, RFGridSearch

experiment = Experiment(experiment_name="my-exp", mode="fit")

config_group = RFGridSearch(
    configs={
        "model_config": RFModelConfig(
            model_name="meta-llama/Llama-3.1-8B-Instruct",
            peft_config=RFLoraConfig(r=List([16, 128]), lora_alpha=List([32, 256])),
            training_args=RFSFTConfig(learning_rate=2e-4, num_train_epochs=2),
            model_kwargs={"device_map": "auto", "torch_dtype": "auto"},
        )
    },
    trainer_type="SFT"
)

experiment.run_fit(config_group, create_model_fn, train_dataset, eval_dataset, num_chunks=4)
experiment.end()
```

### Eval (RAG)
```python
experiment = Experiment(experiment_name="my-rag-exp", mode="evals")
results = experiment.run_evals(config_group, dataset=eval_dataset, num_shards=4, num_actors=8)
experiment.end()
```

## Key Rules
- `mode="fit"` → use `run_fit()` | `mode="evals"` → use `run_evals()`
- `List([...])` wraps discrete knob values; `Range(start, end, dtype=)` for continuous
- `RFGridSearch` → no `Range()` allowed; `RFRandomSearch` → allows both `List` and `Range`
- `num_chunks` / `num_shards` ≥ 4 recommended for meaningful concurrency
- `trainer_type` argument required for `RFGridSearch`/`RFRandomSearch` in fit mode; omit for evals
- Stopping the server forcibly loses experiment artifacts — always use `rapidfireai stop`

## Detailed API Reference
See `.claude/rules/rapidfireai-api.md` for full class signatures, all config knobs, and user-provided function contracts.
