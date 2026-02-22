# RapidFire AI — Full API Reference

## Experiment Class

```python
from rapidfireai import Experiment

experiment = Experiment(
    experiment_name: str,          # Unique name; auto-suffixed if reused
    mode: str = "fit",             # "fit" or "eval" — cannot mix in same experiment
    experiments_path: str = "./rapidfire_experiments"
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run_fit(config_group, create_model_fn, train_dataset, eval_dataset, num_chunks, seed=42)` | Launch multi-config training |
| `run_evals(config_group, dataset, num_shards=4, num_actors, seed=42) → dict` | Launch multi-config evals; returns `{run_id: (aggregated_metrics, cumulative_metrics)}` |
| `end()` | End experiment, clear system state |
| `get_runs_info() → pd.DataFrame` | Metadata for all runs (run_id, status, config, etc.) |
| `get_results() → pd.DataFrame` | All metrics for all steps for all runs |

**Important:** Only one function runs at a time. After interrupting `run_fit()`/`run_evals()`, wait ~2 min before calling another function.

---

## Multi-Config Specification

### Knob Set Generators
```python
from rapidfireai.automl import List, Range

List([val1, val2, val3])          # Discrete set; all same Python type
Range(start, end, dtype="int")    # "int" or "float"; uniform sampling
```

### Config Group Generators
```python
from rapidfireai.automl import RFGridSearch, RFRandomSearch

RFGridSearch(
    configs: Dict | List[Dict],    # Config dict with List() knobs (no Range allowed)
    trainer_type: str = None       # "SFT" | "DPO" | "GRPO" — omit for run_evals()
)

RFRandomSearch(
    configs: Dict,                 # Can have List() or Range() knobs
    trainer_type: str = None,
    num_runs: int,                 # How many random combos to sample
    seed: int = 42
)
```

**param_config** for `run_fit()` / **config_group** for `run_evals()` accepts:
- Single config dict
- List of config dicts
- `RFGridSearch()` / `RFRandomSearch()` output
- Mixed list of any of the above

---

## SFT / RFT APIs

### RFLoraConfig
Wraps HuggingFace `peft.LoraConfig`. Any arg can be `List()` or `Range()`.

```python
from rapidfireai.automl import RFLoraConfig

RFLoraConfig(
    r=16,                                    # Rank: 8–128, powers of 2
    lora_alpha=32,                           # Usually 2x r
    lora_dropout=0.05,                       # 0.0–0.05
    target_modules=["q_proj", "v_proj"],     # or add k,o,gate,up,down projections
    bias="none",
    init_lora_weights=True,                  # or "gaussian", "pissa"
)
```

### RFModelConfig
```python
from rapidfireai.automl import RFModelConfig

RFModelConfig(
    model_name: str,                    # HF hub name or local path
    tokenizer: str = None,              # Defaults to model_name
    tokenizer_kwargs: dict = None,      # padding_side, truncation, model_max_length
    formatting_func: Callable | List,   # Can be List for multi-config
    compute_metrics: Callable | List,   # Can be List for multi-config
    peft_config: RFLoraConfig | List,   # Can be List for multi-config
    training_args,                      # RFSFTConfig | RFDPOConfig | RFGRPOConfig
    model_type: str = "causal_lm",     # Used inside your create_model_fn
    model_kwargs: dict = None,          # torch_dtype, device_map, use_cache, etc.
    # DPO/GRPO only:
    ref_model_name: str = None,
    ref_model_type: str = None,
    ref_model_kwargs: dict = None,
    reward_funcs: Callable | List = None,
    generation_config: dict = None,     # max_new_tokens, temperature, top_p
)
```

### Trainer Configs (all args can be List/Range)
```python
from rapidfireai.automl import RFSFTConfig, RFDPOConfig, RFGRPOConfig

# SFT — wraps HF TRL SFTConfig
RFSFTConfig(learning_rate, lr_scheduler_type, per_device_train_batch_size,
            per_device_eval_batch_size, gradient_accumulation_steps,
            num_train_epochs, logging_steps, eval_strategy, eval_steps,
            fp16, save_strategy, ...)

# DPO — wraps HF TRL DPOConfig
RFDPOConfig(beta, loss_type, model_adapter_name, ref_adapter_name,
            max_prompt_length, max_completion_length, max_length,
            per_device_train_batch_size, learning_rate, ...)

# GRPO — wraps HF TRL GRPOConfig
RFGRPOConfig(learning_rate, num_generations, max_prompt_length,
             max_completion_length, per_device_train_batch_size, ...)
```

### User-Provided Functions for run_fit()

**create_model_fn** (mandatory, passed to `run_fit()`):
```python
def create_model_fn(model_config: dict) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    model_name = model_config["model_name"]
    model_type = model_config["model_type"]
    model_kwargs = model_config["model_kwargs"]
    # load model based on model_type...
    return (model, tokenizer)
```

**compute_metrics_fn** (optional, passed to `RFModelConfig.compute_metrics`):
```python
def compute_metrics_fn(eval_preds: tuple) -> dict[str, float]:
    predictions, labels = eval_preds
    # return {"rougeL": 0.42, "bleu": 0.31}
```

**formatting_fn** (optional, passed to `RFModelConfig.formatting_func`):
```python
def formatting_fn(row: dict) -> dict:
    return {
        "prompt": [{"role": "system", "content": "..."}, {"role": "user", "content": row["input"]}],
        "completion": [{"role": "assistant", "content": row["output"]}]
    }
```

**reward_function** (GRPO, list passed to `RFModelConfig.reward_funcs`):
```python
def reward_fn(prompts, completions, completions_ids, trainer_state, **kwargs) -> list[float]:
    # kwargs contains all dataset columns except "prompt"
    return [1.0 if pred == gt else 0.0 for pred, gt in zip(completions, kwargs["answer"])]
```

---

## RAG / Context Engineering APIs

### RFLangChainRagSpec
```python
from rapidfireai.automl import RFLangChainRagSpec

rag = RFLangChainRagSpec(
    document_loader,              # LangChain BaseLoader
    text_splitter,                # LangChain TextSplitter — can be List()
    embedding_cls=None,           # Pass the CLASS, not an instance
    embedding_kwargs=None,        # Dict to init embedding_cls
    vector_store=None,            # Default: FAISS
    retriever=None,               # Custom retriever; default FAISS if None
    search_type="similarity",     # "similarity" | "similarity_score_threshold" | "mmr"
    search_kwargs={"k": 5},       # k, filter, score_threshold, fetch_k, lambda_mult
    reranker_cls=None,            # Pass the CLASS — can have List() kwargs
    reranker_kwargs=None,
    enable_gpu_search=False,      # True → FAISS GPU exact search
    document_template=None,       # Callable[[Document], str]
)

# Helper methods:
rag.get_context(batch_queries, use_reranker=True, serialize=True)
rag.serialize_documents(batch_docs)  # list[list[Document]] → list[str]
```

### RFvLLMModelConfig (self-hosted LLM)
```python
from rapidfireai.automl import RFvLLMModelConfig

RFvLLMModelConfig(
    model_config={
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "dtype": "half",
        "gpu_memory_utilization": 0.7,
        "tensor_parallel_size": 1,
        "distributed_executor_backend": "mp",  # only "mp" supported
        "max_model_len": 2048,
    },
    sampling_params={"temperature": 0.8, "top_p": 0.95, "max_tokens": 512},
    rag=rag_spec,           # RFLangChainRagSpec or None
    prompt_manager=None,    # RFPromptManager or None
)
```

### RFOpenAIAPIModelConfig (OpenAI API)
```python
from rapidfireai.automl import RFOpenAIAPIModelConfig

RFOpenAIAPIModelConfig(
    client_config={"api_key": OPENAI_API_KEY, "max_retries": 2},
    model_config={
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_completion_tokens": 1024,
        "reasoning_effort": List(["medium", "high"]),  # can be List
    },
    rpm_limit=500,
    tpm_limit=500_000,
    rag=None,
    prompt_manager=prompt_manager,
)
```

### RFPromptManager
```python
from rapidfireai.automl import RFPromptManager

RFPromptManager(
    instructions: str = None,           # or instructions_file_path
    instructions_file_path: str = None,
    examples: list[dict] = None,        # Few-shot examples
    embedding_cls=None,
    embedding_kwargs=None,
    example_selector_cls=None,          # SemanticSimilarityExampleSelector or MaxMarginalRelevanceExampleSelector
    example_prompt_template=None,       # LangChain PromptTemplate
    k: int = 3,                         # Number of examples per prompt
)
```

### Other Eval Config Knobs (in config dict)
```python
config = {
    "batch_size": 8,
    "preprocess_fn": preprocess_fn,          # mandatory
    "postprocess_fn": postprocess_fn,        # optional
    "compute_metrics_fn": compute_metrics_fn,# mandatory
    "accumulate_metrics_fn": accumulate_metrics_fn, # optional
    "online_strategy_kwargs": {
        "strategy_name": "normal",           # "normal" | "wilson" | "hoeffding"
        "confidence_level": 0.95,
        "use_fpc": True,
    },
}
```

### User-Provided Functions for run_evals()

**preprocess_fn** (mandatory):
```python
def preprocess_fn(batch: dict[str, list], rag: RFLangChainRagSpec, prompt_manager) -> dict[str, list]:
    # Must return dict with "prompts" key containing list of formatted prompts
    return {"prompts": [...], **batch}
```

**postprocess_fn** (optional):
```python
def postprocess_fn(batch: dict[str, list]) -> dict[str, list]:
    # batch["generated_text"] contains model outputs
    return batch
```

**compute_metrics_fn** (mandatory):
```python
def compute_metrics_fn(batch: dict[str, list]) -> dict[str, dict]:
    # Returns per-batch metrics
    return {
        "Accuracy": {"value": correct / total},
        "Total": {"value": total},
    }
```

**accumulate_metrics_fn** (optional — if omitted, all metrics treated as distributive):
```python
def accumulate_metrics_fn(aggregated_metrics: dict[str, list[dict]]) -> dict[str, dict]:
    # aggregated_metrics["MetricName"] = list of per-batch metric dicts
    return {
        "Accuracy": {"value": accuracy, "is_algebraic": True, "value_range": (0, 1)},
        "Total": {"value": total, "is_distributive": True, "value_range": (0, 1)},
    }
```

---

## IC Ops (Interactive Control)

Access via dashboard at `http://0.0.0.0:8853` or in-notebook:
```python
from rapidfireai.utils.interactive_controller import InteractiveController
controller = InteractiveController(dispatcher_url="http://127.0.0.1:8851")
controller.display()
```

| Op | When | Effect |
|----|------|--------|
| **Stop** | Any running run | Paused at next chunk boundary; no GPU used |
| **Resume** | Stopped run | Re-added to scheduler at next chunk |
| **Clone-Modify** | Any run | New run from modified knob config; can warm-start from parent weights |
| **Delete** | Any run | Removed from plots; checkpoints preserved on disk |

IC Ops execute at chunk boundaries (not immediately). Warm-start only works if clone has identical architecture as parent.

---

## Troubleshooting Quick Reference

```bash
rapidfireai doctor                              # Full diagnostic report

# Kill port conflicts:
lsof -t -i:8852 | xargs kill -9               # mlflow
lsof -t -i:8851 | xargs kill -9               # dispatcher
lsof -t -i:8853 | xargs kill -9               # frontend

# Select GPUs:
export CUDA_VISIBLE_DEVICES=0,2
rapidfireai start

# HF login issues: login from SAME venv
source .venv/bin/activate
huggingface-cli login
```

## Online Aggregation Metrics Types
- **Distributive**: purely additive (count, sum) — specify `"is_distributive": True`
- **Algebraic**: averages/proportions — specify `"is_algebraic": True`
- Both require `"value_range": (min, max)` for CI calculation
- CI strategies: `"normal"` (default, CLT), `"wilson"` (small n or near 0/1), `"hoeffding"` (distribution-free, loose)
