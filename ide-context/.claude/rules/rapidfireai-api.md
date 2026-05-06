# RapidFire AI — Full API Reference

## Experiment Class

```python
from rapidfireai import Experiment

experiment = Experiment(
    experiment_name: str,          # Unique name; auto-suffixed if reused
    mode: str = "fit",             # "fit" or "evals" — cannot mix in same experiment
    experiment_path: str = "$RF_HOME/rapidfire_experiments",
    num_cpus: int = None,          # evals mode: cap Ray CPU allocation; auto-detected if None
    num_gpus: int = None,          # evals mode: cap Ray GPU allocation; auto-detected if None
)
```

**Methods:**

| Method | Description |
|--------|-------------|
| `run_fit(param_config, create_model_fn, train_dataset, eval_dataset, num_chunks, seed=42, num_gpus=1, monte_carlo_simulations=1000)` | Launch multi-config training |
| `run_evals(config_group, dataset, num_shards=4, seed=42, num_actors=None, gpus_per_actor=None, cpus_per_actor=None) → dict` | Launch multi-config evals; returns `{run_id: (aggregated_metrics, cumulative_metrics)}` |
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
)   # Note: trainer_type is required for fit mode; omit or set None for evals mode


RFRandomSearch(
    configs: Dict,                 # Can have List() or Range() knobs
    trainer_type: str = None,
    num_runs: int = 1,             # How many random combos to sample
)
```

**param_config** (for `run_fit()`) / **config_group** (for `run_evals()`) accepts:
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
All component configuration is passed as `*_cfg` dicts. Each dict carries its class/type
plus the kwargs that class needs.

```python
from rapidfireai.automl import RFLangChainRagSpec
from langchain_community.embeddings import HuggingFaceEmbeddings

rag = RFLangChainRagSpec(
    document_loader,              # LangChain BaseLoader (required unless retriever or vector_store_cfg provided)
    text_splitter=None,           # LangChain TextSplitter — optional; can be List() across configs
    embedding_cfg={               # Dict: "class" key + remaining keys are kwargs for the class.
        "class": HuggingFaceEmbeddings,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
    vector_store_cfg={"type": "faiss"},   # See per-type shapes below; ignored if retriever is set
    retriever=None,               # Custom BaseRetriever; if set, used directly
    search_cfg={"type": "similarity", "k": 5},   # type ∈ {"similarity","similarity_score_threshold","mmr"} + kwargs
    reranker_cfg=None,            # Optional dict: {"class": CrossEncoderReranker, **kwargs}
    enable_gpu_search=False,      # True → FAISS GPU exact search at index build time
    document_template=None,       # Callable[[Document], str]
)

# Helper methods:
rag.get_context(batch_queries, use_reranker=True, serialize=True)
rag.serialize_documents(batch_docs)  # list[list[Document]] → list[str]
```

**`vector_store_cfg` shapes** (pick one per run; can be List()-wrapped to compare):
```python
# FAISS (default; in-memory, per-experiment)
vector_store_cfg = {"type": "faiss"}

# pgvector (PostgreSQL with pgvector extension)
vector_store_cfg = {
    "type": "pgvector",
    "connection": "postgresql+psycopg://user:pass@host:5432/db",
    "collection_name": "rapidfire_corpus",
}

# Pinecone (managed)
vector_store_cfg = {
    "type": "pinecone",
    "index_name": "rapidfire-fiqa",
    "api_key": PINECONE_API_KEY,
}
```

> Removed in 0.15.x: `vector_store=`, `search_type=`, `search_kwargs=`, `reranker_cls=`,
> `reranker_kwargs=`, `embedding_cls=`, `embedding_kwargs=`. Use the `*_cfg` dicts above.

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
    max_completion_tokens=None,    # Defaults to 150 if not set in model_config
    rag=None,
    prompt_manager=prompt_manager,
)
```

### RFGeminiAPIModelConfig (Google Gemini API)
Same constructor shape as `RFOpenAIAPIModelConfig`. Note the `model_config` fallback key
for token cap is `max_output_tokens` (Gemini convention) rather than `max_completion_tokens`.

```python
from rapidfireai.automl import RFGeminiAPIModelConfig

RFGeminiAPIModelConfig(
    client_config={"api_key": GEMINI_API_KEY},   # or {"vertexai": True, "project": "...", "location": "..."}
    model_config={
        "model": "gemini-2.0-flash",
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
        "max_output_tokens": 1024,
    },
    rpm_limit=500,
    tpm_limit=500_000,
    max_completion_tokens=None,    # Falls back to model_config["max_output_tokens"], else 150
    rag=None,
    prompt_manager=prompt_manager,
)
```

### RFPromptManager
```python
from rapidfireai.automl import RFPromptManager
from langchain_community.embeddings import HuggingFaceEmbeddings

RFPromptManager(
    instructions: str = "",                # or instructions_file_path
    instructions_file_path: str = "",
    examples: list[dict] = [],             # Few-shot examples
    embedding_cfg={                        # Dict: "class" key + remaining keys are kwargs.
        "class": HuggingFaceEmbeddings,
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    },
    example_selector_cls=None,             # SemanticSimilarityExampleSelector or MaxMarginalRelevanceExampleSelector
    example_prompt_template=None,          # LangChain PromptTemplate
    k: int = 3,                            # Number of examples per prompt
)
```

> Removed in 0.15.x: `embedding_cls=`, `embedding_kwargs=`. Use `embedding_cfg` dict.

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

Access via dashboard at `http://127.0.0.1:8853` or in-notebook:
```python
from rapidfireai.fit.utils.interactive_controller import InteractiveController
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

# Kill port conflicts (all RapidFire ports):
lsof -t -i:8850 | xargs kill -9               # jupyter
lsof -t -i:8851 | xargs kill -9               # dispatcher
lsof -t -i:8852 | xargs kill -9               # mlflow
lsof -t -i:8853 | xargs kill -9               # frontend
lsof -t -i:8855 | xargs kill -9               # ray (evals mode)

# Select GPUs:
export CUDA_VISIBLE_DEVICES=0,2
rapidfireai start

# Pin CUDA / compute capability when nvidia-smi is unavailable or auto-detection fails:
rapidfireai init --evals --cudaversion 12.4 --computecapabilityversion 8.0

# HF login issues: login from SAME venv
source .venv/bin/activate
huggingface-cli login
```

## Online Aggregation Metrics Types
- **Distributive**: purely additive (count, sum) — specify `"is_distributive": True`
- **Algebraic**: averages/proportions — specify `"is_algebraic": True`
- Both require `"value_range": (min, max)` for CI calculation
- CI strategies: `"normal"` (default, CLT), `"wilson"` (small n or near 0/1), `"hoeffding"` (distribution-free, loose)
