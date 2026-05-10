# RapidFire AI — Agent Integration Guide

Operational instructions for **AI coding agents** (Claude Code, Cursor, Codex, GitHub Copilot, Windsurf, Aider, Junie, and similar) that are helping an end user install, configure, and integrate the `rapidfireai` Python package into the user's own project.

This file is **not** for rapidfireai contributors. If you are working *on* rapidfireai itself, stop and read the repo's root [`AGENTS.md`](https://github.com/RapidFireAI/rapidfireai/blob/main/AGENTS.md) and [`CONTRIBUTING.md`](https://github.com/RapidFireAI/rapidfireai/blob/main/CONTRIBUTING.md) instead.

## 1. Audience and authority

### Source-of-truth rule (read first)

This guide does **not** restate version-specific install commands, package versions, port numbers, or known-issue workarounds. Those live in the canonical [`README.md`](https://github.com/RapidFireAI/rapidfireai/blob/main/README.md) (sections **§Prerequisites**, **§Install and Get Started**, **§Troubleshooting**) and in the codebase, and they change between releases. **Whenever this guide and the README disagree on a specific command or version, trust the README.**

What this guide *does* provide that the README does not:

- A workflow decision tree (RAG vs fine-tuning vs post-training; OpenAI vs self-hosted; lite vs full).
- Code integration patterns for the `Experiment` API and the AutoML config groups.
- Trainer-type taxonomy for fine-tuning (`SFT` / `DPO` / `GRPO`).
- Safety rules for handling user secrets, GPU assumptions, and gated model access.

### Version awareness

After installing, run `rapidfireai --version` to see the live version. This guide assumes the **0.15+ API surface**. If the installed package differs significantly, prefer the canonical docs at <https://oss-docs.rapidfire.ai>.

Canonical raw URL of this file (for `WebFetch`): <https://raw.githubusercontent.com/RapidFireAI/rapidfireai/main/docs/AGENTS.md>.

---

## 2. Workflow decision tree

Pick a branch **before** running any install commands — the two workflows install different dependency sets that are not interchangeable.

- **User wants RAG / context-engineering evaluation** → use `Experiment(..., mode="evals")` and run the *evals* variant of `rapidfireai init`.
  - Generation/embedding via **OpenAI / Azure OpenAI / Anthropic** APIs → use `RFAPIModelConfig`. **No GPU strictly required** — viable on CPU-only machines.
  - Generation via **self-hosted Hugging Face** models → use `RFvLLMModelConfig`. **GPU required.** May require Hugging Face authentication for gated models.
- **User wants fine-tuning or post-training** → use `Experiment(..., mode="fit")` and run the *default* (no-flag) `rapidfireai init`.
  - **SFT** (supervised fine-tuning, e.g., chat/QA tuning) → `trainer_type="SFT"`.
  - **DPO** (direct preference optimization, alignment from `chosen`/`rejected` pairs) → `trainer_type="DPO"`.
  - **GRPO** (group relative policy optimization, RL with reward functions) → `trainer_type="GRPO"`.

### Environment selectors

- **Remote / cloud machine** → an SSH port-forward is required to view the dashboard locally. The set of ports differs by workflow (smaller for fit, larger for evals because Jupyter and MLflow are also exposed). The current port set and the exact `ssh` command are in the README §Install — read them from there, do not memorize.
- **GPU issues** (CUDA absent, OOM, driver mismatch) → run `rapidfireai doctor` and act on its output. For OOM, switch the user to a *lite* tutorial variant (see §9).
- **Hugging Face permission issues** → confirm the user has run the README's HF auth step and has been granted access on the gated model's HF page. If access is blocked, suggest an open-license substitute (TinyLlama, Qwen-0.5B/3B) where licensing permits.

The exact CLI flag name for the evals init mode and the exact set of port numbers are intentionally not restated here — see the README, which is the authoritative install reference.

---

## 3. Setup order

Run the steps in the order below. The **commands** are in the README; the **decisions, ordering, and pitfalls** below are what you should add on top.

1. **Verify the user's environment matches README §Prerequisites *before* installing.** At minimum: (a) check the Python version (`python --version`); (b) for any **GPU-required workflow** — all fine-tuning (SFT/DPO/GRPO) and self-hosted RAG/eval — confirm GPU presence and CUDA via `nvidia-smi`. If GPU is required but absent, **stop and redirect**: the API-based RAG/eval workflow (with `RFAPIModelConfig`) does not need a GPU and is the only viable path on CPU-only hosts. Do not paper over a Python or GPU mismatch by trying to "fix" the system; surface the requirement to the user.
2. **Create and activate a virtual environment** *before* `pip install`. Do not assume the user is already in one.
3. **Install** the package as documented in README §Install. The exact `pip install` line lives there.
4. **Verify** the install: `rapidfireai --version` should return a version string. If it does not, stop and diagnose; do not proceed.
5. **Hugging Face authentication** *before* `init`/`start`, but only if the user's chosen workflow needs it (gated models, or any HF model download for self-hosted RAG/eval, or any fine-tuning). The README gives the auth command. **You must not invent a token; ask the user.**
6. **Apply current workarounds** that the README's install section calls out (for example, an `hf-xet` uninstall step, if still listed). These known-issue steps are *volatile* — read them from the live README each time, never from memory or this guide.
7. **Choose the init variant based on workflow.** This is the most-missed step. The two `rapidfireai init` variants install **different dependency sets** that are not interchangeable. The README §Install shows both invocations; §2 of this guide tells you which one to pick. If the user has already run the wrong variant, they may need to recreate the venv before re-initing.
8. **Run `rapidfireai doctor`** and confirm no critical failures (Python, GPU/CUDA where applicable, ports, packages). Do not proceed past failures without diagnosing the root cause.
9. **Start the dashboard stack** with `rapidfireai start` (frontend, MLflow, dispatcher). For the RAG/evals workflow, also start a Jupyter listener as documented in the README — the evals tutorials run inside Jupyter rather than an arbitrary IDE notebook.
10. **(Remote/cloud machines only) Port-forward** the workflow-appropriate set with `ssh` *before* clicking the dashboard URL. The current set and SSH syntax are in README §Install. VS Code typically auto-forwards the dashboard port; other clients require the explicit `ssh -L` invocation. Skip this step on a local machine.
11. **Open the tutorial notebook** under `./tutorial_notebooks/` (see §9 for the filename matching the workflow). Two non-obvious requirements:
    - Notebooks **cannot** run via `python notebook.ipynb` — there is a multiprocessing restriction. Open the notebook in an IDE that supports Jupyter (VS Code, Cursor, JetBrains), or use `rapidfireai jupyter` (mandatory for the RAG/evals tutorials).
    - The notebook's kernel **must** be set to the `.venv` you created in step 2, not the system Python; otherwise `import rapidfireai` fails. In VS Code / Cursor, click the kernel selector in the upper-right of the notebook and pick the venv interpreter.
    - The dashboard URL printed by `rapidfireai start` (or by `rapidfireai jupyter` for evals) shows live run progress as cells execute.
12. **Stop the stack** with `rapidfireai stop` when the user is done.

**Hard rule**: if any specific command above is missing here and present in the README, **use the README's version**. This guide is intentionally silent on volatile commands so it cannot drift from the README.

---

## 4. Agent safety rules

- **Do not** paste, invent, or guess API keys, Hugging Face tokens, or any other secrets. Always ask the user; if they decline, stop.
- **Do not** assume the user has a GPU. Verify with `nvidia-smi` or `rapidfireai doctor` before suggesting GPU-only workflows.
- **Do not** assume gated model access. Llama, Gemma, and several Mistral variants require explicit HF approval. Ask before substituting; never download as a different account.
- **Do not** bypass a failed `init` or `start`. Diagnose the root cause (missing CUDA, wrong Python, port conflict, dependency mismatch). Suppressing errors leads to silent breakage downstream.
- **Make minimal, explainable edits to user code.** If a change is non-obvious, explain why. Do not refactor surrounding code for style.
- **Prefer lite tutorial configs** when testing on limited hardware. The full variants assume substantial VRAM.

---

## 5. Code integration patterns

The four patterns below cover every supported workflow. Each is the *minimum* a user needs.

### A. RAG / evaluation

```python
from rapidfireai import Experiment
from rapidfireai.automl import RFGridSearch, List, RFvLLMModelConfig

experiment = Experiment("my-rag-eval", mode="evals")

config_group = RFGridSearch({
    "vllm_config": List([vllm_config_a, vllm_config_b]),
    "preprocess_fn": user_preprocess,
    "postprocess_fn": user_postprocess,
    "compute_metrics_fn": user_compute_metrics,
    # Optional:
    # "accumulate_metrics_fn": user_accumulate_metrics,
})

results = experiment.run_evals(config_group, dataset, num_shards=4)
experiment.end()
```

For an **OpenAI / Azure / Anthropic** generator, swap `RFvLLMModelConfig` for `RFAPIModelConfig`. The user-supplied callbacks are:

- `preprocess_fn(batch, rag, prompt_manager) -> dict` — assemble prompts (with retrieved context for RAG).
- `postprocess_fn(batch) -> dict` — attach ground-truth / reference labels.
- `compute_metrics_fn(batch) -> dict[str, dict]` — per-batch metrics (Precision, Recall, NDCG@k, MRR, …).
- `accumulate_metrics_fn(aggregated) -> dict[str, dict]` — optional, aggregates per-batch metrics across shards.

`run_evals` returns `dict[run_id, (aggregated_results, cumulative_metrics)]`.

### B. SFT (supervised fine-tuning)

```python
from rapidfireai import Experiment
from rapidfireai.automl import RFGridSearch

experiment = Experiment("my-sft", mode="fit")

config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

experiment.run_fit(
    param_config=config_group,
    create_model_fn=user_create_model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    num_chunks=4,
)

results_df = experiment.get_results()
experiment.end()
```

User-supplied callbacks:

- `create_model_fn(model_config) -> (model, tokenizer)` — required. Loads model and tokenizer from the per-run config dict.
- `formatting_func(row) -> dict` — optional. Returned in `RFModelConfig`. Maps a dataset row to `{"prompt": ..., "completion": ...}`.
- `compute_metrics(preds, labels) -> dict[str, float]` — optional. Returned in `RFModelConfig`.

### C. DPO (preference alignment)

Same shape as SFT but `trainer_type="DPO"`. The training dataset must have `chosen` and `rejected` columns (or whatever the chosen DPO config maps to). A reference model is handled internally; the user does not have to construct one explicitly.

### D. GRPO (RL-based post-training)

Same shape as SFT but `trainer_type="GRPO"`. Reward functions are passed via the trainer config, not as a separate argument. Use a small model and a small dataset for first runs — GRPO samples multiple generations per prompt and is significantly slower than SFT/DPO.

---

## 6. Config groups and search

Imports:

```python
from rapidfireai.automl import RFGridSearch, RFRandomSearch, List, Range
```

- **Fit mode**: `RFGridSearch(configs=config_set, trainer_type="SFT" | "DPO" | "GRPO")` — exhaustive sweep. `trainer_type` selects the underlying TRL trainer.
- **Evals mode**: `RFGridSearch({"vllm_config": List([...]), "preprocess_fn": ..., ...})` — dict form. **No `trainer_type`** (evals does inference, not training).
- `RFRandomSearch(configs, num_runs, trainer_type=...)` — sampled variant. `trainer_type` is again fit-only.
- `List([a, b, c])` and `Range(min=..., max=...)` define sweep dimensions and work in both modes.

Fit-mode config classes (from `rapidfireai.automl`): `RFModelConfig`, `RFLoraConfig`, `RFSFTConfig`, `RFDPOConfig`, `RFGRPOConfig`. Evals-mode config classes: `RFvLLMModelConfig`, `RFAPIModelConfig`, plus helpers `RFLangChainRagSpec` and `RFPromptManager`.

---

## 7. Experiment lifecycle

`rapidfireai start` (the CLI command) and `Experiment(...)` (the Python constructor) are distinct concerns:

- `rapidfireai start` brings up the **visual stack** (frontend dashboard, MLflow tracking, dispatcher API) so the user can watch runs live.
- `Experiment(...)` separately starts an in-process dispatcher thread for the API calls it needs internally.

Users typically run `rapidfireai start` first (to get the dashboard), then run their notebook or script which constructs `Experiment(...)`. The CLI `start` is **not strictly required** for `Experiment(...)` to function — but without it the dashboard UI is unavailable.

API surface (from `rapidfireai.experiment`):

- `Experiment(name, mode="fit" | "evals")` — `mode` defaults to `"fit"`. `name` is required.
- `run_fit(...)` — fit mode only. Blocking on non-Colab; async on Colab.
- `run_evals(...)` — evals mode only. Returns `dict[run_id, (results, metrics)]`.
- `get_results()` — fit mode only. Returns a pandas `DataFrame` of metrics.
- `get_runs_info()` — fit mode only. Returns a pandas `DataFrame` of run metadata.
- `cancel_current()` — both modes. Cancels an in-flight task.
- `end()` — both modes. **Required** for clean shutdown of worker processes / Ray.

Only `Experiment` is exported at the top level. AutoML primitives are imported from `rapidfireai.automl`.

---

## 8. Troubleshooting

Match the symptom to the diagnostic direction. Specific commands (kill-by-port, SSH-forward, hf-xet uninstall, etc.) are in README §Troubleshooting and §Install — do not paste them from memory.

| Symptom | Likely cause | Diagnostic direction |
|---|---|---|
| `ImportError` on `vllm`, `langchain`, or other RAG/eval-only packages | Wrong `init` variant — fit deps installed when evals deps were needed (or vice versa) | Re-run `rapidfireai init` with the variant that matches the workflow; may require recreating the venv |
| Hugging Face download hangs, or `xet` errors in the traceback | Known `hf-xet` issue (if still listed in the README) | Apply the README's current `hf-xet` workaround |
| 403 on model download | Gated model + missing/wrong HF auth | Have the user run the README's HF auth step; confirm the user has access granted on the model's HF page |
| `Address already in use` on a service port | Stale RapidFire services | Run `rapidfireai stop` first; if it persists, see README §Troubleshooting for the kill-by-port command |
| Frontend not loading from a remote machine | Missing SSH port forward | Forward the workflow-appropriate port set (see README §Install) |
| `CUDA out of memory` during fit | Model too large for available VRAM | Switch to a `*-lite` tutorial; reduce per-device batch size; consider gradient accumulation |
| `nvidia-smi` not found, or no GPU detected | CUDA toolkit absent or no GPU on host | Run `rapidfireai doctor`; on CPU-only hosts, only the API-based RAG/evals workflow is viable |
| `rapidfireai start` exits silently or fails | A service port already bound, or a service init failure | Read the printed log; check service log files reported by `rapidfireai doctor` |
| Notebook crashes immediately on `Experiment(...)` | Often a multiprocessing restriction (CLI-launched Jupyter cannot spawn child processes) | Run the notebook through an IDE (VS Code / Cursor) or `rapidfireai jupyter`, not raw `python notebook.py` |
| `ImportError: No module named 'rapidfireai'` *inside the notebook* (but `rapidfireai --version` works in the shell) | Notebook kernel is the system Python, not the project `.venv` | In the IDE's notebook UI, change the kernel/interpreter to the `.venv` you created; restart the kernel and re-run cells |

---

## 9. Tutorials

After `rapidfireai init`, tutorial notebooks are copied to `./tutorial_notebooks/`. Always prefer the on-disk copy — it matches the installed version.

| Workflow | Lite (small VRAM) | Full |
|---|---|---|
| SFT | `fine-tuning/rf-tutorial-sft-chatqa-lite.ipynb` | `fine-tuning/rf-tutorial-sft-chatqa.ipynb` |
| DPO | `post-training/rf-tutorial-dpo-alignment-lite.ipynb` | `post-training/rf-tutorial-dpo-alignment.ipynb` |
| GRPO | `post-training/rf-tutorial-grpo-mathreasoning-lite.ipynb` | `post-training/rf-tutorial-grpo-mathreasoning.ipynb` |
| RAG, self-hosted (HF) | `rag-contexteng/rf-tutorial-rag-fiqa.ipynb` | `rag-contexteng/rf-tutorial-rag-fiqa-pgvector.ipynb`, `…-pinecone.ipynb` |
| RAG, OpenAI / API | `rag-contexteng/rf-tutorial-scifact-generators.ipynb` | `rag-contexteng/rf-tutorial-scifact-full-evaluation.ipynb` |
| Few-shot eval (no retrieval) | `rag-contexteng/rf-tutorial-gsm8k-fewshot.ipynb` | — |

Online directory: <https://github.com/RapidFireAI/rapidfireai/tree/main/tutorial_notebooks>. Filenames may shift across versions — list `./tutorial_notebooks/` to confirm the current set.

---

## 10. Validation checklist (run before reporting "done")

Before telling the user setup is complete, confirm each:

1. `rapidfireai --version` returns a version string.
2. `rapidfireai doctor` reports no critical failures.
3. The right `init` variant ran for the user's workflow (fit *or* evals — not both, not the wrong one).
4. Hugging Face auth is configured iff the workflow needs it.
5. `rapidfireai start` brought up the stack without errors. The user can reach the dashboard URL printed by `start`.
6. (Remote/cloud) The `ssh -L` forward is active in a separate terminal.
7. (Tutorial path) The notebook's kernel is the project `.venv`, not the system Python. The first import cell (`from rapidfireai import Experiment`) runs without `ImportError`.
8. (Code-integration done) `Experiment(...)` constructs without raising; the user's `create_model_fn` (fit) or `preprocess_fn` (evals) runs on a single example without error.
9. `experiment.end()` is called at the end of the user's script/notebook.

If any check fails, stop and diagnose — do not paper over with retries.

---

## 11. References

- **Canonical install instructions and troubleshooting**: <https://github.com/RapidFireAI/rapidfireai/blob/main/README.md>
- **Full documentation**: <https://oss-docs.rapidfire.ai>
- **Repository**: <https://github.com/RapidFireAI/rapidfireai>
- **PyPI**: <https://pypi.org/project/rapidfireai/>
- **Discord**: <https://discord.gg/6vSTtncKNN>
- **This guide describes the 0.15+ API surface.** Confirm `rapidfireai --version` against the canonical docs when behavior differs.
