# Post-training tutorials

Preference alignment with DPO and math reasoning with GRPO, two of the Hugging Face TRL trainers that RapidFire AI supports. Set up with `rapidfireai init --train`, then run `rapidfireai start`. The [DPO](https://oss-docs.rapidfire.ai/en/latest/dpo.html) and [GRPO](https://oss-docs.rapidfire.ai/en/latest/grpo.html) pages in the docs walk through each use case.

| Notebook | What it shows | Models | Needs |
|---|---|---|---|
| [`rf-tutorial-dpo-alignment.ipynb`](rf-tutorial-dpo-alignment.ipynb) | DPO on 500 examples of `trl-lib/ultrafeedback_binarized`: 4 configs that vary LoRA rank (64, 128), beta and loss (sigmoid, robust, and a weighted sigmoid + BCO-pair + SFT mix) | `rapidfire-ai-inc/mistral-7b-sft-bnb-4bit`, a 4-bit QLoRA SFT of Mistral-7B-Instruct-v0.3 | GPU with ≥80 GB |
| [`rf-tutorial-dpo-alignment-lite.ipynb`](rf-tutorial-dpo-alignment-lite.ipynb) | 3 of those configs with LoRA r=16, 128 examples and shorter sequences | Same as above | GPU with <80 GB |
| [`rf-tutorial-dpo-alignment-fsdp-lite.ipynb`](rf-tutorial-dpo-alignment-fsdp-lite.ipynb) | The 3 lite configs with FSDP across 2 GPUs | TinyLlama-1.1B-Chat-v1.0 | ≥2 A10 GPUs (48 GB total) |
| [`rf-tutorial-grpo-mathreasoning.ipynb`](rf-tutorial-grpo-mathreasoning.ipynb) | GRPO on `openai/gsm8k` with 5 reward functions (correctness, integer answer, strict format, soft format, XML tag count); 4 configs | Llama-3.1-8B-Instruct, Qwen2.5-3B-Instruct (2 learning rates), Qwen2.5-7B-Instruct, all loaded in 4-bit | GPU with ≥80 GB; Hugging Face access to Llama |
| [`rf-tutorial-grpo-mathreasoning-lite.ipynb`](rf-tutorial-grpo-mathreasoning-lite.ipynb) | Same task with 2 small models, not quantized; 2 configs | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct | GPU with <80 GB; Hugging Face access to Llama |

All five notebooks download their datasets from the Hugging Face Hub.
