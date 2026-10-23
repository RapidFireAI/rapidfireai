# Post-training tutorials

These notebooks train a model further after supervised fine-tuning, with two trainers from the Hugging Face TRL library:

- DPO (Direct Preference Optimization) teaches a model to prefer the better of two answers.
- GRPO (Group Relative Policy Optimization) teaches a model to solve math problems. The model writes several answers to each question, and reward functions score them.

To set up, run `rapidfireai init --train`, then `rapidfireai start`. The [DPO](https://oss-docs.rapidfire.ai/en/latest/dpo.html) and [GRPO](https://oss-docs.rapidfire.ai/en/latest/grpo.html) pages in the docs explain each use case step by step.

| Notebook | What it does | Models | What you need |
| --- | --- | --- | --- |
| [`rf-tutorial-dpo-alignment.ipynb`](rf-tutorial-dpo-alignment.ipynb) | Runs DPO on 500 examples from `trl-lib/ultrafeedback_binarized`. Compares 4 configs that change the LoRA rank (64 or 128), the beta value and the loss function (sigmoid, robust, or a weighted mix of sigmoid, BCO-pair and SFT). | `rapidfire-ai-inc/mistral-7b-sft-bnb-4bit`: Mistral-7B-Instruct-v0.3 already fine-tuned with SFT and stored in 4-bit | A GPU with at least 80 GB of memory |
| [`rf-tutorial-dpo-alignment-lite.ipynb`](rf-tutorial-dpo-alignment-lite.ipynb) | A smaller version: 3 of those configs, with LoRA rank 16, 128 examples and shorter sequences | The same model as above | A smaller GPU, with less than 80 GB of memory, is enough. |
| [`rf-tutorial-dpo-alignment-fsdp-lite.ipynb`](rf-tutorial-dpo-alignment-fsdp-lite.ipynb) | The 3 lite configs, with the model split across 2 GPUs using FSDP | TinyLlama-1.1B-Chat-v1.0 | At least 2 A10 GPUs (48 GB in total) |
| [`rf-tutorial-grpo-mathreasoning.ipynb`](rf-tutorial-grpo-mathreasoning.ipynb) | Runs GRPO on the `openai/gsm8k` math problems. 5 reward functions score each answer: whether it's correct, whether it's a whole number, whether it strictly or loosely follows the expected format, and how many of the expected XML tags it uses. Compares 4 configs. | Llama-3.1-8B-Instruct, Qwen2.5-3B-Instruct (with 2 learning rates) and Qwen2.5-7B-Instruct, all loaded in 4-bit | A GPU with at least 80 GB of memory. Access to Llama on Hugging Face. |
| [`rf-tutorial-grpo-mathreasoning-lite.ipynb`](rf-tutorial-grpo-mathreasoning-lite.ipynb) | The same task with 2 small models that aren't loaded in 4-bit. Compares 2 configs. | Qwen2.5-0.5B-Instruct, Llama-3.2-1B-Instruct | A smaller GPU, with less than 80 GB of memory, is enough. Access to Llama on Hugging Face. |

All five notebooks download their datasets from the Hugging Face Hub when they run.
