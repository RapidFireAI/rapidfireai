# Fine-tuning tutorials

Supervised fine-tuning (SFT) of a customer-support chatbot on the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) with LoRA adapters, scored with ROUGE-L (plus BLEU outside Colab). Set up with `rapidfireai init --train`, then run `rapidfireai start`. The [SFT page](https://oss-docs.rapidfire.ai/en/latest/sft.html) in the docs walks through the use case.

| Notebook | What it shows | Models | Needs |
|---|---|---|---|
| [`rf-tutorial-sft-chatqa.ipynb`](rf-tutorial-sft-chatqa.ipynb) | 2 models × 2 LoRA adapters (r=16, r=128) = 4 configs | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 | GPU with ≥80 GB; Hugging Face access to Llama |
| [`rf-tutorial-sft-chatqa-lite.ipynb`](rf-tutorial-sft-chatqa-lite.ipynb) | Same use case on one small model: 2 learning rates × 2 LoRA adapters (r=8, r=32) = 4 configs | TinyLlama-1.1B-Chat-v1.0 | GPU with <80 GB |
| [`rf-tutorial-sft-chatqa-fsdp-lite.ipynb`](rf-tutorial-sft-chatqa-fsdp-lite.ipynb) | FSDP across 2 GPUs: 2 learning rates × 2 LoRA adapters = 4 configs | TinyLlama-1.1B-Chat-v1.0 | ≥2 A10 GPUs (48 GB total) |
| [`rf-tutorial-sft-chatqa-fsdp.ipynb`](rf-tutorial-sft-chatqa-fsdp.ipynb) | FSDP across 4 GPUs: 2 LoRA adapters = 2 configs | Qwen3-32B | ≥4 A10 GPUs (96 GB total) |
| [`rf-tutorial-sft-chatqa-fsdp-large.ipynb`](rf-tutorial-sft-chatqa-fsdp-large.ipynb) | FSDP across 8 GPUs: 2 LoRA adapters = 2 configs | Meta-Llama-3-70B-Instruct | ≥8 A10 GPUs (192 GB total); Hugging Face access to Llama |
| [`rf-tutorial-optuna-sft-chatqa-tiny.ipynb`](rf-tutorial-optuna-sft-chatqa-tiny.ipynb) | `RFOptuna` with the TPE sampler starts 4 configs, prunes weak runs after each shard and suggests replacements, up to 8 in total; maximizes ROUGE-L | TinyLlama-1.1B-Chat-v1.0 | GPU; `pip install optuna` |
| [`rf-colab-tensorboard-tutorial.ipynb`](rf-colab-tensorboard-tutorial.ipynb) ([open in Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-colab-tensorboard-tutorial.ipynb)) | Simplified SFT on Colab, with TensorBoard shown in the notebook | Qwen3-0.6B | Colab GPU runtime |

- The FSDP notebooks train on small data samples for demo purposes. They run on A10s, but the docs recommend A100s or newer with NVLink for reasonable run times.
- [`trackio/`](trackio/) has the same use case with metrics logged to Trackio.
