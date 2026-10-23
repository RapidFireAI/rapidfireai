# Fine-tuning tutorials

These notebooks use supervised fine-tuning (SFT) to train a customer-support chatbot on the [Bitext customer support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset). Instead of changing the whole model, they train LoRA adapters: small sets of extra weights added to the model. Each notebook compares several configurations (configs) and scores the chatbot's answers with ROUGE-L. The notebooks that don't run on Colab also report BLEU.

To set up, run `rapidfireai init --train`, then `rapidfireai start`. The [SFT page](https://oss-docs.rapidfire.ai/en/latest/sft.html) in the docs explains this use case step by step.

| Notebook | What it does | Models | What you need |
| --- | --- | --- | --- |
| [`rf-tutorial-sft-chatqa.ipynb`](rf-tutorial-sft-chatqa.ipynb) | Compares 4 configs: 2 models, each with a small (r=16) and a large (r=128) LoRA adapter | Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3 | A GPU with at least 80 GB of memory. Access to Llama on Hugging Face. |
| [`rf-tutorial-sft-chatqa-lite.ipynb`](rf-tutorial-sft-chatqa-lite.ipynb) | The same chatbot with one small model. Compares 4 configs: 2 learning rates, each with 2 LoRA adapter sizes (r=8 and r=32). | TinyLlama-1.1B-Chat-v1.0 | A smaller GPU, with less than 80 GB of memory, is enough. |
| [`rf-tutorial-sft-chatqa-fsdp-lite.ipynb`](rf-tutorial-sft-chatqa-fsdp-lite.ipynb) | Splits the model across 2 GPUs with FSDP. Compares 4 configs: 2 learning rates, each with 2 LoRA adapter sizes. | TinyLlama-1.1B-Chat-v1.0 | At least 2 A10 GPUs (48 GB in total) |
| [`rf-tutorial-sft-chatqa-fsdp.ipynb`](rf-tutorial-sft-chatqa-fsdp.ipynb) | Splits the model across 4 GPUs with FSDP. Compares 2 configs with different LoRA adapters. | Qwen3-32B | At least 4 A10 GPUs (96 GB in total) |
| [`rf-tutorial-sft-chatqa-fsdp-large.ipynb`](rf-tutorial-sft-chatqa-fsdp-large.ipynb) | Splits the model across 8 GPUs with FSDP. Compares 2 configs with different LoRA adapters. | Meta-Llama-3-70B-Instruct | At least 8 A10 GPUs (192 GB in total). Access to Llama on Hugging Face. |
| [`rf-tutorial-optuna-sft-chatqa-tiny.ipynb`](rf-tutorial-optuna-sft-chatqa-tiny.ipynb) | `RFOptuna` chooses the configs. It starts with 4, stops weak runs early, and tries new configs in their place, up to 8 in total. It looks for the highest ROUGE-L score. | TinyLlama-1.1B-Chat-v1.0 | A GPU. Install Optuna first with `pip install optuna`. |
| [`rf-colab-tensorboard-tutorial.ipynb`](rf-colab-tensorboard-tutorial.ipynb) ([open in Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-colab-tensorboard-tutorial.ipynb)) | A simpler SFT example for Google Colab. It shows the TensorBoard charts inside the notebook. | Qwen3-0.6B | A Colab runtime with a GPU |

- The FSDP notebooks train on small samples of the data to keep the demo short. They work on A10 GPUs, but the docs recommend A100 or newer GPUs connected with NVLink, so that training doesn't take too long.
- The [`trackio/`](trackio/) folder has the same chatbot example, with metrics logged to Trackio.
