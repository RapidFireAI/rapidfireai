# SFT with Trackio

These notebooks run the customer-support chatbot example from [`fine-tuning/`](../) and log the metrics to [Trackio](https://github.com/gradio-app/trackio). Trackio is a lightweight experiment tracker from Hugging Face. By default, it keeps your data on your own machine.

To set up, run `rapidfireai init --train`, then `rapidfireai start`.

| Notebook | What it does | Models | What you need |
| --- | --- | --- | --- |
| [`rf-tutorial-sft-trackio.ipynb`](rf-tutorial-sft-trackio.ipynb) | Turns on Trackio (`RF_TRACKIO_ENABLED=true`) and turns off MLflow and TensorBoard before it imports RapidFire. Compares 4 configs: 2 learning rates, each with 2 LoRA adapter sizes. | TinyLlama-1.1B-Chat-v1.0 | A GPU |
| [`rf-colab-tutorial-sft-trackio.ipynb`](rf-colab-tutorial-sft-trackio.ipynb) ([open in Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/trackio/rf-colab-tutorial-sft-trackio.ipynb)) | The Colab version. It shows the Trackio dashboard inside the notebook. | Qwen3-0.6B | A Colab runtime with a GPU |

To see the runs from the local notebook, run this command in a terminal:

```bash
trackio show --project "exp1-sft-trackio-demo"
```
