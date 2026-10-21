# SFT with Trackio

The customer-support SFT use case from [`fine-tuning/`](../), with metrics logged to [Trackio](https://github.com/gradio-app/trackio), a lightweight, local-first experiment tracker from Hugging Face. Set up with `rapidfireai init --train`, then run `rapidfireai start`.

| Notebook | What it shows | Models | Needs |
|---|---|---|---|
| [`rf-tutorial-sft-trackio.ipynb`](rf-tutorial-sft-trackio.ipynb) | Sets `RF_TRACKIO_ENABLED=true` and turns MLflow and TensorBoard off before importing RapidFire; 2 learning rates × 2 LoRA adapters = 4 configs | TinyLlama-1.1B-Chat-v1.0 | GPU |
| [`rf-colab-tutorial-sft-trackio.ipynb`](rf-colab-tutorial-sft-trackio.ipynb) ([open in Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/trackio/rf-colab-tutorial-sft-trackio.ipynb)) | Colab version that shows the Trackio dashboard in the notebook | Qwen3-0.6B | Colab GPU runtime |

To view the local notebook's runs, run `trackio show --project "exp1-sft-trackio-demo"` in a terminal.
