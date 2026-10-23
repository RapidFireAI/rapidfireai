# RapidFire AI tutorial notebooks

These notebooks show how to use RapidFire AI from start to finish, with public datasets and models. Each notebook runs several configurations (configs) side by side so you can compare their results. The fine-tuning and post-training notebooks train models with `run_fit()`. The RAG and context-engineering notebooks evaluate pipelines with `run_evals()`.

For a step-by-step guide to each use case, see the [tutorials page](https://oss-docs.rapidfire.ai/en/latest/tutorials.html) in the docs.

| Folder | What's inside | How to set up |
| --- | --- | --- |
| [`fine-tuning/`](fine-tuning/) | Supervised fine-tuning (SFT) of a customer-support chatbot. There are versions for one large GPU, for smaller GPUs, for several GPUs (FSDP), for automatic tuning (Optuna), for Google Colab and for Trackio. | Run `rapidfireai init --train`, then `rapidfireai start` |
| [`post-training/`](post-training/) | Training a model to prefer better answers (DPO), and training a model to solve math problems (GRPO) | Run `rapidfireai init --train`, then `rapidfireai start` |
| [`rag-contexteng/`](rag-contexteng/) | Comparing RAG and context-engineering pipelines on the FiQA, SciFact, GSM8K and MMDocIR datasets. The data for these notebooks is in [`rag-contexteng/datasets/`](rag-contexteng/datasets/). | Run `rapidfireai init`, then `rapidfireai start`, then `rapidfireai jupyter` |

## What the file names mean

- `rf-tutorial-*` notebooks run on your own machine, with the `rapidfireai` version you installed.
- `rf-colab-*` notebooks run on Google Colab. They install a specific `rapidfireai` version themselves.
- `-lite` marks a smaller version of a notebook. It uses smaller models, fewer configs or less data. The full single-GPU SFT, DPO and GRPO notebooks need a GPU with at least 80 GB of memory. Their `-lite` versions run on smaller GPUs.
- `-fsdp` marks notebooks that split each model across several GPUs, using FSDP (Fully Sharded Data Parallel).
- `optuna` marks notebooks where `RFOptuna` chooses the configs for you. Install Optuna first with `pip install optuna`.
- Notebooks in a `trackio/` folder log their metrics to Trackio.

## How to run the notebooks

- `rapidfireai init` copies this folder to `./tutorial_notebooks`, or to the path in `$RF_TUTORIAL_PATH` if you set it. Use that copy, because it matches the version you installed. The notebooks on GitHub `main` can be newer than the latest release on PyPI.
- Open a notebook in Jupyter, or in an editor that supports notebooks, such as VS Code. Select the virtual environment where you installed `rapidfireai` as the kernel. You can't run a notebook as a Python script (`python notebook.ipynb`).
- Open the RAG notebooks in the Jupyter server that `rapidfireai jupyter` starts.
- The Llama models are gated on Hugging Face, which means you need permission to download them. Request access on each model's Hugging Face page, then log in with `hf auth login`.
- To install RapidFire AI, see [Install and Get Started](https://oss-docs.rapidfire.ai/en/latest/walkthrough.html) in the docs or the [main README](https://github.com/RapidFireAI/rapidfireai#install-and-get-started).

## Try it on Google Colab

These two notebooks run on Colab, so you don't need to install anything on your machine:

- [SFT on Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-colab-tensorboard-tutorial.ipynb) (`fine-tuning/rf-colab-tensorboard-tutorial.ipynb`)
- [RAG on Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb) (`rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb`)

## For contributors

At each release, `setup/bump_version.sh` copies the `rf-colab-*` notebooks and the `rag-contexteng/datasets/` folder into this folder from `tests/staging/tutorial_notebooks/`. To change those files, edit the copies in `tests/staging/tutorial_notebooks/`. Otherwise, the next release overwrites your change.
