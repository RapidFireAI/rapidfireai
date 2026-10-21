# RapidFire AI tutorial notebooks

End-to-end examples of RapidFire AI on public datasets and models. Each notebook runs several configs side by side: fine-tuning and post-training runs with `run_fit()`, and RAG or context-engineering pipelines with `run_evals()`. The [tutorials page](https://oss-docs.rapidfire.ai/en/latest/tutorials.html) in the docs walks through each use case.

| Folder | Contents | Setup |
|---|---|---|
| [`fine-tuning/`](fine-tuning/) | SFT of a customer-support chatbot, in single-GPU, lite, FSDP, Optuna, Colab and Trackio variants | `rapidfireai init --train`, then `rapidfireai start` |
| [`post-training/`](post-training/) | DPO preference alignment and GRPO math reasoning | `rapidfireai init --train`, then `rapidfireai start` |
| [`rag-contexteng/`](rag-contexteng/) | RAG and context-engineering evals on FiQA, SciFact, GSM8K and MMDocIR, plus the bundled [`datasets/`](rag-contexteng/datasets/) | `rapidfireai init`, `rapidfireai start`, then `rapidfireai jupyter` |

## File names

- `rf-tutorial-*` runs on your own machine, with the `rapidfireai` you installed.
- `rf-colab-*` runs on Google Colab and installs a pinned `rapidfireai` version itself.
- `-lite` is a smaller version of a notebook, with smaller models, fewer configs or less data. The single-GPU SFT, DPO and GRPO notebooks need a GPU with at least 80 GB; their `-lite` versions are for smaller GPUs.
- `-fsdp` shards each model across several GPUs with FSDP.
- `optuna` uses `RFOptuna` adaptive search and needs `pip install optuna`.
- Notebooks in a `trackio/` folder log metrics to Trackio.

## Running the notebooks

- `rapidfireai init` copies this folder to `./tutorial_notebooks`, or to `$RF_TUTORIAL_PATH` if it is set. Use that copy: it matches your installed version, while GitHub `main` can be ahead of the release on PyPI.
- Open a notebook in Jupyter or in an IDE with notebook support, such as VS Code, and pick the virtual environment where you installed `rapidfireai` as its kernel. Running a notebook as a script (`python notebook.ipynb`) does not work.
- Open the RAG notebooks from `rapidfireai jupyter`.
- The Llama models are gated on Hugging Face. Request access on each model's page, then log in with `hf auth login`.
- To install RapidFire AI, see [Install and Get Started](https://oss-docs.rapidfire.ai/en/latest/walkthrough.html) in the docs or the [main README](https://github.com/RapidFireAI/rapidfireai#install-and-get-started).

## Try it on Colab

No local install is needed for these two:

- [SFT on Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/fine-tuning/rf-colab-tensorboard-tutorial.ipynb) (`fine-tuning/rf-colab-tensorboard-tutorial.ipynb`)
- [RAG on Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb) (`rag-contexteng/rf-colab-rag-fiqa-tutorial.ipynb`)

## For contributors

`setup/bump_version.sh` copies the `rf-colab-*` notebooks and `rag-contexteng/datasets/` into this folder from `tests/staging/tutorial_notebooks/` at each release. Edit them there, or the next release overwrites your change.
