# RAG with Trackio

These notebooks run the FiQA RAG example from [`rag-contexteng/`](../) and log the metrics to [Trackio](https://github.com/gradio-app/trackio). Trackio is a lightweight experiment tracker from Hugging Face. By default, it keeps your data on your own machine.

To set up, run `rapidfireai init`, then `rapidfireai start`. Then open the notebook from `rapidfireai jupyter`.

| Notebook | What it does | Models | What you need |
| --- | --- | --- | --- |
| [`rf-tutorial-rag-fiqa-trackio.ipynb`](rf-tutorial-rag-fiqa-trackio.ipynb) | Runs FiQA RAG on 200 questions and compares 8 configs. It logs metrics to Trackio only, with MLflow and TensorBoard turned off. | all-MiniLM-L6-v2 (embeddings), ms-marco-MiniLM-L6-v2 (reranker), Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct (generators, served with vLLM) | A GPU |
| [`rf-colab-tutorial-rag-fiqa-trackio.ipynb`](rf-colab-tutorial-rag-fiqa-trackio.ipynb) ([open in Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/trackio/rf-colab-tutorial-rag-fiqa-trackio.ipynb)) | The Colab version. It compares 4 configs on 0.1% of the questions, and shows the Trackio dashboard inside the notebook. | all-MiniLM-L6-v2, ms-marco-MiniLM-L6-v2, Qwen2.5-0.5B-Instruct | A Colab runtime with a GPU |

The local notebook loads its data from [`../datasets/`](../datasets/). To see its runs, run this command in a terminal:

```bash
trackio show --project "exp1-fiqa-rag-trackio"
```
