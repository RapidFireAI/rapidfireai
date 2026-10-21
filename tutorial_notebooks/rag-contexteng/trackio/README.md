# RAG with Trackio

The FiQA RAG use case from [`rag-contexteng/`](../), with metrics logged to [Trackio](https://github.com/gradio-app/trackio), a lightweight, local-first experiment tracker from Hugging Face. Set up with `rapidfireai init`, run `rapidfireai start`, then open the notebook from `rapidfireai jupyter`.

| Notebook | What it shows | Models | Needs |
|---|---|---|---|
| [`rf-tutorial-rag-fiqa-trackio.ipynb`](rf-tutorial-rag-fiqa-trackio.ipynb) | FiQA RAG on 200 questions with 8 configs, logged to Trackio only (MLflow and TensorBoard off); view the runs with `trackio show --project "exp1-fiqa-rag-trackio"` | all-MiniLM-L6-v2, ms-marco-MiniLM-L6-v2 reranker, Qwen2.5-0.5B-Instruct and Qwen2.5-3B-Instruct on vLLM | GPU |
| [`rf-colab-tutorial-rag-fiqa-trackio.ipynb`](rf-colab-tutorial-rag-fiqa-trackio.ipynb) ([open in Colab](https://colab.research.google.com/github/RapidFireAI/rapidfireai/blob/main/tutorial_notebooks/rag-contexteng/trackio/rf-colab-tutorial-rag-fiqa-trackio.ipynb)) | Colab version with 4 configs on a 0.1% question sample, showing the Trackio dashboard in the notebook | all-MiniLM-L6-v2, ms-marco-MiniLM-L6-v2, Qwen2.5-0.5B-Instruct | Colab GPU runtime |

The local notebook reads its data from [`../datasets/`](../datasets/).
