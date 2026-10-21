# Datasets for the RAG tutorials

Data for the notebooks in [`rag-contexteng/`](../). They open it through the relative path `datasets/` (`../datasets/` from `trackio/`), so keep this folder next to them. `fiqa/` and `scifact/` use the [BEIR](https://github.com/beir-cellar/beir) layout: `corpus.jsonl` holds the documents, `queries.jsonl` the questions or claims, and `qrels.tsv` the relevant documents for each query.

| Folder | Contents | Used by |
|---|---|---|
| [`fiqa/`](fiqa/) | FiQA-2018 financial opinion Q&A: 57,638 answer passages and 6,648 questions | Every FiQA notebook, including the Colab and Trackio ones |
| [`scifact/`](scifact/) | SciFact scientific claim verification: 5,183 abstracts and 809 claims | The 3 SciFact notebooks |
| [`gsm8k/`](gsm8k/) | 8,792 GSM8K worked solutions | No current tutorial notebook loads them |

[`rf-tutorial-MMDocIR.ipynb`](../rf-tutorial-MMDocIR.ipynb) downloads about 1 GB of PDFs into `mmdocir/` in this folder when it runs; they are not bundled here.
