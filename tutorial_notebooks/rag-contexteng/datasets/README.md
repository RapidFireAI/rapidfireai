# Datasets for the RAG tutorials

This folder holds the data for the notebooks in [`rag-contexteng/`](../). The notebooks load it with the relative path `datasets/` (or `../datasets/` from `trackio/`), so keep this folder next to them.

`fiqa/` and `scifact/` use the [BEIR](https://github.com/beir-cellar/beir) format, which has three files:

- `corpus.jsonl`: the documents to search
- `queries.jsonl`: the questions or claims
- `qrels.tsv`: which documents are relevant to each query

| Folder | What's inside | Used by |
| --- | --- | --- |
| [`fiqa/`](fiqa/) | FiQA-2018, a set of financial questions and answers: 57,638 answer passages and 6,648 questions | All the FiQA notebooks, including the Colab and Trackio ones |
| [`scifact/`](scifact/) | SciFact, a set of scientific claims to check: 5,183 research abstracts and 809 claims | The 3 SciFact notebooks |
| [`gsm8k/`](gsm8k/) | 8,792 worked solutions to GSM8K math problems | No current tutorial notebook loads them. |

[`rf-tutorial-MMDocIR.ipynb`](../rf-tutorial-MMDocIR.ipynb) doesn't use data from this folder. When it runs, it downloads about 1 GB of PDFs into a new `mmdocir/` folder here.
