# FiQA

This folder holds [FiQA-2018](https://sites.google.com/view/fiqa/), a dataset of financial opinion questions and answers, in the BEIR format. The FiQA notebooks index the answer passages, answer a sample of the questions, and check how well their search finds the relevant passages.

| File | What's inside |
| --- | --- |
| `corpus.jsonl` | 57,638 answer passages, one JSON object per line. Each has an `_id` and a `text`. The `title` and `metadata` fields are empty. |
| `queries.jsonl` | 6,648 questions, each with an `_id` and a `text` |
| `qrels.tsv` | 17,110 relevance judgments that cover all 6,648 questions. Each row has a `query-id`, a `corpus-id` and a `score`. The notebooks use them to score how well retrieval works. |

- These notebooks use this data: [`rf-tutorial-rag-fiqa.ipynb`](../../rf-tutorial-rag-fiqa.ipynb), [`rf-tutorial-rag-fiqa-pgvector.ipynb`](../../rf-tutorial-rag-fiqa-pgvector.ipynb), [`rf-tutorial-rag-fiqa-pinecone.ipynb`](../../rf-tutorial-rag-fiqa-pinecone.ipynb), [`rf-tutorial-optuna-rag-fiqa.ipynb`](../../rf-tutorial-optuna-rag-fiqa.ipynb), [`rf-colab-rag-fiqa-tutorial.ipynb`](../../rf-colab-rag-fiqa-tutorial.ipynb) and the two notebooks in [`trackio/`](../../trackio/).
- When the Colab FiQA notebooks run, they also create `corpus_sampled.jsonl` in this folder. It holds only the passages that are relevant to the questions they sampled.
