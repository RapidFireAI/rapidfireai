# FiQA

Financial opinion question answering from [FiQA-2018](https://sites.google.com/view/fiqa/), in the BEIR layout. The FiQA notebooks index the answer passages, answer a sample of the questions, and score retrieval against the relevance judgments.

| File | Contents |
|---|---|
| `corpus.jsonl` | 57,638 answer passages, one JSON object per line with `_id` and `text` (`title` and `metadata` are empty) |
| `queries.jsonl` | 6,648 questions (`_id`, `text`) |
| `qrels.tsv` | 17,110 relevance judgments (`query-id`, `corpus-id`, `score`) covering all 6,648 questions; used for the retrieval metrics |

- Used by [`rf-tutorial-rag-fiqa.ipynb`](../../rf-tutorial-rag-fiqa.ipynb), [`rf-tutorial-rag-fiqa-pgvector.ipynb`](../../rf-tutorial-rag-fiqa-pgvector.ipynb), [`rf-tutorial-rag-fiqa-pinecone.ipynb`](../../rf-tutorial-rag-fiqa-pinecone.ipynb), [`rf-tutorial-optuna-rag-fiqa.ipynb`](../../rf-tutorial-optuna-rag-fiqa.ipynb), [`rf-colab-rag-fiqa-tutorial.ipynb`](../../rf-colab-rag-fiqa-tutorial.ipynb) and the two notebooks in [`trackio/`](../../trackio/).
- The Colab FiQA notebooks also write `corpus_sampled.jsonl` here at run time. It holds only the passages relevant to their sampled questions.
