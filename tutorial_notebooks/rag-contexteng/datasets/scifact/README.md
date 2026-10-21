# SciFact

Scientific claim verification from [SciFact](https://github.com/allenai/scifact), in the BEIR layout. Each claim is checked against research abstracts and labeled SUPPORT, CONTRADICT or NOINFO.

| File | Contents |
|---|---|
| `corpus.jsonl` | 5,183 research abstracts (`_id`, `title`, `text`) |
| `queries.jsonl` | 809 claims (`_id`, `text`). For 505 of them, `metadata` maps cited abstracts to SUPPORT or CONTRADICT evidence; the notebooks label the other 304 NOINFO. |
| `qrels.tsv` | 919 relevance judgments (`query-id`, `corpus-id`, `score`) for those 809 claims |
| `test-queries.jsonl` | 300 more claims, with no qrels in this folder. No current tutorial notebook loads it. |
| `corpus_sampled.jsonl` | 18 abstracts sampled from `corpus.jsonl`. No current tutorial notebook loads it. |

Used by [`rf-tutorial-scifact-full-evaluation.ipynb`](../../rf-tutorial-scifact-full-evaluation.ipynb), [`rf-tutorial-scifact-generators.ipynb`](../../rf-tutorial-scifact-generators.ipynb) and [`rf-tutorial-optuna-rag-scifact.ipynb`](../../rf-tutorial-optuna-rag-scifact.ipynb), which each evaluate a shuffled sample of the 809 claims.
