# SciFact

This folder holds [SciFact](https://github.com/allenai/scifact), a dataset of scientific claims, in the BEIR format. The task is to check each claim against research abstracts and give it one of three labels:

- SUPPORT: the abstracts support the claim.
- CONTRADICT: the abstracts contradict the claim.
- NOINFO: the abstracts don't give enough information.

| File | What's inside |
| --- | --- |
| `corpus.jsonl` | 5,183 research abstracts, each with an `_id`, a `title` and a `text` |
| `queries.jsonl` | 809 claims, each with an `_id` and a `text`. For 505 of them, the `metadata` field lists the abstracts that support or contradict the claim. The notebooks label the other 304 claims NOINFO. |
| `qrels.tsv` | 919 relevance judgments for those 809 claims. Each row has a `query-id`, a `corpus-id` and a `score`. |
| `test-queries.jsonl` | 300 more claims. This folder has no relevance judgments for them. No current tutorial notebook loads this file. |
| `corpus_sampled.jsonl` | 18 abstracts taken from `corpus.jsonl`. No current tutorial notebook loads this file. |

The 3 SciFact notebooks use this data: [`rf-tutorial-scifact-full-evaluation.ipynb`](../../rf-tutorial-scifact-full-evaluation.ipynb), [`rf-tutorial-scifact-generators.ipynb`](../../rf-tutorial-scifact-generators.ipynb) and [`rf-tutorial-optuna-rag-scifact.ipynb`](../../rf-tutorial-optuna-rag-scifact.ipynb). Each one evaluates a random sample of the 809 claims.
