# GSM8K worked solutions

Worked solutions from [GSM8K](https://huggingface.co/datasets/openai/gsm8k), a dataset of grade-school math word problems.

| File | Contents |
|---|---|
| `cheat_sheet_0.txt` to `cheat_sheet_17.txt` | 8,792 GSM8K worked solutions: step-by-step text with calculator annotations such as `<<5*30=150>>`, each ending in `#### <answer>`. Solutions only, no questions. No current tutorial notebook loads them. |

[`rf-tutorial-gsm8k-fewshot.ipynb`](../../rf-tutorial-gsm8k-fewshot.ipynb) downloads `openai/gsm8k` from the Hugging Face Hub instead and keeps its 10 few-shot examples in the notebook.
