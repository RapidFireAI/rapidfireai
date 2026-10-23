# GSM8K worked solutions

This folder holds worked solutions from [GSM8K](https://huggingface.co/datasets/openai/gsm8k), a dataset of grade-school math word problems.

| File | What's inside |
| --- | --- |
| `cheat_sheet_0.txt` to `cheat_sheet_17.txt` | 8,792 worked solutions, without the questions. Each solution explains the steps in words, marks each calculation like `<<5*30=150>>`, and ends with `#### <answer>`. No current tutorial notebook loads these files. |

[`rf-tutorial-gsm8k-fewshot.ipynb`](../../rf-tutorial-gsm8k-fewshot.ipynb) doesn't use these files. It downloads `openai/gsm8k` from the Hugging Face Hub instead, and its 10 worked examples are written in the notebook itself.
