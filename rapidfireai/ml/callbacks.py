from typing import Callable, Dict, List, Optional

import torch
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy, SaveStrategy


class GenerationMetricsCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer,
        eval_dataset: Dataset,
        generation_config: Optional[Dict] = None,
        compute_metrics: Callable = None,
        batch_size: int = 8,
        mlflow_manager=None,
        mlflow_run_id: str = None,
        completed_steps: int = 0,
    ):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        self.batch_size = batch_size
        self.generation_config = generation_config or {
            "max_new_tokens": 128,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.9,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        self.mlflow_manager = mlflow_manager
        self.mlflow_run_id = mlflow_run_id
        self.completed_steps = completed_steps

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        model = kwargs.get("model")
        if model is None:
            return

        metrics = self._compute_generation_metrics(model, state.global_step)

        # Ensure metrics are added to log history
        if hasattr(state, "log_history") and state.log_history:
            state.log_history[-1].update(metrics)
        else:
            # If no log history exists, create a new entry
            if not hasattr(state, "log_history"):
                state.log_history = []
            state.log_history.append(metrics)

        for key, value in metrics.items():
            if self.mlflow_manager:
                self.mlflow_manager.log_metric(
                    self.mlflow_run_id,
                    key,
                    value,
                    step=self.completed_steps + state.global_step,
                )

    def _prepare_data(self, eval_dataset: Dataset) -> tuple:
        """Prepare batch data for generation"""
        input_texts = []
        references = []

        for item in eval_dataset:
            if isinstance(item, dict):
                if "input" in item and "output" in item:
                    input_text = item["input"]
                    reference = item["output"]
                elif "prompt" in item and "completion" in item:
                    input_text = item["prompt"]
                    reference = item["completion"][-1]["content"]
                    input_text = self.tokenizer.apply_chat_template(
                        input_text, tokenize=False
                    )
                else:
                    continue

                input_texts.append(input_text)
                references.append(reference)

        return input_texts, references

    def _generate_batch(self, model, input_texts: List[str]) -> List[str]:
        """Generate text for a batch of inputs"""
        # Tokenize batch
        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Adjust based on your model's context length
        ).to(model.device)

        return inputs["input_ids"]

    def _compute_generation_metrics(self, model, step: int) -> Dict[str, float]:
        """Generate text and compute BLEU/ROUGE metrics with batch processing"""
        model.eval()

        # Determine evaluation samples
        eval_size = len(self.eval_dataset)
        indices = list(range(eval_size))

        predictions = []
        references = []

        # Process in batches
        input_texts, batch_references = self._prepare_data(self.eval_dataset)
        input_ids = self._generate_batch(model, input_texts)
        with torch.no_grad():
            for i in tqdm(
                range(0, len(indices), self.batch_size), desc="Generating for metrics"
            ):
                input_ids_batch = input_ids[i : i + self.batch_size]
                with torch.inference_mode(), torch.cuda.amp.autocast():
                    outputs_batch = model.generate(
                        input_ids_batch, **self.generation_config
                    )
                generated_texts = self.tokenizer.batch_decode(
                    outputs_batch[:, input_ids_batch.shape[1] :],
                    skip_special_tokens=True,
                )
                predictions.extend(generated_texts)
                references.extend(batch_references[i : i + self.batch_size])

        # Compute metrics
        metrics = {}
        try:
            if self.compute_metrics and predictions:
                metrics = self.compute_metrics((predictions, references))
        except Exception:
            return {}

        # Cleanup
        del predictions, references
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        return metrics


class MLflowLoggingCallback(TrainerCallback):
    """Callback for logging metrics to MLflow during training"""

    def __init__(
        self,
        mlflow_manager,
        mlflow_run_id: str,
        excluded_keys: list = None,
        completed_steps: int = 0,
        chunk_id: int = 0,
        num_epochs_completed: int = 0,
    ):
        self.mlflow_manager = mlflow_manager
        self.mlflow_run_id = mlflow_run_id
        self.completed_steps = completed_steps
        self.excluded_keys = excluded_keys or [
            "step",
            "epoch",
        ]
        self.chunk_id = chunk_id
        self.num_epochs_completed = num_epochs_completed

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ):
        """Called when the trainer logs metrics"""
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key not in self.excluded_keys:
                    try:
                        self.mlflow_manager.log_metric(
                            self.mlflow_run_id,
                            key,
                            value,
                            step=self.completed_steps + state.global_step,
                        )
                    except Exception as e:
                        print(f"Warning: Failed to log metric {key} to MLflow: {e}")
            if "eval_loss" not in logs and "train_runtime" not in logs:
                self.mlflow_manager.log_metric(
                    self.mlflow_run_id,
                    "chunk number",
                    self.chunk_id,
                    step=self.completed_steps + state.global_step,
                )
                self.mlflow_manager.log_metric(
                    self.mlflow_run_id,
                    "num_epochs_completed",
                    self.num_epochs_completed,
                    step=self.completed_steps + state.global_step,
                )


class LogLevelCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def __init__(self, global_step_args: Dict):
        self.eval_first_step = global_step_args.get("eval_first_step", 0)
        self.actual_steps = global_step_args.get("actual_steps", 0)
        self.log_first_step = global_step_args.get("log_first_step", 0)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Log
        control.should_log = False
        control.should_evaluate = False
        if state.global_step == 1 and args.logging_first_step:
            control.should_log = True
        if args.logging_strategy == IntervalStrategy.STEPS and (
            self.log_first_step <= state.global_step
            and (state.global_step - self.log_first_step) % state.logging_steps == 0
        ):
            control.should_log = True

        # Evaluate
        if args.eval_strategy == IntervalStrategy.STEPS and (
            self.eval_first_step <= state.global_step
            and (state.global_step - self.eval_first_step) % state.eval_steps == 0
        ):
            control.should_evaluate = True
        # Save
        if (
            args.save_strategy == SaveStrategy.STEPS
            and state.save_steps > 0
            and state.global_step % state.save_steps == 0
        ):
            control.should_save = True

        # End training
        if state.global_step >= state.max_steps:
            control.should_training_stop = True
            # Save the model at the end if we have a save strategy
            if args.save_strategy == SaveStrategy.STEPS:
                control.should_save = True

        return control

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        # Log
        if args.logging_strategy == IntervalStrategy.EPOCH:
            control.should_log = True

        # Evaluate
        if (
            args.eval_strategy == IntervalStrategy.EPOCH
            and args.eval_delay <= state.epoch
        ):
            control.should_evaluate = True

        # Save
        if args.save_strategy == SaveStrategy.EPOCH:
            control.should_save = True

        return control
