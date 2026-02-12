from collections.abc import Callable

import torch
import torch.distributed as dist
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
        generation_config: dict | None = None,
        compute_metrics: Callable = None,
        batch_size: int = 8,
        metric_logger=None,
        metric_run_id: str = None,
        completed_steps: int = 0,
        use_fsdp: bool = False,
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
        self.metric_logger = metric_logger
        self.metric_run_id = metric_run_id
        self.completed_steps = completed_steps
        self.use_fsdp = use_fsdp

        # Always force use_cache=True (model may have it disabled for FSDP training)
        self.generation_config["use_cache"] = True

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
            step = self.completed_steps + state.global_step
            if self.metric_logger:
                self.metric_logger.log_metric(
                    self.metric_run_id,
                    key,
                    value,
                    step=step,
                )

    def _prepare_data(self, eval_dataset: Dataset) -> tuple:
        """Prepare batch data for generation with defensive validation"""
        input_texts = []
        references = []

        for item in eval_dataset:
            if not isinstance(item, dict):
                continue

            input_text = None
            reference = None

            # Support multiple field name patterns
            if "input" in item and "output" in item:
                input_text = item["input"]
                reference = item["output"]
            elif "prompt" in item and "completion" in item:
                input_text = item["prompt"]
                reference = item["completion"][-1]["content"]
                input_text = self.tokenizer.apply_chat_template(
                    input_text, tokenize=False
                )
            elif "text" in item:
                # SFT format - use text as input, response as reference
                input_text = item["text"]
                reference = item.get("response", item.get("instruction", item["text"]))
            elif "instruction" in item and "response" in item:
                # Direct instruction/response format
                input_text = item["instruction"]
                reference = item["response"]

            # Validate non-empty strings
            if input_text and isinstance(input_text, str) and input_text.strip():
                if reference and isinstance(reference, str) and reference.strip():
                    input_texts.append(input_text.strip())
                    references.append(reference.strip())

        # Return safe empty values to prevent downstream errors
        if not input_texts:
            return [], []

        return input_texts, references

    def _generate_batch(self, model, input_texts: list[str]) -> torch.Tensor:
        """Generate text for a batch of inputs with defensive validation"""
        # Defensive validation for empty inputs
        if not input_texts:
            return torch.empty((0, 0), dtype=torch.long).to(model.device)

        try:
            # Tokenize batch
            inputs = self.tokenizer(
                input_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,  # Adjust based on your model's context length
            ).to(model.device)

            return inputs["input_ids"]
        except Exception as e:
            # Log error and return empty tensor to prevent crash
            print(f"Warning: Tokenization error in generation callback: {e}")
            return torch.empty((0, 0), dtype=torch.long).to(model.device)


    def _run_generation_loop(
        self,
        model,
        input_ids: torch.Tensor,
        batch_references: list[str],
        device,
        override_batch_size: int | None = None,
    ) -> tuple[list[str], list[str]]:
        """Run the autoregressive generation loop over batches and return predictions + references."""
        predictions = []
        references = []
        num_samples = input_ids.shape[0]
        batch_size = override_batch_size or self.batch_size

        with torch.no_grad():
            for i in tqdm(
                range(0, num_samples, batch_size), desc="Generating for metrics"
            ):
                input_ids_batch = input_ids[i : i + batch_size].to(device)
                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    outputs_batch = model.generate(
                        input_ids_batch, **self.generation_config
                    )
                generated_texts = self.tokenizer.batch_decode(
                    outputs_batch[:, input_ids_batch.shape[1] :],
                    skip_special_tokens=True,
                )
                predictions.extend(generated_texts)
                references.extend(batch_references[i : i + batch_size])

        return predictions, references

    def _compute_generation_metrics(self, model, step: int) -> dict[str, float]:
        """Generate text and compute BLEU/ROUGE metrics with batch processing.

        For FSDP: all ranks participate in generation (required for collective
        ops), but only rank 0 computes text metrics. Results are broadcast.
        """
        model.eval()

        # Prepare data
        input_texts, batch_references = self._prepare_data(self.eval_dataset)

        # Early return if no valid data
        if not input_texts:
            print("Warning: No valid eval data for generation metrics")
            return {}

        input_ids = self._generate_batch(model, input_texts)

        # Check for empty generation batch
        if input_ids.numel() == 0:
            print("Warning: Empty input_ids from tokenization")
            return {}

        # Enable KV cache for generation (model may have use_cache=False for FSDP training)
        # Must set BOTH model.config and model.generation_config so that
        # model.generate() actually activates the cache in forward()
        original_config_use_cache = getattr(model.config, "use_cache", False)
        original_genconfig_use_cache = getattr(
            getattr(model, "generation_config", None), "use_cache", None
        )
        model.config.use_cache = True
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.use_cache = True

        # Use a larger batch size for generation (training batch_size=1 is wasteful here)
        gen_batch_size = max(self.batch_size, 4) if self.use_fsdp else self.batch_size

        try:
            if self.use_fsdp:
                metrics = self._compute_metrics_fsdp(
                    model, input_ids, batch_references, gen_batch_size
                )
            else:
                metrics = self._compute_metrics_standard(
                    model, input_ids, batch_references
                )
        finally:
            # Restore original use_cache settings for training
            model.config.use_cache = original_config_use_cache
            if (
                hasattr(model, "generation_config")
                and model.generation_config is not None
                and original_genconfig_use_cache is not None
            ):
                model.generation_config.use_cache = original_genconfig_use_cache

        return metrics

    def _compute_metrics_standard(
        self,
        model,
        input_ids: torch.Tensor,
        batch_references: list[str],
    ) -> dict[str, float]:
        """Non-FSDP generation path."""
        predictions, references = self._run_generation_loop(
            model, input_ids, batch_references, model.device
        )

        metrics = {}
        try:
            if self.compute_metrics and predictions:
                metrics = self.compute_metrics((predictions, references))
        except Exception:
            return {}

        del predictions, references
        return metrics

    def _compute_metrics_fsdp(
        self,
        model,
        input_ids: torch.Tensor,
        batch_references: list[str],
        gen_batch_size: int = 4,
    ) -> dict[str, float]:
        """FSDP-aware generation: all ranks participate in model.generate()
        (required -- FSDP forward is a collective op), but only rank 0
        computes BLEU/ROUGE to avoid redundant work. Metrics are broadcast
        to all ranks afterward."""
        rank = dist.get_rank() if dist.is_initialized() else 0

        # All ranks MUST call model.generate() together -- FSDP forward
        # uses all-gather/reshard which are collective NCCL operations.
        predictions, references = self._run_generation_loop(
            model,
            input_ids,
            batch_references,
            model.device,
            override_batch_size=gen_batch_size,
        )

        # Only rank 0 computes the expensive text metrics (BLEU/ROUGE)
        metrics = {}
        if rank == 0:
            try:
                if self.compute_metrics and predictions:
                    metrics = self.compute_metrics((predictions, references))
            except Exception:
                metrics = {}

        del predictions, references

        # Broadcast metrics from rank 0 to all other ranks
        if dist.is_initialized():
            metrics_list = [metrics]
            dist.broadcast_object_list(metrics_list, src=0)
            metrics = metrics_list[0]
            dist.barrier()

        return metrics


class MetricLoggingCallback(TrainerCallback):
    """Callback for logging metrics to tracking backend during training"""

    def __init__(
        self,
        metric_logger,
        metric_run_id: str,
        excluded_keys: list = None,
        completed_steps: int = 0,
        chunk_id: int = 0,
        num_epochs_completed: int = 0,
    ):
        self.metric_logger = metric_logger
        self.metric_run_id = metric_run_id
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
            step = self.completed_steps + state.global_step
            for key, value in logs.items():
                if isinstance(value, (int, float)) and key not in self.excluded_keys:
                    try:
                        if self.metric_logger:
                            self.metric_logger.log_metric(
                                self.metric_run_id,
                                key,
                                value,
                                step=step,
                            )
                    except Exception as e:
                        print(
                            f"Warning: Failed to log metric {key} to tracking backend: {e}"
                        )
            if "eval_loss" not in logs and "train_runtime" not in logs:
                if self.metric_logger:
                    self.metric_logger.log_metric(
                        self.metric_run_id,
                        "chunk number",
                        self.chunk_id,
                        step=step,
                    )
                    self.metric_logger.log_metric(
                        self.metric_run_id,
                        "num_epochs_completed",
                        self.num_epochs_completed,
                        step=step,
                    )


class LogLevelCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """

    def __init__(self, global_step_args: dict):
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
