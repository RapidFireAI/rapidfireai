from collections.abc import Callable
import gc

import torch
import torch.distributed as dist
from datasets import Dataset
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.trainer_utils import IntervalStrategy, SaveStrategy

from rapidfireai.fit.ml.checkpoint_utils import flush_cuda_cache, purge_model_kv_caches
from rapidfireai.fit.utils.logging import RFLogger


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

        # Let caller/model defaults decide on KV cache (can cause OOM)
        self.generation_config.pop("use_cache", None)

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

        step = self.completed_steps + state.global_step

        try:
            metrics = self._compute_generation_metrics(model, state.global_step)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not isinstance(e, torch.cuda.OutOfMemoryError) and "out of memory" not in str(e).lower():
                raise  # Re-raise non-OOM RuntimeErrors
            model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            return

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

        # Switch model back to training mode
        model.train()

        if self.use_fsdp:
            purge_model_kv_caches(model)
        flush_cuda_cache()

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
            print(f"Warning: Tokenization error in generation callback: {e}")
            return torch.empty((0, 0), dtype=torch.long).to(model.device)

    @staticmethod
    def _truncate_at_eos(
        generated_ids: torch.Tensor,
        eos_token_ids: list[int],
    ) -> torch.Tensor:
        """Replace tokens after the first EOS with pad=0 for clean decoding."""
        generated_ids = generated_ids.clone()
        for eos_id in eos_token_ids:
            mask = generated_ids == eos_id  # (batch, seq_len)
            for row_idx in range(generated_ids.shape[0]):
                eos_positions = mask[row_idx].nonzero(as_tuple=False)
                if eos_positions.numel() > 0:
                    first_eos = eos_positions[0].item()
                    generated_ids[row_idx, first_eos:] = 0
        return generated_ids

    def _run_generation_loop(
        self,
        model,
        input_ids: torch.Tensor,
        batch_references: list[str],
        device,
        override_batch_size: int | None = None,
        generation_config: dict | None = None,
        eos_token_ids: list[int] | None = None,
    ) -> tuple[list[str], list[str]]:
        """Run autoregressive generation over batches and return predictions + references."""
        predictions = []
        references = []
        num_samples = input_ids.shape[0]
        batch_size = override_batch_size or self.batch_size
        gen_config = (
            generation_config
            if generation_config is not None
            else self.generation_config
        )

        with torch.no_grad():
            for i in range(0, num_samples, batch_size):
                input_ids_batch = input_ids[i : i + batch_size].to(device)

                with torch.inference_mode(), torch.amp.autocast("cuda"):
                    outputs_batch = model.generate(input_ids_batch, **gen_config)

                # Keep only newly generated tokens
                new_tokens = outputs_batch[:, input_ids_batch.shape[1] :]

                # Truncate at real EOS (disabled during FSDP generation to keep ranks in sync)
                if eos_token_ids:
                    new_tokens = self._truncate_at_eos(new_tokens, eos_token_ids)

                generated_texts = self.tokenizer.batch_decode(
                    new_tokens,
                    skip_special_tokens=True,
                )
                predictions.extend(generated_texts)
                references.extend(batch_references[i : i + batch_size])

                # Free GPU tensors from this batch immediately
                del outputs_batch, input_ids_batch, new_tokens
                torch.cuda.empty_cache()

        # Final cleanup after all batches - clear any lingering CUDA state
        torch.cuda.empty_cache()

        return predictions, references

    def _compute_generation_metrics(self, model, step: int) -> dict[str, float]:
        """Generate text and compute BLEU/ROUGE metrics."""
        # Free cached memory before generation (more memory-intensive than training)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        model.eval()

        # Prepare data
        input_texts, batch_references = self._prepare_data(self.eval_dataset)

        # Early return if no valid data
        if not input_texts:
            return {}

        input_ids = self._generate_batch(model, input_texts)

        # Check for empty generation batch
        if input_ids.numel() == 0:
            return {}

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
            del input_ids
            torch.cuda.empty_cache()
            gc.collect()

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
        finally:
            # Explicitly delete predictions and references to free memory
            del predictions, references
            gc.collect()

        return metrics

    def _compute_metrics_fsdp(
        self,
        model,
        input_ids: torch.Tensor,
        batch_references: list[str],
        gen_batch_size: int = 4,
    ) -> dict[str, float]:
        """FSDP-aware generation with data sharding across ranks."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        total_eval_samples = input_ids.shape[0]

        # ---- shard the eval data across ranks ----
        samples_per_gpu = (total_eval_samples + world_size - 1) // world_size  # ceil div
        start_idx = rank * samples_per_gpu
        end_idx = min(start_idx + samples_per_gpu, total_eval_samples)

        shard_input_ids = input_ids[start_idx:end_idx]
        shard_references = batch_references[start_idx:end_idx]
        actual_samples_on_this_gpu = shard_input_ids.shape[0]

        # Pad to keep generate() call count identical across ranks (NCCL collective)
        if actual_samples_on_this_gpu < samples_per_gpu:
            pad_count = samples_per_gpu - actual_samples_on_this_gpu
            pad_ids = input_ids[:pad_count]  # reuse first samples as padding
            shard_input_ids = torch.cat([shard_input_ids, pad_ids], dim=0)
            shard_references = shard_references + batch_references[:pad_count]

        # Disable eos_token_id to keep ranks in sync; truncate post-generation
        gen_config = dict(self.generation_config)
        saved_eos = gen_config.pop("eos_token_id", None)

        # Resolve real EOS ids for post-generation truncation
        if saved_eos is not None:
            real_eos_ids = saved_eos if isinstance(saved_eos, list) else [saved_eos]
        elif self.tokenizer.eos_token_id is not None:
            real_eos_ids = [self.tokenizer.eos_token_id]
        else:
            real_eos_ids = None

        # ---- run generation on this rank's shard ----
        predictions, references = self._run_generation_loop(
            model,
            shard_input_ids,
            shard_references,
            model.device,
            override_batch_size=gen_batch_size,
            generation_config=gen_config,
            eos_token_ids=real_eos_ids,
        )

        # Free per-rank GPU shard (predictions already decoded to strings)
        del shard_input_ids
        torch.cuda.empty_cache()

        # Drop padding predictions
        predictions = predictions[:actual_samples_on_this_gpu]
        references = references[:actual_samples_on_this_gpu]

        # ---- gather predictions from all ranks to rank 0 ----
        if dist.is_initialized() and world_size > 1:
            all_predictions: list[list[str] | None] = [None] * world_size
            all_references: list[list[str] | None] = [None] * world_size
            dist.gather_object(
                predictions,
                all_predictions if rank == 0 else None,
                dst=0,
            )
            dist.gather_object(
                references,
                all_references if rank == 0 else None,
                dst=0,
            )
        else:
            all_predictions = [predictions]
            all_references = [references]

        # ---- rank 0 computes expensive text metrics ----
        metrics: dict[str, float] = {}
        if rank == 0:
            flat_preds = [p for shard in all_predictions for p in shard]
            flat_refs = [r for shard in all_references for r in shard]
            # trim to exact original count (discard per-rank padding artefacts)
            flat_preds = flat_preds[:total_eval_samples]
            flat_refs = flat_refs[:total_eval_samples]
            try:
                if self.compute_metrics and flat_preds:
                    metrics = self.compute_metrics((flat_preds, flat_refs))
            except Exception:
                metrics = {}
            finally:
                # Clean up large lists on rank 0
                del flat_preds, flat_refs, all_predictions, all_references

        # Clean up per-rank predictions/references
        del predictions, references
        
        # Force garbage collection across all ranks
        gc.collect()

        # ---- broadcast final metrics from rank 0 to all ranks ----
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
