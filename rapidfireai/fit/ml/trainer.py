import math
import os

import torch
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from transformers.utils.logging import set_verbosity_error
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer
from rapidfireai.utils.constants import RF_TRAINER_OUTPUT

from rapidfireai.fit.ml.callbacks import GenerationMetricsCallback, LogLevelCallback, MetricLoggingCallback
from rapidfireai.fit.ml.checkpoint_utils import (
    ensure_gradient_compatibility,
    load_checkpoint_from_disk,
    load_checkpoint_from_shared_memory,
    load_or_create_ref_model,
    move_tensors_to_cpu,
    move_tensors_to_device,
    restore_trainer_from_disk,
    restore_trainer_from_shared_memory,
)
from rapidfireai.fit.utils.constants import SHMObjectType
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager
from rapidfireai.fit.utils.trainer_config import TrainerConfig

set_verbosity_error()


def create_trainer_instance(
    trainer_config: TrainerConfig,
    shm_manager: SharedMemoryManager,
    use_shared_memory: bool = False,
    metric_logger=None,
    chunk_id: int = 0,
) -> tuple[SFTTrainer | DPOTrainer | GRPOTrainer | None, str]:
    """
    Create a trainer instance with proper state restoration.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(trainer_config.worker_id)
    device = "cuda:0"

    trainer = None
    config_leaf = trainer_config.config_leaf
    trainer_type = config_leaf.get("trainer_type", "SFT")
    training_args = config_leaf.get("training_args", {})
    additional_trainer_kwargs = config_leaf.get("additional_kwargs", {})
    compute_metrics = additional_trainer_kwargs.get("compute_metrics", None)

    # Configure training arguments
    training_args, global_step_args = _configure_training_args(training_args, trainer_config)
    trainer_config_obj = _create_trainer_config_object(trainer_type, training_args)
    # check if peft params is empty dict
    is_peft = bool(config_leaf.get("peft_params"))
    # Load model and tokenizer
    if use_shared_memory:
        model_instance, tokenizer = load_checkpoint_from_shared_memory(trainer_config, shm_manager, is_peft=is_peft)
    else:
        model_instance, tokenizer = load_checkpoint_from_disk(trainer_config, is_peft=is_peft)
    # add model name to model config
    config_leaf["model_name"] = model_instance.config._name_or_path

    # Handle reference model for DPO
    ref_model_instance = None
    if config_leaf.get("trainer_type") == "DPO":
        model_instance, ref_model_instance = _setup_reference_model(
            model_instance,
            trainer_config,
            config_leaf,
            use_shared_memory,
            shm_manager,
            device,
            is_peft,
        )

    model_instance = model_instance.to(device)

    trainer_kwargs, formatting_func, additional_trainer_kwargs = _prepare_trainer_kwargs(
        model_instance,
        trainer_config_obj,
        tokenizer,
        trainer_config,
        additional_trainer_kwargs,
        ref_model_instance,
        config_leaf,
    )

    callbacks, additional_trainer_kwargs = _setup_callbacks(  # FIXME: avoid returning additional_trainer_kwargs
        metric_logger,
        trainer_config,
        chunk_id,
        compute_metrics,
        additional_trainer_kwargs,
        tokenizer,
        training_args,
        formatting_func,
        global_step_args,
    )

    if callbacks:
        trainer_kwargs["callbacks"] = callbacks

    trainer_kwargs.update(additional_trainer_kwargs)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if v is not None}

    trainer = _create_trainer_by_type(trainer_type, trainer_kwargs, trainer_config, use_shared_memory, shm_manager)
    return trainer, config_leaf["model_name"]


def _configure_training_args(training_args: dict, trainer_config: TrainerConfig) -> dict:
    """Configure training arguments with default values."""
    completed_steps = trainer_config.completed_steps
    per_device_train_batch_size = training_args.get("per_device_train_batch_size", 1)
    gradient_accumulation_steps = training_args.get("gradient_accumulation_steps", 1)
    len_dataloader = math.ceil(trainer_config.train_dataset.num_rows / per_device_train_batch_size)
    steps_per_epoch = max(
        len_dataloader // gradient_accumulation_steps + int(len_dataloader % gradient_accumulation_steps > 0),
        1,
    )

    if trainer_config.config_leaf.get("trainer_type", "SFT") == "GRPO":
        num_generations = training_args.get("num_generations", 8)
        steps_per_epoch = (num_generations * trainer_config.train_dataset.num_rows) // (
            gradient_accumulation_steps * per_device_train_batch_size
        )
    left_over_steps = trainer_config.total_steps - completed_steps
    if left_over_steps > steps_per_epoch:
        training_args["num_train_epochs"] = 1
        training_args.pop("max_steps", None)
    else:
        training_args["max_steps"] = left_over_steps
        training_args.pop("num_train_epochs", None)

    eval_first_step = 0
    global_step_args = {}
    actual_steps = min(left_over_steps, steps_per_epoch)
    if training_args.get("eval_steps") is not None:
        eval_steps = training_args.get("eval_steps")
        eval_first_step = eval_steps - (completed_steps % eval_steps)
        global_step_args["eval_first_step"] = eval_first_step
    log_first_step = 0
    if training_args.get("logging_steps") is not None:
        logging_steps = training_args.get("logging_steps")
        log_first_step = logging_steps - (completed_steps % logging_steps)
        global_step_args["log_first_step"] = log_first_step
    global_step_args["actual_steps"] = actual_steps

    if training_args.get("eval_on_start", False) and completed_steps > 0:
        training_args.pop("eval_on_start")
    if training_args.get("logging_first_step", False) and completed_steps > 0:
        training_args.pop("logging_first_step")

    training_args["save_strategy"] = "no"
    training_args["do_train"] = True
    training_args["do_eval"] = True
    training_args["dataloader_pin_memory"] = False
    training_args["no_cuda"] = False
    training_args["local_rank"] = -1
    training_args["disable_tqdm"] = True
    if training_args.get("output_dir") is None:
        training_args["output_dir"] = RF_TRAINER_OUTPUT
    if training_args.get("report_to") is None:
        training_args["report_to"] = "none"

    if "save_steps" in training_args:
        training_args.pop("save_steps")

    return training_args, global_step_args


def _create_trainer_config_object(trainer_type: str, training_args: dict):
    """Create the appropriate trainer config object based on trainer type."""
    if trainer_type == "SFT":
        return SFTConfig(**training_args)
    elif trainer_type == "DPO":
        return DPOConfig(**training_args)
    elif trainer_type == "GRPO":
        return GRPOConfig(**training_args)
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")


def _setup_reference_model(
    model_instance,
    trainer_config,
    config_leaf,
    use_shared_memory,
    shm_manager,
    device,
    is_peft,
):
    """Setup reference model for DPO training."""
    ref_model_instance = None
    training_args = config_leaf.get("training_args", {})
    if is_peft and not training_args.get("force_use_ref_model", False):
        model_adapter_name = training_args.get("model_adapter_name", "default")
        ref_adapter_name = training_args.get("ref_adapter_name", "reference")

        if model_adapter_name is not None and ref_adapter_name is not None:
            if use_shared_memory:
                peft_config = LoraConfig(**config_leaf["peft_params"])
                if trainer_config.completed_steps == 0 and trainer_config.warm_started_from is None:
                    reference_state_dict = get_peft_model_state_dict(model_instance)
                    reference_state_dict = move_tensors_to_cpu(reference_state_dict)
                    shm_manager.save_model_object(
                        trainer_config.run_id,
                        SHMObjectType.REF_STATE_DICT,
                        reference_state_dict,
                    )
                else:
                    reference_state_dict = shm_manager.load_model_object(
                        trainer_config.run_id, SHMObjectType.REF_STATE_DICT
                    )
                    reference_state_dict = move_tensors_to_device(reference_state_dict, device)
                model_instance.add_adapter(ref_adapter_name, peft_config)
                model_instance.set_adapter(ref_adapter_name)
                set_peft_model_state_dict(model_instance, reference_state_dict, adapter_name=ref_adapter_name)
                model_instance.set_adapter(model_adapter_name)
            else:
                base_run_path = DataPath.base_run_path(trainer_config.run_id)
                ref_model_path = DataPath.ref_model_path(base_run_path)
                reference_adapter_path = ref_model_path / "reference"

                if not reference_adapter_path.exists():
                    os.makedirs(reference_adapter_path, exist_ok=True)
                    model_instance.save_pretrained(reference_adapter_path)
                torch.cuda.empty_cache()
                model_instance.load_adapter(
                    reference_adapter_path,
                    adapter_name=ref_adapter_name,
                    device_map={"": device},
                )
                model_instance.set_adapter(model_adapter_name)
            model_instance = model_instance.to(device)
    else:
        ref_model_instance = load_or_create_ref_model(
            model_instance, trainer_config, device, use_shared_memory, shm_manager
        )
        ref_model_instance = ref_model_instance.to(device)
    return model_instance, ref_model_instance


def _prepare_trainer_kwargs(
    model_instance,
    trainer_config_obj,
    tokenizer,
    trainer_config,
    additional_trainer_kwargs,
    ref_model_instance,
    config_leaf,
):
    """Prepare keyword arguments for trainer creation."""
    if config_leaf.get("trainer_type") == "DPO  ":
        model_instance = ensure_gradient_compatibility(
            model_instance, hasattr(model_instance, "peft_config")
        )  # FIXME: change function for DPO
    trainer_kwargs = {
        "model": model_instance,
        "args": trainer_config_obj,
        "processing_class": tokenizer,
    }

    train_dataset = trainer_config.train_dataset
    eval_dataset = trainer_config.eval_dataset
    formatting_func = None

    if additional_trainer_kwargs.get("formatting_func") is not None:
        formatting_func = additional_trainer_kwargs.get("formatting_func")
        train_dataset = train_dataset.map(formatting_func)  # FIXME: add try exception with batched/unbatched
        if eval_dataset is not None:
            eval_dataset = eval_dataset.map(formatting_func)
        additional_trainer_kwargs_copy = additional_trainer_kwargs.copy()
        additional_trainer_kwargs_copy.pop("formatting_func")
        additional_trainer_kwargs = additional_trainer_kwargs_copy

    trainer_kwargs["train_dataset"] = train_dataset
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset

    if config_leaf.get("trainer_type") == "DPO" and ref_model_instance is not None:
        trainer_kwargs["ref_model"] = ref_model_instance

    if config_leaf.get("trainer_type") == "GRPO":
        reward_funcs = config_leaf.get("reward_funcs")
        if reward_funcs is not None:
            trainer_kwargs["reward_funcs"] = reward_funcs

    return trainer_kwargs, formatting_func, additional_trainer_kwargs


def _setup_callbacks(
    metric_logger,
    trainer_config,
    chunk_id,
    compute_metrics,
    additional_trainer_kwargs,
    tokenizer,
    training_args,
    formatting_func,
    global_step_args,
):
    """Setup callbacks for the trainer."""
    callbacks = []

    if metric_logger is not None and trainer_config.metric_run_id is not None:
        metric_callback = MetricLoggingCallback(
            metric_logger=metric_logger,
            metric_run_id=trainer_config.metric_run_id,
            completed_steps=trainer_config.completed_steps,
            chunk_id=chunk_id,
            num_epochs_completed=trainer_config.num_epochs_completed
        )
        callbacks.append(metric_callback)

    if compute_metrics is not None and additional_trainer_kwargs.get("generation_config") is not None:
        compute_metrics_function = compute_metrics
        if formatting_func is not None:
            formatted_eval_dataset = trainer_config.eval_dataset.map(formatting_func)
        else:
            formatted_eval_dataset = trainer_config.eval_dataset

        generation_callback = GenerationMetricsCallback(
            tokenizer=tokenizer,
            eval_dataset=formatted_eval_dataset,
            generation_config=additional_trainer_kwargs.get("generation_config"),
            compute_metrics=compute_metrics_function,
            batch_size=training_args.get("per_device_eval_batch_size"),
            metric_logger=metric_logger,
            metric_run_id=trainer_config.metric_run_id,
            completed_steps=trainer_config.completed_steps,
        )
        callbacks.append(generation_callback)
        additional_trainer_kwargs.pop("generation_config")
        additional_trainer_kwargs.pop("compute_metrics")
        callbacks.append(LogLevelCallback(global_step_args=global_step_args))

    return callbacks, additional_trainer_kwargs


def _create_trainer_by_type(trainer_type, trainer_kwargs, trainer_config, use_shared_memory, shm_manager):
    """Create trainer instance based on type with proper state restoration."""
    if trainer_type == "SFT":
        dummy_trainer = SFTTrainer(**trainer_kwargs)
        dummy_trainer.create_optimizer_and_scheduler(num_training_steps=trainer_config.total_steps)
        trainer = SFTTrainer(
            **trainer_kwargs,
            optimizers=(dummy_trainer.optimizer, dummy_trainer.lr_scheduler),
        )
        del dummy_trainer

    elif trainer_type == "DPO":
        dummy_trainer = DPOTrainer(**trainer_kwargs)
        dummy_trainer.create_optimizer_and_scheduler(num_training_steps=trainer_config.total_steps)
        trainer = DPOTrainer(
            **trainer_kwargs,
            optimizers=(dummy_trainer.optimizer, dummy_trainer.lr_scheduler),
        )
        del dummy_trainer

    elif trainer_type == "GRPO":
        dummy_trainer = GRPOTrainer(**trainer_kwargs)
        dummy_trainer.create_optimizer_and_scheduler(num_training_steps=trainer_config.total_steps)
        trainer = GRPOTrainer(
            **trainer_kwargs,
            optimizers=(dummy_trainer.optimizer, dummy_trainer.lr_scheduler),
        )
        del dummy_trainer
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")

    if trainer_config.completed_steps > 0:
        if use_shared_memory:
            trainer = restore_trainer_from_shared_memory(trainer, trainer_config, shm_manager)
        else:
            trainer = restore_trainer_from_disk(trainer, trainer_config)

    return trainer
