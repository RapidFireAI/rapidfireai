import math
import os
import warnings

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
    move_tensors_to_device,
    restore_trainer_from_disk,
    restore_trainer_from_shared_memory,
)
from rapidfireai.fit.utils.constants import SHMObjectType
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager
from rapidfireai.fit.utils.trainer_config import TrainerConfig

set_verbosity_error()


def create_rf_trainer(trainer_type: str, trainer_config=None, shm_manager=None, use_fsdp=False, **kwargs):
    """
    Factory function to create a custom trainer that uses total_steps from trainer_config and integrates with state restoration.
    """
    base_trainer_map = {
        "SFT": SFTTrainer,
        "DPO": DPOTrainer,
        "GRPO": GRPOTrainer,
    }

    if trainer_type not in base_trainer_map:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")

    base_trainer_class = base_trainer_map[trainer_type]

    class RFTrainer(base_trainer_class):
        """Custom trainer that uses total_steps from trainer_config for scheduler setup."""

        def __init__(self, trainer_config=None, shm_manager=None, use_fsdp=False, **kwargs):
            super().__init__(**kwargs)
            self.trainer_config = trainer_config
            self.shm_manager = shm_manager
            self.use_fsdp = use_fsdp
            self._pending_optimizer_state = None
            self._pending_scheduler_state = None
            if trainer_config.completed_steps > 0 or trainer_config.warm_started:
                training_state = shm_manager.load_model_object(trainer_config.run_id, SHMObjectType.CHECKPOINTS)
                self._pending_optimizer_state = training_state["optimizer_state"]
                self._pending_scheduler_state = training_state["scheduler_state"]

        def restore_optimizer_state(self):
            """Restore optimizer state for Single GPU trainer"""
            device = next(self.model.parameters()).device
            try:
                if hasattr(self, "_pending_optimizer_state") and self._pending_optimizer_state is not None:
                    optimizer_state = self._pending_optimizer_state
                    device_optimizer_state = move_tensors_to_device(optimizer_state, device)
                    self.optimizer.load_state_dict(device_optimizer_state)
                    delattr(self, "_pending_optimizer_state")
            except Exception as e:
                print(f"Warning: Error restoring FSDP scheduler state: {e}")

        def restore_fsdp_optimizer_state(self, rank=0):
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

            fsdp_plugin = self.accelerator.state.fsdp_plugin
            model = self.model
            from rapidfireai.utils.distributed_utils import barrier
            barrier()

            ctx = (
                FSDP.state_dict_type(
                    model, fsdp_plugin.state_dict_type, fsdp_plugin.state_dict_config, fsdp_plugin.optim_state_dict_config
                )
            )
            with ctx:
                if fsdp_plugin.state_dict_type == StateDictType.FULL_STATE_DICT:
                    optim_state = self._pending_optimizer_state
                    
                    flattened_osd = FSDP.optim_state_dict_to_load(model=model, optim=self.optimizer, optim_state_dict=optim_state)
                    self.optimizer.load_state_dict(flattened_osd)
                
                delattr(self, "_pending_optimizer_state")

        def restore_scheduler_state(
            self
        ):
            """Restore scheduler state for FSDP trainer after scheduler is created"""
            device = next(self.model.parameters()).device
            try:
                if hasattr(self, "_pending_scheduler_state") and self._pending_scheduler_state is not None:
                    scheduler_state = self._pending_scheduler_state
                    device_scheduler_state = move_tensors_to_device(scheduler_state, device)
                    self.lr_scheduler.load_state_dict(device_scheduler_state)
                    delattr(self, "_pending_scheduler_state")
            except Exception as e:
                print(f"Warning: Error restoring FSDP scheduler state: {e}")

        def create_optimizer_and_scheduler(self, num_training_steps: int):
            if self.trainer_config is not None:
                num_training_steps = self.trainer_config.total_steps

            self.create_optimizer()
            self.create_scheduler(num_training_steps=num_training_steps, optimizer=self.optimizer)


        def _load_optimizer_and_scheduler(self, resume_from_checkpoint: str | None):
            super()._load_optimizer_and_scheduler(resume_from_checkpoint)
            if self.trainer_config.completed_steps > 0 or self.trainer_config.warm_started:
                if self.use_fsdp:
                    self.restore_fsdp_optimizer_state(rank=self.trainer_config.local_rank) 
                else:   
                    self.restore_optimizer_state()
                self.restore_scheduler_state()

        def _move_model_to_vllm(self):
            # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
            from accelerate.utils import is_peft_model
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from contextlib import nullcontext
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
            gather_if_zero3 = nullcontext() # type: ignore
            if is_peft_model(self.model):
                # Handle FSDP + PEFT separately
                if self.is_fsdp_enabled:
                    # For FSDP + PEFT, we need to use FSDP's summon_full_params to gather the model
                    with FSDP.summon_full_params(self.model, writeback=False):
                        # Now the model is fully gathered, we can merge adapters
                        self.model.merge_adapter()
                        
                        # Sync weights to vLLM while parameters are gathered
                        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                        fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                        
                        # For PEFT models, we need to extract the merged weights directly
                        for name, param in self.model.named_parameters():
                            # Clean up the parameter name for vLLM
                            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                            if self.model.prefix in name:
                                continue
                            if "original_module" in name:
                                continue
                            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default.", "_fsdp_wrapped_module."])
                            
                            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                                self.vllm_client.update_named_param(name, param.data)
                            elif self.vllm_mode == "colocate":
                                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                                llm_model.load_weights([(name, param.data)])
                        
                        # Unmerge adapters while parameters are still gathered
                        self.model.unmerge_adapter()
                        # Parameters will automatically be repartitioned when exiting the context
                
                # Handle DeepSpeed ZeRO-3 + PEFT
                elif zero_stage_3:
                    with gather_if_zero3(list(self.model.parameters())):
                        self.model.merge_adapter()
                        
                        # DeepSpeed ZeRO-3 with PEFT
                        for name, param in self.model.named_parameters():
                            # When using PEFT, we need to recover the original parameter name and discard some parameters
                            name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                            if self.model.prefix in name:
                                continue
                            # When module to save, remove its prefix and discard the original module
                            if "original_module" in name:
                                continue
                            name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])

                            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                                self.vllm_client.update_named_param(name, param.data)
                            elif self.vllm_mode == "colocate":
                                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                                llm_model.load_weights([(name, param.data)])
                                
                        # Unmerge adapters while parameters are still gathered
                        self.model.unmerge_adapter()
                
                # Handle regular PEFT without distributed training
                else:
                    self.model.merge_adapter()
                    
                    for name, param in self.model.named_parameters():
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        if "original_module" in name:
                            continue
                        name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
                        
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                            llm_model.load_weights([(name, param.data)])
                    
                    self.model.unmerge_adapter()
            
            else:
                # For non-PEFT models, simply gather (if needed) and update each parameter individually.
                if self.is_fsdp_enabled:
                    fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                    fsdp_version = getattr(fsdp_plugin, "fsdp_version", 1) if fsdp_plugin else 1
                    if fsdp_version == 1:
                        self._sync_fsdp1_params_to_vllm(self.model)  # use memory-efficient post-order traversal for FSDP
                    elif fsdp_version == 2:
                        self._sync_fsdp2_params_to_vllm(self.model)
                else:
                    for name, param in self.model.named_parameters():
                        name = self._fix_param_name_to_vllm(name)
                        with gather_if_zero3([param]):
                            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                                self.vllm_client.update_named_param(name, param.data)
                            elif self.vllm_mode == "colocate":
                                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                                llm_model.load_weights([(name, param.data)])

            # Reset cache on vLLM
            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.reset_prefix_cache()
            elif self.vllm_mode == "colocate":
                self.llm.reset_prefix_cache()

    return RFTrainer(trainer_config=trainer_config, shm_manager=shm_manager, use_fsdp=use_fsdp, **kwargs)


def create_trainer_instance(
    trainer_config: TrainerConfig,
    shm_manager: SharedMemoryManager,
    use_shared_memory: bool = False,
    metric_logger=None,
    chunk_id: int = 0,
    use_fsdp: bool = False,
) -> tuple[SFTTrainer | DPOTrainer | GRPOTrainer | None, str]:
    """
    Create a trainer instance with proper state restoration.
    """
    if not use_fsdp:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(trainer_config.worker_id)

    # Set device based on distributed training
    device = "cpu" if use_fsdp else "cuda:0"

    trainer = None
    config_leaf = trainer_config.config_leaf
    trainer_type = config_leaf.get("trainer_type", "SFT")
    training_args = config_leaf.get("training_args", {})
    additional_trainer_kwargs = config_leaf.get("additional_kwargs", {})
    compute_metrics = additional_trainer_kwargs.get("compute_metrics", None)

    # Configure training arguments
    training_args, global_step_args = _configure_training_args(training_args, trainer_config, use_fsdp=use_fsdp)
    trainer_config_obj = _create_trainer_config_object(trainer_type, training_args)
    # check if peft params is empty dict
    is_peft = bool(config_leaf.get("peft_params"))
    # Load model and tokenizer
    if use_shared_memory:
        model_instance, tokenizer = load_checkpoint_from_shared_memory(
            trainer_config, shm_manager, device, is_peft=is_peft, use_fsdp=use_fsdp
        )
    else:
        model_instance, tokenizer = load_checkpoint_from_disk(trainer_config, device, is_peft=is_peft)
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
            use_fsdp=use_fsdp,
        )

    if model_instance.device.type != "meta":  # model got loaded in meta device
        model_instance = model_instance.to(device)

    trainer_kwargs, formatting_func, additional_trainer_kwargs = _prepare_trainer_kwargs(
        model_instance,
        trainer_config_obj,
        tokenizer,
        trainer_config,
        additional_trainer_kwargs,
        ref_model_instance,
        config_leaf,
        use_fsdp=use_fsdp,
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

    trainer = _create_trainer_by_type(
        trainer_type, trainer_kwargs, trainer_config, use_shared_memory, shm_manager, device, use_fsdp=use_fsdp
    )
    return trainer, config_leaf["model_name"]


def _configure_training_args(training_args: dict, trainer_config: TrainerConfig, use_fsdp: bool = False) -> dict:
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
    if use_fsdp:
        steps_per_epoch = steps_per_epoch//trainer_config.world_size
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

    # Configure distributed training arguments for FSDP
    if use_fsdp:
        training_args["local_rank"] = trainer_config.local_rank
        # training_args["dataloader_num_workers"] = trainer_config.world_size  # Avoid multiprocessing issues with FSDP
        training_args["dataloader_pin_memory"] = False
        training_args["remove_unused_columns"] = False  # FSDP requires this

    return training_args, global_step_args


def _create_trainer_config_object(trainer_type: str, training_args: dict):
    """Create the appropriate trainer config object based on trainer type."""
    # Filter out parameters that are not accepted by TRL config classes
    # no_cuda is a TrainingArguments parameter but not accepted by DPOConfig/GRPOConfig
    filtered_args = training_args.copy()
    if trainer_type in ["DPO", "GRPO"]:
        filtered_args.pop("no_cuda", None)
    
    if trainer_type == "SFT":
        return SFTConfig(**filtered_args)
    elif trainer_type == "DPO":
        return DPOConfig(**filtered_args)
    elif trainer_type == "GRPO":
        return GRPOConfig(**filtered_args)
    else:
        raise ValueError(f"Unsupported trainer type: {trainer_type}")


def _setup_reference_model(
    model_instance, trainer_config, config_leaf, use_shared_memory, shm_manager, device, is_peft, use_fsdp=False
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
                    if not use_fsdp or trainer_config.local_rank == 0:
                        reference_state_dict = move_tensors_to_device(reference_state_dict, "cpu")
                        shm_manager.save_model_object(
                            trainer_config.run_id, SHMObjectType.REF_STATE_DICT, reference_state_dict
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
            model_instance, trainer_config, device, use_shared_memory, shm_manager, use_fsdp=use_fsdp
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
    use_fsdp=False,
):
    """Prepare keyword arguments for trainer creation."""
    if config_leaf.get("trainer_type") == "DPO":
        model_instance = ensure_gradient_compatibility(
            model_instance, hasattr(model_instance, "peft_config"), use_fsdp=use_fsdp
        )
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
    additional_trainer_kwargs.pop("num_gpus", None)
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


def _create_trainer_by_type(trainer_type, trainer_kwargs, trainer_config, use_shared_memory, shm_manager, device, use_fsdp=False):
    """Create trainer instance based on type with proper state restoration."""
    trainer = create_rf_trainer(
        trainer_type, trainer_config=trainer_config, shm_manager=shm_manager, use_fsdp=use_fsdp, **trainer_kwargs
    )

    if trainer_config.completed_steps > 0:
        if use_shared_memory:
            trainer = restore_trainer_from_shared_memory(trainer, trainer_config, shm_manager, use_fsdp=use_fsdp)
        else:
            trainer = restore_trainer_from_disk(trainer, trainer_config, device)

    return trainer
