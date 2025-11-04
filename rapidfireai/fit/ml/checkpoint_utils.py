import copy
import json
import os
from collections.abc import Callable

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoTokenizer
from trl import DPOTrainer, GRPOTrainer, SFTTrainer

from rapidfireai.fit.utils.constants import SHMObjectType
from rapidfireai.fit.utils.datapaths import DataPath
from rapidfireai.fit.utils.shm_manager import SharedMemoryManager
from rapidfireai.fit.utils.trainer_config import TrainerConfig


def move_tensors_to_device(obj, device: torch.device):
    """Recursively move all tensors in a nested structure to device"""
    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=True)
    elif isinstance(obj, dict):
        return {key: move_tensors_to_device(value, device) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_tensors_to_device(item, device) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_tensors_to_device(item, device) for item in obj)
    else:
        return obj


def move_tensors_to_cpu(obj):
    """Recursively move all tensors in a nested structure to CPU"""
    if isinstance(obj, torch.Tensor):
        return obj.cpu().clone()
    elif isinstance(obj, dict):
        return {key: move_tensors_to_cpu(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [move_tensors_to_cpu(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(move_tensors_to_cpu(item) for item in obj)
    else:
        return obj


def ensure_gradient_compatibility(model, use_peft: bool = False):
    """Ensure model parameters have proper gradient settings"""
    model.train()
    if use_peft:
        model.base_model.eval()
        for name, param in model.named_parameters():
            if any(adapter_key in name.lower() for adapter_key in ["lora", "adapter", "modules_to_save"]):
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True
    for n, p in model.named_parameters():
        if "reference" in n:
            p.requires_grad = False
    model.train()
    torch.set_grad_enabled(True)

    return model


def _configure_tokenizer(tokenizer: AutoTokenizer) -> None:
    """Configure tokenizer with proper padding token."""
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})


def create_model_instance(
    model_config: dict,
    create_model_fn: Callable,
    checkpoint_path: str | None = None,
    is_peft: bool = False,
    device: str | None = None,
) -> tuple[nn.Module, AutoTokenizer]:
    """Create a model instance from a model configuration"""
    if device is not None:
        model_config["model_kwargs"]["device_map"] = {"": device}
    if checkpoint_path and not is_peft:
        model_config["model_name"] = checkpoint_path

    model_instance, tokenizer = create_model_fn(model_config)

    if is_peft and checkpoint_path:
        model_instance = PeftModel.from_pretrained(model_instance, checkpoint_path)
    _configure_tokenizer(tokenizer)

    return model_instance, tokenizer


def save_checkpoint_to_shared_memory(
    trainer: SFTTrainer | DPOTrainer | GRPOTrainer, trainer_config: TrainerConfig, shm_manager: SharedMemoryManager
) -> None:
    """Save checkpoint to shared memory"""
    checkpoint = {}

    if hasattr(trainer.model, "peft_config"):
        peft_state_dict = get_peft_model_state_dict(
            trainer.model,
            adapter_name=trainer_config.config_leaf.get("training_args").get("model_adapter_name", "default"),
        )
        checkpoint["state_dict"] = {k: v.cpu().clone() for k, v in peft_state_dict.items()}
        checkpoint["adapter_config"] = trainer.model.peft_config

    if trainer.optimizer is not None:
        checkpoint["optimizer_state"] = move_tensors_to_cpu(trainer.optimizer.state_dict())

    if trainer.lr_scheduler is not None:
        checkpoint["scheduler_state"] = move_tensors_to_cpu(trainer.lr_scheduler.state_dict())

    if hasattr(trainer, "state"):
        checkpoint["trainer_state"] = move_tensors_to_cpu(trainer.state.__dict__.copy())

    if hasattr(trainer, "rng_state"):
        checkpoint["rng_state"] = move_tensors_to_cpu(trainer.rng_state)

    if hasattr(trainer.args, "__dict__"):
        checkpoint["training_args"] = trainer.args.__dict__.copy()

    if hasattr(trainer.model, "generation_config"):
        checkpoint["generation_config"] = trainer.model.generation_config.to_dict()

    if hasattr(trainer, "scaler"):
        checkpoint["scaler"] = move_tensors_to_cpu(trainer.scaler.state_dict())

    if hasattr(trainer.model, "config"):
        config = trainer.model.config
        if hasattr(config, "special_tokens_map"):
            checkpoint["special_tokens_map"] = config.special_tokens_map
        if hasattr(config, "tokenizer_config"):
            checkpoint["tokenizer_config"] = config.tokenizer_config

    shm_manager.save_model_object(trainer_config.run_id, SHMObjectType.CHECKPOINTS, checkpoint)


def load_checkpoint_from_shared_memory(
    trainer_config: TrainerConfig,
    shm_manager: SharedMemoryManager,
    ref: bool = False,
    is_peft: bool = False,
) -> tuple[nn.Module, AutoTokenizer, dict]:
    """Load checkpoint from shared memory"""
    run_id = trainer_config.run_id
    device = "cuda:0"
    base_model = None
    model_id = trainer_config.config_leaf.get("model_name")

    if trainer_config.warm_started_from is not None and not shm_manager.model_exists(run_id):
        shm_manager.create_warm_start_checkpoint(run_id, trainer_config.warm_started_from)

    if is_peft:
        if not shm_manager.model_exists(model_id):
            base_model, tokenizer = create_model_instance(
                trainer_config.config_leaf,
                trainer_config.create_model_fn,
                checkpoint_path=None,
                is_peft=is_peft,
                device=device,
            )
            if model_id is None:
                model_id = base_model.config._name_or_path
            save_model_to_shared_memory(
                base_model,
                tokenizer,
                trainer_config,
                shm_manager,
                SHMObjectType.BASE_MODEL,
                model_id,
            )
        else:
            base_model, tokenizer = load_model_from_shared_memory(
                trainer_config, shm_manager, SHMObjectType.BASE_MODEL, model_id
            )

    if base_model == "" or (not is_peft and not shm_manager.model_exists(run_id)):
        base_model, tokenizer = create_model_instance(
            trainer_config.config_leaf,
            trainer_config.create_model_fn,
            checkpoint_path=None,
            is_peft=is_peft,
            device=device,
        )

    model = base_model
    peft_config = LoraConfig(**trainer_config.config_leaf.get("peft_params", {}))
    if is_peft:
        model = get_peft_model(model, peft_config)

    # Load weights from shared memory
    if trainer_config.completed_steps > 0 or trainer_config.warm_started_from is not None:
        checkpoint = shm_manager.load_model_object(run_id, SHMObjectType.CHECKPOINTS)

        if "adapter_config" in checkpoint and trainer_config.config_leaf.get("trainer_type") == "DPO" and is_peft:
            reference_state_dict = shm_manager.load_model_object(trainer_config.run_id, SHMObjectType.REF_STATE_DICT)
            reference_state_dict = move_tensors_to_device(reference_state_dict, device)
            model.add_adapter(trainer_config.config_leaf.get("training_args", {}).get("ref_adapter_name"), peft_config)
            model.set_adapter(trainer_config.config_leaf.get("training_args", {}).get("model_adapter_name", "default"))
            set_peft_model_state_dict(
                model,
                reference_state_dict,
                adapter_name=trainer_config.config_leaf.get("training_args").get("ref_adapter_name"),
            )
            model.set_adapter(trainer_config.config_leaf.get("training_args").get("model_adapter_name", "default"))

        if checkpoint.get("state_dict"):
            state_dict = {k: v.to(device) for k, v in checkpoint["state_dict"].items()}
            if is_peft:
                set_peft_model_state_dict(
                    model,
                    state_dict,
                    adapter_name=trainer_config.config_leaf.get("training_args", {}).get(
                        "model_adapter_name", "default"
                    ),
                )
                if trainer_config.config_leaf.get("trainer_type") == "DPO" and is_peft:
                    model.set_adapter(
                        trainer_config.config_leaf.get("training_args", {}).get("model_adapter_name", "default")
                    )
            else:
                model.load_state_dict(state_dict)

        elif not is_peft:
            model, tokenizer = load_model_from_shared_memory(
                trainer_config,
                shm_manager,
                SHMObjectType.FULL_MODEL,
                trainer_config.run_id,
            )

    return model, tokenizer


def load_model_from_shared_memory(
    trainer_config: TrainerConfig,
    shm_manager: SharedMemoryManager,
    model_object_type: SHMObjectType,
    model_id: str,
) -> tuple[nn.Module, AutoTokenizer]:
    """Load model from shared memory"""
    model_data = shm_manager.load_model_object(model_id, model_object_type)
    model = copy.deepcopy(model_data[model_object_type])
    tokenizer = model_data["tokenizer"]
    bnb_modules = move_tensors_to_device(model_data["bnb_modules"], device="cuda:0")
    model = get_model_to_device(model, bnb_modules, device="cuda:0")
    return model, tokenizer


def save_model_to_shared_memory(
    model: nn.Module | str,
    tokenizer: AutoTokenizer,
    trainer_config: TrainerConfig,
    shm_manager: SharedMemoryManager,
    model_type: SHMObjectType,
    model_id: str,
) -> None:
    """Save model to shared memory"""
    if model_type != SHMObjectType.FULL_MODEL and shm_manager.model_exists(model_id):
        return
    model_cpu = model.cpu()
    model_data = {model_type.value: model_cpu, "tokenizer": tokenizer}
    shm_manager.save_model_object(model_id, model_type, model_data)


def load_or_create_ref_model(
    model_instance,
    trainer_config: TrainerConfig,
    device: str,
    use_shared_memory: bool,
    shm_manager: SharedMemoryManager,
) -> nn.Module | None:
    """Load or create reference model for DPO training based on configuration"""
    config_leaf = trainer_config.config_leaf
    device = "cuda:0"
    ref_model_name = trainer_config.config_leaf.get("ref_model_config", {}).get("model_name", None)
    model_id = trainer_config.config_leaf.get("model_name")
    ref_model_id = "ref_" + (trainer_config.config_leaf.get("ref_model_config", {}).get("model_name") or model_id)
    if use_shared_memory and shm_manager.model_exists(ref_model_id):
        ref_model_instance, _ = load_model_from_shared_memory(
            trainer_config, shm_manager, SHMObjectType.REF_FULL_MODEL, ref_model_id
        )
    else:
        if ref_model_name is not None:
            ref_model_instance, _ = create_model_instance(
                config_leaf.get("ref_model_config"),
                trainer_config.create_model_fn,
                device=device,
            )
        elif trainer_config.completed_steps == 0:
            ref_model_instance = copy.deepcopy(model_instance)
        save_model_to_shared_memory(
            ref_model_instance,
            None,
            trainer_config,
            shm_manager,
            SHMObjectType.REF_FULL_MODEL,
            ref_model_id,
        )

    return ref_model_instance


def get_model_to_device(model, bnb_modules, device="cuda:0"):
    """Move model from shared memory to specified device with proper BitsAndBytes restoration"""
    for _, param in model.named_parameters():
        if param.data is not None:
            param.data = move_tensors_to_device(param.data, device)

    for name, buffer in model.named_buffers():
        if isinstance(buffer, torch.Tensor) and buffer is not None:
            parent_module = model
            attr_path = name.split(".")

            for attr in attr_path[:-1]:
                parent_module = getattr(parent_module, attr)

            device_buffer = move_tensors_to_device(buffer, device)
            setattr(parent_module, attr_path[-1], device_buffer)

    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue

        try:
            import bitsandbytes as bnb

            bnb_layer_types = [
                bnb.nn.Linear4bit,
                bnb.nn.LinearFP4,
                bnb.nn.LinearNF4,
                bnb.nn.Params4bit,
            ]
        except ImportError:
            continue

        is_bnb_layer = any(isinstance(module, layer_type) for layer_type in bnb_layer_types)

        if is_bnb_layer and name in bnb_modules:
            bnb_attrs = bnb_modules[name]
            weight = module.weight

            if hasattr(weight, "data") and weight.data is not None:
                weight.data = move_tensors_to_device(weight.data, device)

            if "quant_state_data" in bnb_attrs:
                if not hasattr(weight, "quant_state") or weight.quant_state is None:
                    from bitsandbytes.functional import QuantState

                    quant_data = bnb_attrs["quant_state_data"]

                    weight.quant_state = QuantState(absmax=quant_data["absmax"], code=quant_data["code"])

                quant_data = bnb_attrs["quant_state_data"]

                for attr, value in quant_data.items():
                    if isinstance(value, torch.Tensor):
                        value = move_tensors_to_device(value, device)
                    setattr(weight.quant_state, attr, value)
                if "state2_data" in bnb_attrs:
                    for attr, value in bnb_attrs["state2_data"].items():
                        if isinstance(value, torch.Tensor):
                            value = move_tensors_to_device(value, device)
                        setattr(weight.quant_state.state2, attr, value)

            for attr, value in bnb_attrs.items():
                if attr not in [
                    "quant_state_data",
                    "quant_state_class",
                    "weight_class",
                    "state2_data",
                ]:
                    if isinstance(value, torch.Tensor):
                        value = move_tensors_to_device(value, device)
                    setattr(weight, attr, value)

        elif hasattr(module, "weight") and hasattr(module.weight, "data") and module.weight.data is not None:
            if name not in bnb_modules:
                module.weight.data = move_tensors_to_device(module.weight.data, device)

    model = model.to(device)
    return model


def restore_trainer_from_shared_memory(
    trainer: SFTTrainer | DPOTrainer | GRPOTrainer,
    trainer_config: TrainerConfig,
    shm_manager: SharedMemoryManager,
) -> SFTTrainer | DPOTrainer | GRPOTrainer:
    """Restore complete training state to trainer"""
    try:
        device = next(trainer.model.parameters()).device

        if shm_manager.model_exists(trainer_config.run_id):
            training_state = shm_manager.load_model_object(trainer_config.run_id, SHMObjectType.CHECKPOINTS)
        else:
            raise ValueError(f"Training state for run {trainer_config.run_id} not found in shared memory")

        if training_state.get("trainer_state") is not None and hasattr(trainer, "state"):
            trainer_state_dict = training_state["trainer_state"]
            device_trainer_state = move_tensors_to_device(trainer_state_dict, device)
            for key, value in device_trainer_state.items():
                if hasattr(trainer.state, key):
                    setattr(trainer.state, key, value)

        if training_state.get("optimizer_state") is not None and trainer.optimizer is not None:
            optimizer_state = training_state["optimizer_state"]
            device_optimizer_state = move_tensors_to_device(optimizer_state, device)
            trainer.optimizer.load_state_dict(device_optimizer_state)

        if training_state.get("scheduler_state") is not None and trainer.lr_scheduler is not None:
            scheduler_state = training_state["scheduler_state"]
            device_scheduler_state = move_tensors_to_device(scheduler_state, device)
            trainer.lr_scheduler.load_state_dict(device_scheduler_state)

            if hasattr(trainer.state, "global_step"):
                trainer.lr_scheduler._step_count = trainer.state.global_step + 1

        if training_state.get("rng_state") is not None:
            rng_state = training_state["rng_state"]
            device_rng_state = move_tensors_to_device(rng_state, device)
            trainer.rng_state = device_rng_state

        if training_state.get("generation_config") is not None and hasattr(trainer.model, "generation_config"):
            trainer.model.generation_config = type(trainer.model.generation_config)(
                **training_state["generation_config"]
            )
        if training_state.get("scaler") is not None:
            trainer.scaler.load_state_dict(training_state["scaler"])

    except Exception as e:
        print(f"Warning: Error restoring training state: {e}")

    return trainer


def save_checkpoint_to_disk(
    trainer: SFTTrainer | DPOTrainer | GRPOTrainer,
    trainer_config: TrainerConfig,
    first: bool = False,
    last: bool = False,
    completed_steps: int = 0,
) -> None:
    base_run_path = DataPath.base_run_path(trainer_config.run_id)
    if first:
        checkpoint_path = DataPath.initial_checkpoint_path(base_run_path)
    elif last:
        checkpoint_path = DataPath.final_checkpoint_path(base_run_path)
    else:
        checkpoint_path = DataPath.intermediate_checkpoint_path(base_run_path) / "checkpoint"
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    trainer.model.save_pretrained(checkpoint_path)

    trainer_state_path = checkpoint_path / "trainer_state.json"
    trainer_state_dict = trainer.state.__dict__.copy()
    with open(trainer_state_path, "w") as f:
        json.dump(trainer_state_dict, f, indent=2)

    if trainer.optimizer is not None:
        optimizer_path = checkpoint_path / "optimizer.pt"
        optimizer_state = move_tensors_to_cpu(trainer.optimizer.state_dict())
        torch.save(optimizer_state, optimizer_path)

    if trainer.lr_scheduler is not None:
        scheduler_path = checkpoint_path / "scheduler.pt"
        scheduler_state = move_tensors_to_cpu(trainer.lr_scheduler.state_dict())
        torch.save(scheduler_state, scheduler_path)

    if hasattr(trainer, "rng_state"):
        rng_state_path = checkpoint_path / "rng_state.pth"
        torch.save(trainer.rng_state, rng_state_path)


def load_checkpoint_from_disk(
    trainer_config: TrainerConfig, ref: bool = False, is_peft: bool = False
) -> tuple[nn.Module, AutoTokenizer, dict]:
    """Load checkpoint from disk"""
    device = "cuda:0"
    checkpoint_path = None
    if trainer_config.warm_started_from is not None and trainer_config.completed_steps == 0:
        base_run_path = DataPath.base_run_path(trainer_config.warm_started_from)
        checkpoint_path = DataPath.intermediate_checkpoint_path(base_run_path) / "checkpoint"
    elif trainer_config.completed_steps > 0:
        base_run_path = DataPath.base_run_path(trainer_config.run_id)
        checkpoint_path = DataPath.intermediate_checkpoint_path(base_run_path) / "checkpoint"

    model_instance, tokenizer = create_model_instance(
        trainer_config.config_leaf,
        trainer_config.create_model_fn,
        checkpoint_path,
        is_peft=is_peft,
        device=device,
    )
    if is_peft and checkpoint_path is None:
        model_instance = get_peft_model(
            model_instance,
            LoraConfig(**trainer_config.config_leaf.get("peft_params", {})),
        )

    if ref:
        model_instance, tokenizer = create_model_instance(
            trainer_config.config_leaf.get("ref_model_config", {}),
            trainer_config.create_model_fn,
            device=device,
        )

    return model_instance, tokenizer


def restore_trainer_from_disk(
    trainer: SFTTrainer | DPOTrainer | GRPOTrainer, trainer_config: TrainerConfig
) -> SFTTrainer | DPOTrainer | GRPOTrainer:
    """Restore trainer from disk with proper state accumulation"""
    base_run_path = DataPath.base_run_path(trainer_config.run_id)
    checkpoint_path = DataPath.intermediate_checkpoint_path(base_run_path) / "checkpoint"
    device = "cuda:0"

    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.exists():
        with open(trainer_state_path) as f:
            trainer_state_dict = json.load(f)

        for key, value in trainer_state_dict.items():
            if hasattr(trainer.state, key):
                setattr(trainer.state, key, value)

    optimizer_path = checkpoint_path / "optimizer.pt"
    if optimizer_path.exists() and trainer.optimizer is not None:
        optimizer_state = torch.load(optimizer_path, map_location=device)
        model_device = next(trainer.model.parameters()).device
        optimizer_state = move_tensors_to_device(optimizer_state, model_device)
        trainer.optimizer.load_state_dict(optimizer_state)

    lr_scheduler_path = checkpoint_path / "scheduler.pt"
    if lr_scheduler_path.exists() and trainer.lr_scheduler is not None:
        lr_scheduler_state = torch.load(lr_scheduler_path, map_location=device)
        model_device = next(trainer.model.parameters()).device
        lr_scheduler_state = move_tensors_to_device(lr_scheduler_state, model_device)
        trainer.lr_scheduler.load_state_dict(lr_scheduler_state)

        if hasattr(trainer.state, "global_step") and trainer.lr_scheduler is not None:
            trainer.lr_scheduler._step_count = trainer.state.global_step + 1

    rng_state_path = checkpoint_path / "rng_state.pth"
    if rng_state_path.exists():
        rng_state = torch.load(rng_state_path, map_location=device, weights_only=False)
        model_device = next(trainer.model.parameters()).device
        rng_state = move_tensors_to_device(rng_state, model_device)
        trainer.rng_state = rng_state

    return trainer


def save_ref_model_to_disk(model_instance: nn.Module, trainer_config: TrainerConfig, ref: bool = False) -> None:
    """Save reference model to disk"""
    base_run_path = DataPath.base_run_path(trainer_config.run_id)
    ref_model_path = DataPath.ref_model_path(base_run_path)
    os.makedirs(ref_model_path, exist_ok=True)

    if hasattr(model_instance, "peft_config"):
        peft_state_dict = get_peft_model_state_dict(model_instance)
        torch.save(peft_state_dict, ref_model_path / "pytorch_model.bin")
    else:
        torch.save(model_instance.state_dict(), ref_model_path / "pytorch_model.bin")
