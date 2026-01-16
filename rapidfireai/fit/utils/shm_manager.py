import copy
import gc
import threading
from multiprocessing import Lock, Manager

import torch

from rapidfireai.fit.utils.constants import SHM_MIN_FREE_SPACE, SHM_WARN_THRESHOLD, SHMObjectType
from rapidfireai.fit.utils.exceptions import InsufficientSharedMemoryException
from rapidfireai.fit.utils.logging import RFLogger


def _get_shm_usage():
    """Get shared memory storage usage information in GiB."""
    import shutil

    stat = shutil.disk_usage("/dev/shm")
    total_gib = stat.total / (1024**3)
    used_gib = stat.used / (1024**3)
    free_gib = stat.free / (1024**3)
    return {
        "total": total_gib,
        "used": used_gib,
        "free": free_gib,
        "percent_used": (stat.used / stat.total) * 100,
    }


def _estimate_tensor_size_gib(obj):
    """Recursively estimate the size of tensors in a nested structure in GiB."""
    if isinstance(obj, torch.Tensor):
        # Calculate size in bytes: numel * element_size
        size_bytes = obj.numel() * obj.element_size()
        return size_bytes / (1024**3)
    elif isinstance(obj, dict):
        return sum(_estimate_tensor_size_gib(v) for v in obj.values())
    elif isinstance(obj, (list, tuple)):
        return sum(_estimate_tensor_size_gib(item) for item in obj)
    else:
        return 0.0


def _verify_sufficient_model_size(model: torch.nn.Module | None, logger: RFLogger):
    # Check available storage space in /dev/shm
    shm_info = None
    model_size_gib = 0.0
    try:
        shm_info = _get_shm_usage()
        free_gib = shm_info["free"]
        total_gib = shm_info["total"]
        percent_used = shm_info["percent_used"]

        # Estimate the size of the model to be saved
        model_size_gib = 0.0
        if model is not None and not isinstance(model, str):
            # Estimate parameters size
            for param in model.parameters():
                if param.data is not None:
                    model_size_gib += _estimate_tensor_size_gib(param.data)

            # Estimate buffers size
            for buffer in model.buffers():
                if buffer is not None:
                    model_size_gib += _estimate_tensor_size_gib(buffer)
    except Exception as e:
        logger.warning(f"Could not check shared memory space at /dev/shm: {e}")

    if shm_info and model_size_gib > 0.0:
        # Warn if usage is high
        if percent_used > SHM_WARN_THRESHOLD:
            logger.warning(
                f"Shared memory usage is high: {percent_used:.1f}%. Available space: {free_gib:.2f}/{total_gib:.2f} GiB"
            )

        # Check if at least SHM_MIN_FREE_SPACE GiB will be left after saving the model
        if free_gib - model_size_gib < SHM_MIN_FREE_SPACE:
            raise InsufficientSharedMemoryException(
                f"Insufficient shared memory space: {free_gib:.2f} GiB available, model size: "
                f"{model_size_gib:.2f} GiB, need at least {SHM_MIN_FREE_SPACE} GiB remaining after save"
            )

    return True


def _verify_sufficient_ref_state_dict_size(ref_state_dict: dict, logger: RFLogger):
    # Check available storage space in /dev/shm
    shm_info = None
    state_dict_size_gib = 0.0
    try:
        shm_info = _get_shm_usage()
        free_gib = shm_info["free"]
        total_gib = shm_info["total"]
        percent_used = shm_info["percent_used"]

        # Estimate the size of the state dict to be saved
        state_dict_size_gib = _estimate_tensor_size_gib(ref_state_dict)

    except Exception as e:
        logger.warning(f"Could not check shared memory space at /dev/shm: {e}")

    if shm_info and state_dict_size_gib > 0.0:
        # Warn if usage is high
        if percent_used > SHM_WARN_THRESHOLD:
            logger.warning(
                f"Shared memory usage is high: {percent_used:.1f}%. Available space: {free_gib:.2f}/{total_gib:.2f} GiB"
            )

        # Check if at least SHM_MIN_FREE_SPACE GiB will be left after saving the state dict
        if free_gib - state_dict_size_gib < SHM_MIN_FREE_SPACE:
            raise InsufficientSharedMemoryException(
                f"Insufficient shared memory space: {free_gib:.2f} GiB available, state dict size: "
                f"{state_dict_size_gib:.2f} GiB, need at least {SHM_MIN_FREE_SPACE} GiB remaining after save"
            )

    return True


class SharedMemoryManager:
    """Manages PyTorch models and checkpoints in shared memory across multiple processes."""

    def __init__(self, name: str, registry=None, multiprocess_lock=None):
        """Initialize the shared memory manager with process-safe registry and locks"""
        # initialize registry
        if registry is None:
            self._manager = Manager()
            self._registry = self._manager.dict()
        else:
            self._registry = registry

        # initialize multiprocess lock
        if multiprocess_lock is None:
            self._process_lock = self._manager.Lock()
        else:
            self._process_lock = multiprocess_lock

        # initialize thread lock for operations within a single process
        self._thread_lock = threading.Lock()

        self.logger = RFLogger().create_logger(name)

    # shared memory operations
    def _safe_tensor_to_shared_memory(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        """Safely convert a tensor to shared memory format"""
        if tensor is None:
            return None
        tensor = tensor.cpu()
        tensor = tensor.detach().contiguous().clone()
        tensor.share_memory_()

        return tensor

    def _move_tensors_to_shared_memory(self, obj):
        """Recursively move all tensors in a nested structure to shared memory"""
        if isinstance(obj, torch.Tensor):
            obj.share_memory_()
            return obj
        elif isinstance(obj, dict):
            return {k: self._move_tensors_to_shared_memory(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(self._move_tensors_to_shared_memory(item) for item in obj)
        else:
            return obj

    def _clean_model_for_pickling(self, model):
        """Remove unpicklable attributes (hooks, closures) from model before saving to shared memory.
        """
        if hasattr(model, "disable_input_require_grads"):
            try:
                model.disable_input_require_grads()
            except Exception:
                pass
        
        for module in model.modules():
            if hasattr(module, "_forward_hooks"):
                module._forward_hooks.clear()
            
            if hasattr(module, "make_inputs_require_grads"):
                try:
                    delattr(module, "make_inputs_require_grads")
                except (AttributeError, TypeError):
                    pass
        
        if hasattr(model, "make_inputs_require_grads"):
            try:
                delattr(model, "make_inputs_require_grads")
            except (AttributeError, TypeError):
                pass
        
        return model

    def _move_model_to_shared_memory(self, model):
        """Move model to shared memory with proper BitsAndBytes handling"""
        model = self._clean_model_for_pickling(model)
        model = model.cpu()
        for _, param in model.named_parameters():
            if param.data is not None:
                param.data = self._safe_tensor_to_shared_memory(param.data)

        for name, buffer in model.named_buffers():
            if isinstance(buffer, torch.Tensor) and buffer is not None:
                parent_module = model
                attr_path = name.split(".")

                for attr in attr_path[:-1]:
                    parent_module = getattr(parent_module, attr)

                shared_buffer = self._safe_tensor_to_shared_memory(buffer)
                setattr(parent_module, attr_path[-1], shared_buffer)

        bnb_modules = {}

        for name, module in model.named_modules():
            if not hasattr(module, "weight"):
                continue

            import bitsandbytes as bnb

            bnb_layer_types = [
                bnb.nn.Linear4bit,
                bnb.nn.LinearFP4,
                bnb.nn.LinearNF4,
                bnb.nn.Params4bit,
            ]

            is_bnb_layer = any(isinstance(module, layer_type) for layer_type in bnb_layer_types)

            if is_bnb_layer and hasattr(module, "weight"):
                bnb_attrs = {}
                weight = module.weight

                if hasattr(weight, "data") and weight.data is not None:
                    weight.data = self._safe_tensor_to_shared_memory(weight.data)

                if hasattr(weight, "quant_state") and weight.quant_state is not None:
                    quant_state = weight.quant_state
                    bnb_attrs["quant_state_data"] = {}

                    for attr_name in dir(quant_state):
                        if not attr_name.startswith("_") and hasattr(quant_state, attr_name):
                            attr_val = getattr(quant_state, attr_name)

                            if isinstance(attr_val, torch.Tensor):
                                bnb_attrs["quant_state_data"][attr_name] = self._safe_tensor_to_shared_memory(attr_val)
                            elif not callable(attr_val):
                                bnb_attrs["quant_state_data"][attr_name] = attr_val

                    if hasattr(quant_state, "state2") and quant_state.state2 is not None:
                        state2 = quant_state.state2
                        bnb_attrs["state2_data"] = {}
                        for attr_name in dir(state2):
                            if not attr_name.startswith("_") and hasattr(state2, attr_name):
                                attr_val = getattr(state2, attr_name)
                                if isinstance(attr_val, torch.Tensor):
                                    bnb_attrs["state2_data"][attr_name] = self._safe_tensor_to_shared_memory(attr_val)
                                elif not callable(attr_val):
                                    bnb_attrs["state2_data"][attr_name] = attr_val

                    bnb_attrs["quant_state_class"] = type(quant_state).__name__

                weight_attrs = [
                    "compress_statistics",
                    "quant_type",
                    "blocksize",
                    "bnb_quantized",
                ]
                for attr in weight_attrs:
                    if hasattr(weight, attr):
                        attr_val = getattr(weight, attr)
                        if isinstance(attr_val, torch.Tensor):
                            attr_val = self._safe_tensor_to_shared_memory(attr_val)
                        bnb_attrs[attr] = attr_val

                bnb_attrs["weight_class"] = type(weight).__name__
                bnb_modules[name] = bnb_attrs

        return model, bnb_modules

    # model object operations
    def _save_full_model(self, model_id: str, model_data: dict, model_object_type: SHMObjectType):
        """Save the full model in shared memory. model_id can be either run_id or name of a base model"""
        with self._process_lock if self._process_lock else self._thread_lock:
            if model_id in self._registry and model_object_type != SHMObjectType.FULL_MODEL:
                self.logger.debug(f"Model {model_id} already exists in shared memory. Skipping save.")
                return

            # verify sufficient shared memory space before saving model
            _verify_sufficient_model_size(model_data[model_object_type.value], self.logger)

            # create model entry in registry
            if model_id not in self._registry:
                self._registry[model_id] = {model_object_type: {}}

            # move model to shared memory
            model_cpu = model_data[model_object_type.value]
            tokenizer = model_data["tokenizer"]
            model, bnb_modules = self._move_model_to_shared_memory(model_cpu)
            shared_model = {
                model_object_type: model,
                "tokenizer": tokenizer,
                "bnb_modules": self._move_tensors_to_shared_memory(bnb_modules),
            }
            model_entry = dict(self._registry[model_id])
            model_entry[model_object_type] = shared_model
            self._registry[model_id] = model_entry

            self.logger.debug(f"Saved {model_object_type.value} for run {model_id}")

    def _save_ref_state_dict(self, model_id: str, ref_state_dict: dict):
        """Save the reference state dict."""
        with self._thread_lock:
            # verify sufficient shared memory space before saving ref_state_dict
            _verify_sufficient_ref_state_dict_size(ref_state_dict, self.logger)

            # create model entry in registry
            if model_id not in self._registry:
                self._registry[model_id] = {SHMObjectType.REF_STATE_DICT: {}}

            # move ref_state_dict to shared memory
            shared_ref_state_dict = self._move_tensors_to_shared_memory(ref_state_dict)
            model_entry = dict(self._registry[model_id])
            model_entry[SHMObjectType.REF_STATE_DICT] = shared_ref_state_dict
            self._registry[model_id] = model_entry

            self.logger.debug(f"Saved ref_state_dict for {model_id}")

    def _update_checkpoints(self, model_id: str, checkpoint_updates: dict):
        """Update checkpoints in-place when possible, add new keys when needed."""
        with self._thread_lock:
            # create model entry in registry
            if model_id not in self._registry:
                self._registry[model_id] = {SHMObjectType.CHECKPOINTS: {}}

            model_entry = self._registry[model_id]
            if SHMObjectType.CHECKPOINTS not in model_entry:
                model_entry[SHMObjectType.CHECKPOINTS] = {}
            current_checkpoints = model_entry[SHMObjectType.CHECKPOINTS]

            updates_made = {"in_place": 0, "new_keys": 0}

            def update_nested_dict(current_dict, updates_dict, path=""):
                for key, new_value in updates_dict.items():
                    current_path = f"{path}.{key}" if path else key

                    if key in current_dict:
                        current_value = current_dict[key]

                        if isinstance(new_value, torch.Tensor) and isinstance(current_value, torch.Tensor):
                            # In-place tensor update if shapes match
                            if (
                                current_value.shape == new_value.shape
                                and current_value.dtype == new_value.dtype
                                and current_value.is_shared()
                            ):
                                current_value.copy_(new_value.cpu())
                                updates_made["in_place"] += 1
                            else:
                                # Need new shared tensor
                                new_shared = new_value.cpu().clone()
                                new_shared.share_memory_()
                                current_dict[key] = new_shared
                                updates_made["new_keys"] += 1
                                self.logger.debug(f"New tensor (shape/type change): {current_path}")

                        elif isinstance(new_value, dict) and isinstance(current_value, dict):
                            # Recursively update nested dicts
                            update_nested_dict(current_value, new_value, current_path)

                        else:
                            # Non-tensor value update
                            current_dict[key] = new_value

                    else:
                        # New key - add to shared memory
                        if isinstance(new_value, torch.Tensor):
                            new_shared = new_value.cpu().clone()
                            new_shared.share_memory_()
                            current_dict[key] = new_shared
                            updates_made["new_keys"] += 1
                        elif isinstance(new_value, dict):
                            # New nested dict
                            current_dict[key] = self._move_tensors_to_shared_memory(new_value)
                            updates_made["new_keys"] += 1
                        else:
                            # New non-tensor value
                            current_dict[key] = new_value

            # Update the checkpoints
            update_nested_dict(current_checkpoints, checkpoint_updates)

            # Update the registry entry to ensure Manager sees changes
            updated_entry = dict(model_entry)
            updated_entry[SHMObjectType.CHECKPOINTS] = current_checkpoints
            self._registry[model_id] = updated_entry

            self.logger.debug(f"Checkpoint update:{updates_made['in_place']} in-place, {updates_made['new_keys']} new")

    def get_shm_objects(self) -> tuple[dict, Lock]:
        """Get the shared registry and process lock"""
        return self._registry, self._process_lock

    def load_model_object(self, model_id: str, model_object_type: SHMObjectType):
        """Load a model object from shared memory."""
        model_entry = self._registry.get(model_id)
        if model_entry is None:
            self.logger.warning(f"Model {model_id} not found in shared memory")
            return None
        model_obj = model_entry.get(model_object_type)
        return model_obj

    def save_model_object(self, model_id: str, model_object_type: SHMObjectType, model_object: dict):
        """Save a model object to shared memory."""
        # save model object
        if model_object_type in [
            SHMObjectType.BASE_MODEL,
            SHMObjectType.FULL_MODEL,
            SHMObjectType.REF_FULL_MODEL,
        ]:
            self._save_full_model(model_id, model_object, model_object_type)
        elif model_object_type == SHMObjectType.REF_STATE_DICT:
            self._save_ref_state_dict(model_id, model_object)
        elif model_object_type == SHMObjectType.CHECKPOINTS:
            self._update_checkpoints(model_id, model_object)

    def delete_model_object(self, model_id: str, base_model_name: str | None = None):
        """Delete model object from shared memory registry and clean up resources."""
        with self._process_lock if self._process_lock else self._thread_lock:
            if model_id not in self._registry:
                self.logger.warning(f"Model '{model_id}' not found in shared memory during delete")
                return

            # remove checkpoints
            # TODO: add code to save to disk before deleting
            if (
                SHMObjectType.CHECKPOINTS in self._registry[model_id]
                and self._registry[model_id][SHMObjectType.CHECKPOINTS]
            ):
                del self._registry[model_id][SHMObjectType.CHECKPOINTS]
                self.logger.debug(f"Deleted checkpoints for model {model_id} from shared memory")

            # remove full_model
            # TODO: add code to save to disk before deleting
            if (
                SHMObjectType.FULL_MODEL in self._registry[model_id]
                and self._registry[model_id][SHMObjectType.FULL_MODEL]
            ):
                del self._registry[model_id][SHMObjectType.FULL_MODEL]
                self.logger.debug(f"Deleted full_model for model {model_id} from shared memory")

            # remove ref_state_dict
            if (
                SHMObjectType.REF_STATE_DICT in self._registry[model_id]
                and self._registry[model_id][SHMObjectType.REF_STATE_DICT]
            ):
                del self._registry[model_id][SHMObjectType.REF_STATE_DICT]
                self.logger.debug(f"Deleted ref_state_dict for model {model_id} from shared memory")

            # remove ref_full_model
            if (
                SHMObjectType.REF_FULL_MODEL in self._registry[model_id]
                and self._registry[model_id][SHMObjectType.REF_FULL_MODEL]
            ):
                del self._registry[model_id][SHMObjectType.REF_FULL_MODEL]
                self.logger.debug(f"Deleted ref_full_model for model {model_id} from shared memory")

            # remove shared objects (entire registry entry is deleted for base_model, not just SHMObjectType.BASE_MODEL key)
            if base_model_name and base_model_name in self._registry:
                del self._registry[base_model_name]
                self.logger.debug(f"Deleted base_model for model {model_id} from shared memory")

            # remove registry entry
            del self._registry[model_id]
            self.logger.debug(f"Deleted model registry entry for {model_id} from shared memory")

            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.debug("Force garbage collection and empty cache")

    def create_warm_start_checkpoint(self, model_id: str, warm_started_from: str):
        """Copy warm start checkpoint from model_id to warm_started_from"""
        with self._thread_lock:
            if warm_started_from not in self._registry:
                raise KeyError(f"Run '{warm_started_from}' not found in shared memory")

            # create model entry in registry
            if model_id not in self._registry:
                self._registry[model_id] = {
                    SHMObjectType.FULL_MODEL: {},
                    SHMObjectType.REF_STATE_DICT: {},
                    SHMObjectType.CHECKPOINTS: {},
                }

            # copy full_model, ref_state_dict, and checkpoints from warm_started_from to model_id
            model_entry = dict(self._registry[model_id])
            if SHMObjectType.FULL_MODEL in self._registry[warm_started_from]:
                model_entry[SHMObjectType.FULL_MODEL] = copy.deepcopy(
                    dict(self._registry[warm_started_from])[SHMObjectType.FULL_MODEL]
                )
            if SHMObjectType.REF_STATE_DICT in self._registry[warm_started_from]:
                model_entry[SHMObjectType.REF_STATE_DICT] = copy.deepcopy(
                    dict(self._registry[warm_started_from])[SHMObjectType.REF_STATE_DICT]
                )
            if SHMObjectType.CHECKPOINTS in self._registry[warm_started_from]:
                model_entry[SHMObjectType.CHECKPOINTS] = copy.deepcopy(
                    dict(self._registry[warm_started_from])[SHMObjectType.CHECKPOINTS]
                )
            self._registry[model_id] = model_entry
            self.logger.debug(f"Copied warm start checkpoint from run {warm_started_from} to run {model_id}")

    def list_models(self):
        """Get list of all model IDs currently in shared memory."""
        with self._process_lock if self._process_lock else self._thread_lock:
            return list(self._registry.keys())

    def model_exists(self, model_id: str):
        """Check if a model exists in shared memory."""
        with self._process_lock if self._process_lock else self._thread_lock:
            return model_id in self._registry
