"""Model configuration for AutoML training."""

import inspect
import copy
from dataclasses import dataclass
from typing import Any, Callable, Optional, Type, Union, get_type_hints

from peft import LoraConfig
from trl import DPOConfig, GRPOConfig, SFTConfig

from rapidfireai.automl.datatypes import List, Range

def _create_rf_class(base_class: Type, class_name: str):
    """Creating a RF class that dynamically inherits all constructor parameters and supports singleton, list, and Range values."""
    if not inspect.isclass(base_class):
        raise ValueError(f"base_class must be a class, got {type(base_class)}")

    sig = inspect.signature(base_class.__init__)
    constructor_params = [p for p in sig.parameters.keys() if p != "self"]

    type_hints = get_type_hints(base_class)
    new_type_hints = {}

    for param_name, param_type in type_hints.items():
        if param_name in constructor_params:
            new_type_hints[param_name] = param_type | List | Range

    def __init__(self, **kwargs):
        self._user_params = copy.deepcopy(kwargs)
        self._constructor_params = constructor_params
        self._initializing = True  
        
        parent_kwargs = {}
        for key, value in kwargs.items():
            if not isinstance(value, (List, Range)):
                parent_kwargs[key] = value
        
        base_class.__init__(self, **parent_kwargs)
        
        self._initializing = False  
    def copy_config(self):
        """Create a deep copy of the configuration."""
        copied_params = copy.deepcopy(self._user_params)        
        new_instance = self.__class__(**copied_params)
        
        return new_instance
    
    def __setattr__(self, name, value):
        """Override setattr to update _user_params when constructor parameters are modified."""
        
        if (hasattr(self, '_constructor_params') and 
            name in self._constructor_params and 
            hasattr(self, '_user_params') and
            name in self._user_params and
            not getattr(self, '_initializing', True)):  # Don't update during init
            self._user_params[name] = value
        
        base_class.__setattr__(self, name, value)
        

    return type(
        class_name,
        (base_class,),
        {
            "__doc__": f"RF version of {base_class.__name__}", 
            "__annotations__": new_type_hints, 
            "__init__": __init__,
            "copy": copy_config,
            "__setattr__": __setattr__
        },
    )


# Create RF wrapper classes for external libraries
RFLoraConfig = _create_rf_class(LoraConfig, "RFLoraConfig")
RFSFTConfig = _create_rf_class(SFTConfig, "RFSFTConfig")
RFDPOConfig = _create_rf_class(DPOConfig, "RFDPOConfig")
RFGRPOConfig = _create_rf_class(GRPOConfig, "RFGRPOConfig")


@dataclass
class RFModelConfig:
    """Model configuration for AutoML training."""

    model_name: str = None
    tokenizer: Optional[str] = None
    tokenizer_kwargs: Optional[dict[str, Any]] = None
    formatting_func: Optional[Union[Callable, List]] = None
    compute_metrics: Optional[Union[Callable, List]] = None
    peft_config: Optional[Union[RFLoraConfig, List]] = None
    training_args: Optional[Union[RFSFTConfig, RFDPOConfig, RFGRPOConfig]] = None
    model_type: Optional[str] = "causal_lm"
    model_kwargs: Optional[dict[str, Any]] = None
    ref_model_name: Optional[str] = None
    ref_model_type: Optional[str] = None
    ref_model_kwargs: Optional[dict[str, Any]] = None
    reward_funcs: Optional[Union[str, List, Callable, Any]] = None
    generation_config: Optional[dict[str, Any]] = None

    def copy(self):#FIXME: Handle similar to create_rf_class
        """Create a deep copy of the RFModelConfig."""
        return copy.deepcopy(self)
