"""Model configuration for AutoML training."""

import copy
import inspect
from typing import get_type_hints

from rapidfireai.evals.automl.datatypes import List, Range
from rapidfireai.evals.rag.prompt_manager import PromptManager
from rapidfireai.evals.rag.rag_pipeline import LangChainRagSpec


def _create_rf_class(base_class: type, class_name: str):
    """Creating a RF class that dynamically inherits all constructor parameters and supports singleton, list, and Range values."""
    if not inspect.isclass(base_class):
        raise ValueError(f"base_class must be a class, got {type(base_class)}")

    sig = inspect.signature(base_class.__init__)
    constructor_params = list(sig.parameters.keys())

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

        if (
            hasattr(self, "_constructor_params")
            and name in self._constructor_params
            and hasattr(self, "_user_params")
            and name in self._user_params
            and not getattr(self, "_initializing", True)
        ):  # Don't update during init
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
            "__setattr__": __setattr__,
        },
    )


RFLangChainRagSpec = _create_rf_class(LangChainRagSpec, "RFLangChainRagSpec")
RFPromptManager = _create_rf_class(PromptManager, "RFPromptManager")
