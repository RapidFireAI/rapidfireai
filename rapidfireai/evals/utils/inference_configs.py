"""
Utility module for generating inference pipeline configurations.

This module provides functions to generate various model configurations
for multi-pipeline experimentation and hyperparameter exploration.
"""

from rapidfireai.evals.utils.config import VLLMModelConfig


def get_inference_configs() -> list[VLLMModelConfig]:
    """
    Generate a list of hardcoded inference configurations for testing.

    This is a placeholder function that returns predefined model configurations
    with varying hyperparameters. In production, this would be replaced with
    a more sophisticated configuration system (e.g., reading from config files,
    hyperparameter search algorithms, etc.).

    Returns:
        List of VLLMModelConfig instances with different hyperparameter settings
    """
    configs = []

    # Config 1: Baseline configuration with Qwen 3B model
    config_1 = VLLMModelConfig(
        model_config={
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dtype": "half",
            "gpu_memory_utilization": 0.6,
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "max_model_len": 2048,
            "disable_log_stats": True,
        },
        sampling_params={
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
        },
        context_generator=None,
    )
    configs.append(config_1)

    # Config 2: Higher temperature for more creative outputs
    config_2 = VLLMModelConfig(
        model_config={
            "model": "Qwen/Qwen2.5-3B-Instruct",
            "dtype": "half",
            "gpu_memory_utilization": 0.6,
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "max_model_len": 2048,
            "disable_log_stats": True,
        },
        sampling_params={
            "temperature": 1.0,  # Higher temperature
            "top_p": 0.95,  # Higher top_p for more diversity
            "max_tokens": 512,
        },
        context_generator=None,
    )
    configs.append(config_2)

    # Config 3: Different model (smaller) with lower temperature
    config_3 = VLLMModelConfig(
        model_config={
            "model": "Qwen/Qwen2.5-0.5B-Instruct",  # Smaller model
            "dtype": "half",
            "gpu_memory_utilization": 0.4,  # Less memory needed
            "tensor_parallel_size": 1,
            "distributed_executor_backend": "mp",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "max_model_len": 2048,
            "disable_log_stats": True,
        },
        sampling_params={
            "temperature": 0.3,  # Lower temperature for more focused outputs
            "top_p": 0.85,
            "max_tokens": 256,  # Shorter outputs
        },
        context_generator=None,
    )
    configs.append(config_3)

    return configs


def get_inference_configs_with_names(context_generator=None) -> list[tuple[str, VLLMModelConfig]]:
    """
    Generate inference configurations with descriptive names.

    Args:
        context_generator: Optional ContextGenerator to attach to all configs

    Returns:
        List of tuples (config_name, config) for easier identification
    """
    configs = get_inference_configs()

    # Attach context_generator to all configs if provided
    if context_generator:
        for config in configs:
            config.context_generator = context_generator

    named_configs = [
        ("baseline_3B_temp0.7", configs[0]),
        ("creative_3B_temp1.0", configs[1]),
        ("focused_0.5B_temp0.3", configs[2]),
    ]

    return named_configs


# Export for external use
__all__ = ["get_inference_configs", "get_inference_configs_with_names"]
