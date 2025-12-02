"""
Utility module for generating inference pipeline configurations.

This module provides functions to generate various model configurations
for multi-pipeline experimentation and hyperparameter exploration.
"""

from rapidfireai.automl import RFvLLMModelConfig


def get_inference_configs() -> list[RFvLLMModelConfig]:
    """
    Generate a list of hardcoded inference configurations for testing.

    This is a placeholder function that returns predefined model configurations
    with varying hyperparameters. In production, this would be replaced with
    a more sophisticated configuration system (e.g., reading from config files,
    hyperparameter search algorithms, etc.).

    Returns:
        List of RFvLLMModelConfig instances with different hyperparameter settings
    """
    configs = []

    # Config 1: Baseline configuration with Qwen 3B model
    config_1 = RFvLLMModelConfig(
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
        rag=None,
        prompt_manager=None,
    )
    configs.append(config_1)

    # Config 2: Higher temperature for more creative outputs
    config_2 = RFvLLMModelConfig(
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
        rag=None,
        prompt_manager=None,
    )
    configs.append(config_2)

    # Config 3: Different model (smaller) with lower temperature
    config_3 = RFvLLMModelConfig(
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
        rag=None,
        prompt_manager=None,
    )
    configs.append(config_3)

    return configs


def get_inference_configs_with_names(rag=None, prompt_manager=None) -> list[tuple[str, RFvLLMModelConfig]]:
    """
    Generate inference configurations with descriptive names.

    Args:
        rag: Optional RAG specification to attach to all configs
        prompt_manager: Optional PromptManager to attach to all configs

    Returns:
        List of tuples (config_name, config) for easier identification
    """
    configs = get_inference_configs()

    # Attach rag and prompt_manager to all configs if provided
    if rag or prompt_manager:
        for config in configs:
            if rag:
                config.rag = rag
            if prompt_manager:
                config.prompt_manager = prompt_manager

    named_configs = [
        ("baseline_3B_temp0.7", configs[0]),
        ("creative_3B_temp1.0", configs[1]),
        ("focused_0.5B_temp0.3", configs[2]),
    ]

    return named_configs


# Export for external use
__all__ = ["get_inference_configs", "get_inference_configs_with_names"]
