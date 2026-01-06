"""
Utility functions for doctor command.
"""

import os
import platform
from pathlib import Path
from rapidfireai.utils.python_info import get_python_info, get_pip_packages
from rapidfireai.utils.gpu_info import get_gpu_info, get_torch_version
from rapidfireai.utils.ping import ping_server
from rapidfireai.utils.constants import (
    JupyterConfig,
    DispatcherConfig,
    MLFlowConfig,
    FrontendConfig,
    RayConfig,
    ColabConfig,
    RF_HOME,
    RF_LOG_PATH,
    RF_EXPERIMENT_PATH,
    RF_DB_PATH,
    RF_TENSORBOARD_LOG_DIR,
    RF_TRAINING_LOG_FILENAME,
    RF_MLFLOW_ENABLED,
    RF_TENSORBOARD_ENABLED,
    RF_TRACKIO_ENABLED,
    RF_LOG_FILENAME,
)

def get_doctor_info(log_lines: int = 10):
    """
    Get doctor information.
    """
    status = 0
    # Get mode from rf_mode.txt in RF_HOME
    mode_file = Path(RF_HOME) / "rf_mode.txt"
    if mode_file.exists():
        mode = mode_file.read_text().strip()
    else:
        mode = "unknown"
    print(f"🩺 RapidFire AI System Diagnostics, Mode: {mode}")
    print("=" * 50)

    # Python Information
    print("\n🐍 Python Environment:")
    print("-" * 30)
    python_info = get_python_info()
    print(f"Version: {python_info['version'].split()[0]}")
    print(f"Implementation: {python_info['implementation']}")
    print(f"Executable: {python_info['executable']}")
    print(f"Site Packages: {python_info['site_packages']}")
    print(f"Conda Environment: {python_info['conda_env']}")
    print(f"Virtual Environment: {python_info['venv']}")
    # Pip Packages
    print("\n📦 Installed Packages:")
    print("-" * 30)
    pip_output = get_pip_packages()
    if pip_output != "Failed to get pip packages":
        # Show only relevant packages
        relevant_packages = [
            "rapidfireai",
            "mlflow",
            "torch",
            "transformers",
            "flask",
            "gunicorn",
            "peft",
            "trl",
            "bitsandbytes",
            "nltk",
            "langchain",
            "ray",
            "sentence-transformers",
            "openai",
            "tiktoken",
            "langchain-core",
            "langchain-community",
            "langchain-openai",
            "langchain-huggingface",
            "langchain-classic",
            "unstructured",
            "waitress",
            "vllm",
            "rf-faiss",
            "rf-faiss-gpu-12-8",
            "faiss-gpu-cu12",
            "vllm",
            "flash-attn",
            "flash_attn",
            "flashinfer-python",
            "flashinfer-cubin",
            "flashinfer-jit-cache",
            "tensorboard",
            "numpy",
            "pandas",
            "torch",
            "torchvision",
            "torchaudio",
            "scipy",
            "datasets",
            "evaluate",
            "rouge-score",
            "sentencepiece",
        ]
        lines = pip_output.split("\n")
        found_packages = []
        for line in lines:
            if any(pkg.lower() in line.lower() for pkg in relevant_packages):
                found_packages.append(line)
                print(line)
        print("... (showing only relevant packages)")
        if len(found_packages) < 5:
            status = 1 if status == 0 else status
            print("⚠️ Not many packages installed, was rapidfireai init run (see installation instructions)?")
    else:
        print(pip_output)

    # GPU Information
    print("\n🚀 GPU & CUDA Information:")
    print("-" * 30)
    gpu_info = get_gpu_info()
    if gpu_info["status"] == 1:
        print("⚠️ Some GPU information not found")
        status = 1 if status == 0 else status
    elif gpu_info["status"] == 2:
        print("❌ Some GPU information not found")
        status = 2 if status < 2 else status
    print(f"nvidia-smi: {gpu_info['nvidia_smi']}")

    if gpu_info["nvidia_smi"] == "found":
        print(f"Driver Version: {gpu_info['driver_version']}")
        print(f"CUDA Runtime: {gpu_info['cuda_runtime']}")
        print(f"GPU Count: {gpu_info['gpu_count']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")

        if gpu_info["gpu_count"] > 0:
            if "gpu_details" in gpu_info:
                print("GPU Details:")
                for i, gpu in enumerate(gpu_info["gpu_details"]):
                    print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']} GB)")
            else:
                print(f"GPU Model: {gpu_info['gpu_model']}")
                print(f"Total VRAM: {gpu_info['gpu_memory_gb']} GB")

    print(f"nvcc: {gpu_info['nvcc']}")
    if gpu_info["nvcc"] == "found":
        print(f"nvcc Version: {gpu_info['nvcc_version']}")

    print(f"CUDA Installation: {gpu_info['cuda_installation']}")
    print(f"CUDA on PATH: {gpu_info['cuda_on_path']}")
    # Get torch cuda version
    major, minor, patch, torch_cuda_major, torch_cuda_minor = get_torch_version()
    if int(major) > 0:
        print(f"Torch Version: {major}.{minor}.{patch}")
    else:
        status = 1 if status == 0 else status
        print("⚠️ Torch version not found") 
    if int(torch_cuda_major) > 0:
        print(f"Torch CUDA Version: {torch_cuda_major}.{torch_cuda_minor}")
    else:
        status = 1 if status == 0 else status
        print("⚠️ Torch CUDA Version: unknown")

    # Check RapidFire AI ports
    print ("\n🛜 RapidFire AI Ports:")
    print ("-" * 30)
    for check_item in [
        {"host": JupyterConfig.HOST, "port": JupyterConfig.PORT, "service": "Jupyter"}, 
        {"host": DispatcherConfig.HOST, "port": DispatcherConfig.PORT, "service": "API(Dispatcher)"}, 
        {"host": MLFlowConfig.HOST, "port": MLFlowConfig.PORT, "service": "MLFlow"},
        {"host": FrontendConfig.HOST, "port": FrontendConfig.PORT, "service": "Frontend"},
        {"host": RayConfig.HOST, "port": RayConfig.PORT, "service": "Ray"}]:

        for host_index, host_check in enumerate(["127.0.0.1", check_item["host"]]):
            if host_index == 0 or (host_check not in ["127.0.0.1", "0.0.0.0"]):
                checker = ping_server(host_check, check_item["port"])
                if checker:
                    print(f"{check_item['service']}: is currently Listening on {host_check}:{check_item['port']}")
                else:
                    print(f"{check_item['service']}: is NOT currently listening on {host_check}:{check_item['port']}")
        
    # System Information
    print("\n💻 System Information:")
    print("-" * 30)
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Processor: {platform.processor()}")

    # Environment Variables
    print("\n🔧 Environment Variables:")
    print("-" * 30)
    relevant_vars = ["CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "PATH"]
    for var in relevant_vars:
        value = os.environ.get(var, "not set")
        if value != "not set" and len(value) > 200:
            value = value[:200] + "..."
        print(f"{var}: {value}")
    print("\n🔍 RF_ Constants:")
    print("-" * 30)
    print(f"RF_HOME: {RF_HOME}")
    print(f"RF_DB_PATH: {RF_DB_PATH}")
    print(f"RF_LOG_PATH: {RF_LOG_PATH}")
    print(f"RF_EXPERIMENT_PATH: {RF_EXPERIMENT_PATH}")
    print(f"RF_TENSORBOARD_LOG_DIR: {RF_TENSORBOARD_LOG_DIR}")
    print(f"RF_LOG_FILENAME: {RF_LOG_FILENAME}")
    print(f"RF_TRAINING_LOG_FILENAME: {RF_TRAINING_LOG_FILENAME}")
    print(f"RF_MLFLOW_ENABLED: {RF_MLFLOW_ENABLED}")
    print(f"RF_TENSORBOARD_ENABLED: {RF_TENSORBOARD_ENABLED}")
    print(f"RF_TRACKIO_ENABLED: {RF_TRACKIO_ENABLED}")
    print(f"JupyterConfig: {JupyterConfig()}")
    print(f"DispatcherConfig: {DispatcherConfig()}")
    print(f"MLFlowConfig: {MLFlowConfig()}")
    print(f"FrontendConfig: {FrontendConfig()}")
    print(f"RayConfig: {RayConfig()}")
    print(f"ColabConfig: {ColabConfig()}")
    # Print all files recursively under RF_LOG_PATH
    lines_to_log = str(log_lines)
    if log_lines == -1:
        lines_to_log = "all"
    print(f"\n🪵 Log Files (last {lines_to_log} lines):")
    for root, dirs, list_files in os.walk(RF_LOG_PATH):
        for file in list_files:
            current_item = os.path.join(root, file)
            print(f"\n📄{current_item}:")
            if log_lines != 0:
                if not os.path.isdir(current_item):
                    with open(current_item, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        read_lines = lines[-log_lines:]
                        if log_lines == -1:
                            read_lines = lines
                        for line in read_lines:
                            print(line.strip())
    print("\n")
    if status == 0:
        print("\n✅ Diagnostics complete!")
    elif status == 1:
        print("\n⚠️ Diagnostics complete with warnings")
    elif status == 2:
        print("\n❌ Diagnostics complete with errors")
    else:
        print("\n❌ Diagnostics completed with unknown status")