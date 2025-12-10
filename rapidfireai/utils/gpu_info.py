"""Utility functions for GPU information."""

import shutil
import subprocess
import re
import os

def get_compute_capability():
    """Get compute capability from nvidia-smi"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        match = re.search(r"(\d+)\.(\d+)", result.stdout)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return major + minor / 10.0  # Return as float (e.g., 7.5, 8.0, 8.6)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def get_gpu_info():
    """Get comprehensive GPU and CUDA information."""
    info = {"status": 0}

    # Check for nvidia-smi
    nvidia_smi_path = shutil.which("nvidia-smi")
    info["nvidia_smi"] = "found" if nvidia_smi_path else "not found"

    if nvidia_smi_path:
        try:
            # Get driver and CUDA runtime version from the full nvidia-smi output
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                # Look for the header line that contains CUDA version
                for line in lines:
                    if "CUDA Version:" in line:
                        # Extract CUDA version from line like "NVIDIA-SMI 535.183.06 Driver Version: 535.183.06 CUDA Version: 12.2"
                        cuda_version = line.split("CUDA Version:")[1].split()[0]
                        info["cuda_runtime"] = cuda_version
                        # Also extract driver version from the same line
                        if "Driver Version:" in line:
                            driver_version = line.split("Driver Version:")[1].split("CUDA Version:")[0].strip()
                            info["driver_version"] = driver_version
                        break
                else:
                    info["driver_version"] = "unknown"
                    info["cuda_runtime"] = "unknown"
                    info["status"] = 2 if info["status"] < 2 else info["status"]
        except (subprocess.CalledProcessError, ValueError):
            info["driver_version"] = "unknown"
            info["cuda_runtime"] = "unknown"
            info["status"] = 2 if info["status"] < 2 else info["status"]

        info["compute_capability"] = get_compute_capability()

        # Get GPU count, models, and VRAM
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count,name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
            )
            if result.stdout.strip():
                lines = result.stdout.strip().split("\n")
                if lines:
                    count, name, memory = lines[0].split(", ")
                    info["gpu_count"] = int(count)
                    info["gpu_model"] = name.strip()
                    # Convert memory from MiB to GB
                    memory_mib = int(memory.split()[0])
                    memory_gb = memory_mib / 1024
                    info["gpu_memory_gb"] = f"{memory_gb:.1f}"

                    # Get detailed info for multiple GPUs if present
                    if info["gpu_count"] > 1:
                        info["gpu_details"] = []
                        for line in lines:
                            count, name, memory = line.split(", ")
                            memory_mib = int(memory.split()[0])
                            memory_gb = memory_mib / 1024
                            info["gpu_details"].append({"name": name.strip(), "memory_gb": f"{memory_gb:.1f}"})
        except (subprocess.CalledProcessError, ValueError):
            info["gpu_count"] = 0
            info["gpu_model"] = "unknown"
            info["gpu_memory_gb"] = "unknown"
            info["status"] = 2 if info["status"] < 2 else info["status"]
    else:
        info["driver_version"] = "N/A"
        info["cuda_runtime"] = "N/A"
        info["gpu_count"] = 0
        info["gpu_model"] = "N/A"
        info["gpu_memory_gb"] = "N/A"
        info["status"] = 2 if info["status"] < 2 else info["status"]

    # Check for nvcc (CUDA compiler)
    nvcc_path = shutil.which("nvcc")
    info["nvcc"] = "found" if nvcc_path else "not found"

    if nvcc_path:
        try:
            result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
            # Extract version from output like "Cuda compilation tools, release 11.8, V11.8.89"
            version_lines = result.stdout.split("\n")
            for line in version_lines:
                if "release" in line:
                    info["nvcc_version"] = line.split("release")[1].split(",")[-1].strip()
                    break
            else:
                info["nvcc_version"] = "unknown"
                info["status"] = 1 if info["status"] < 2 else info["status"]
        except subprocess.CalledProcessError:
            info["nvcc_version"] = "unknown"
            info["status"] = 2 if info["status"] < 2 else info["status"]
    else:
        info["nvcc_version"] = "N/A"

    # Check CUDA installation paths
    cuda_paths = ["/usr/local/cuda", "/opt/cuda", "/usr/cuda", os.path.expanduser("~/cuda")]

    cuda_installed = False
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_installed = True
            break

    info["cuda_installation"] = "present" if cuda_installed else "not present"

    # Check if CUDA is on PATH
    cuda_on_path = any("cuda" in p.lower() for p in os.environ.get("PATH", "").split(os.pathsep))
    info["cuda_on_path"] = "yes" if cuda_on_path else "no"

    return info

def get_torch_version():
    """Get torch major, minor, patch version, along with cuda version if installed"""
    try:
        result = subprocess.run(["python", "-c", "import torch; print(torch.__version__)"], capture_output=True, text=True, check=True)
        version = result.stdout.strip()
        # version maybe like 2.8.0+cu128 or 2.8.0
        cuda_major = "0"
        cuda_minor = "0"
        if "+" in version:
            cuda_version = version.split("+")[1]
            cuda_major = cuda_version.split("cu")[1][:-1]
            cuda_minor = cuda_version.split("cu")[1][-1]
        major, minor, patch = version.split("+")[0].split(".")
        return major, minor, patch, cuda_major, cuda_minor
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return "0","0","0","0","0"