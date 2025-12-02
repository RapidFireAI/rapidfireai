#!/usr/bin/env python3
"""
Command-line interface for RapidFire AI
"""

import argparse
import os
import platform
import re
import shutil
import site
import subprocess
import sys
from pathlib import Path
from importlib.resources import files
from rapidfireai.utils.colab import is_running_in_colab
from rapidfireai.utils.get_ip_address import get_ip_address
from rapidfireai.evals.utils.constants import DispatcherConfig

from .version import __version__


def get_script_path():
    """Get the path to the start.sh script."""
    # Get the directory where this package is installed
    package_dir = Path(__file__).parent

    # Try setup/fit directory relative to package directory
    script_path = package_dir.parent / "setup" / "fit" / "start.sh"

    if not script_path.exists():
        # Fallback: try to find it relative to the current working directory
        script_path = Path.cwd() / "setup" / "fit" / "start.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"Could not find start.sh script at {script_path}")

    return script_path


def run_script(args):
    """Run the start.sh script with the given arguments."""
    script_path = get_script_path()

    # Make sure the script is executable
    if not os.access(script_path, os.X_OK):
        os.chmod(script_path, 0o755)

    # Run the script with the provided arguments
    try:
        result = subprocess.run([str(script_path)] + args, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error running start.sh: {e}", file=sys.stderr)
        return e.returncode
    except FileNotFoundError:
        print(f"Error: start.sh script not found at {script_path}", file=sys.stderr)
        return 1


def get_python_info():
    """Get comprehensive Python information."""
    info = {}

    # Python version and implementation
    info["version"] = sys.version
    info["implementation"] = platform.python_implementation()
    info["executable"] = sys.executable

    # Environment information
    info["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV", "none")
    info["venv"] = (
        "yes"
        if hasattr(sys, "real_prefix") or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        else "no"
    )
    info["site_packages"] = ", ".join(site.getsitepackages())

    return info


def get_pip_packages():
    """Get list of installed pip packages."""
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Failed to get pip packages"


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

def run_doctor():
    """Run the doctor command to diagnose system issues."""
    status = 0
    print("🔍 RapidFire AI System Diagnostics")
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
    if status == 0:
        print("\n✅ Diagnostics complete!")
    elif status == 1:
        print("\n⚠️ Diagnostics complete with warnings")
    elif status == 2:
        print("\n❌ Diagnostics complete with errors")
    else:
        print("\n❌ Diagnostics completed with unknown status")
    return 0


def get_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi"""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        match = re.search(r"release (\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            match = re.search(r"CUDA Version: (\d+)\.(\d+)", result.stdout)
            if match:
                return int(match.group(1)), int(match.group(2))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return 0, 0


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

def get_os_package_installed(package_pattern: str):
    """Get list of installed packages matching a pattern."""
    import distro
    dist_id = distro.id()
    
    try:
        if dist_id in ['ubuntu', 'debian']:
            # Use dpkg-query for Debian-based
            result = subprocess.run(
                ['dpkg-query', '-W', '-f=${Package}\n', package_pattern],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
            return []
            
        elif dist_id in ['rhel', 'centos', 'fedora', 'rocky', 'almalinux']:
            # Use rpm for Red Hat-based
            result = subprocess.run(
                ['rpm', '-qa', package_pattern],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
            return []
            
        elif dist_id in ['arch', 'manjaro']:
            # Use pacman for Arch-based
            result = subprocess.run(
                ['pacman', '-Qq'],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                all_packages = result.stdout.strip().split('\n')
                # Convert shell glob pattern to regex
                pattern_regex = package_pattern.replace('*', '.*')
                return [pkg for pkg in all_packages if re.match(pattern_regex, pkg)]
            return []
            
        elif dist_id in ['opensuse', 'sles']:
            # Use rpm for openSUSE
            result = subprocess.run(
                ['rpm', '-qa', package_pattern],
                capture_output=True,
                text=True,
                check=False
            )
            if result.returncode == 0:
                return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
            return []
            
        else:
            # Fallback: try dpkg first, then rpm
            for cmd in [['dpkg-query', '-W', '-f=${Package}\n', package_pattern],
                       ['rpm', '-qa', package_pattern]]:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if result.returncode == 0 and result.stdout.strip():
                        return [pkg.strip() for pkg in result.stdout.strip().split('\n') if pkg.strip()]
                except FileNotFoundError:
                    continue
            return []
            
    except Exception as e:
        print(f"Error checking packages: {e}")
        return []


def install_packages(evals: bool = False):
    """Install packages for the RapidFire AI project."""
    packages = []
    # Generate CUDA requirements file
    cuda_major, cuda_minor = get_cuda_version()
    compute_capability = get_compute_capability()
    python_info = get_python_info()
    site_packages = python_info["site_packages"]
    is_colab = is_running_in_colab()
    setup_directory = None
    for site_package in site_packages.split(","):
        if os.path.exists(os.path.join(site_package, "setup", "fit")):
            setup_directory = os.path.join(site_package, "setup")
            break
    if not setup_directory:
        print("❌ Setup directory not found, skipping package installation")
        return 1
    if is_colab and evals:
        print("Colab environment detected, installing evals packages")
        requirements_file = os.path.join(setup_directory, "evals", "requirements-colab.txt")
    elif is_colab and not evals:
        print("Colab environment detected, installing fit packages")
        requirements_file = os.path.join(setup_directory, "fit", "requirements-colab.txt")
    elif not is_colab and evals:
        print("Non-Colab environment detected, installing evals packages")
        requirements_file = os.path.join(setup_directory, "evals", "requirements-local.txt")
    elif not is_colab and not evals:
        print("Non-Colab environment detected, installing fit packages")
        requirements_file = os.path.join(setup_directory, "fit", "requirements-local.txt")
    else:
        print("❌ Unknown environment detected, skipping package installation")
        return 1

    try:
        print(f"Installing packages from {requirements_file}...")
        cmd = [sys.executable, "-m", "uv", "pip", "install", "-r", requirements_file]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages from {requirements_file}")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Standard output: {e.stdout}")
        if e.stderr:
            print(f"   Standard error: {e.stderr}")
        print(f"   You may need to install {requirements_file} manually")
        return 1
    print(f"✅ Successfully installed packages from {requirements_file}")

    vllm_version = "0.10.2"
    torch_version = "2.5.1"
    torchvision_version = "0.20.1"
    torchaudio_version = "2.5.1"
    torch_cuda = "cu121"
    flash_cuda = "cu121"
    if cuda_major==12:
        if cuda_minor>=9:
            # Supports Torch 2.8.0
            torch_version = "2.8.0"
            torchvision_version = "0.23.0"
            torchaudio_version = "2.8.0"
            torch_cuda = "cu129"
            flash_cuda = "cu129"
            vllm_cuda = "cu129"
            vllm_version = "0.11.0"
        elif cuda_minor>=8:
            # Supports Torch 2.9.0/1
            torch_version = "2.8.0"
            torchvision_version = "0.23.0"
            torchaudio_version = "2.8.0"
            torch_cuda = "cu128"
            flash_cuda = "cu128"
            vllm_cuda = "cu128"
            vllm_version = "0.11.0"
        elif cuda_minor>=6:
            # Supports Torch 2.9.0/1
            torch_version = "2.8.0"
            torchvision_version = "0.23.0"
            torchaudio_version = "2.8.0"
            torch_cuda = "cu126"
            flash_cuda = "cu126"
            vllm_cuda = "cu126"
        elif cuda_minor>=4:
            # Supports Torch 2.6.0
            torch_version = "2.6.0"
            torchvision_version = "0.21.0"
            torchaudio_version = "2.6.0"
            torch_cuda = "cu124"
            flash_cuda = "cu124"
            vllm_cuda = "cu124"
        else:
            # Supports Torch 2.5.1
            vllm_version = "0.7.3"
            torch_version = "2.5.1"
            torchvision_version = "0.20.1"
            torchaudio_version = "2.5.1"
            torch_cuda = "cu121"
            flash_cuda = "cu121"
            vllm_cuda = "cu121"

    elif cuda_major==13:
        # Supports Torch 2.9.0/1
        torch_version = "2.8.0"
        torchvision_version = "0.23.0"
        torchaudio_version = "2.8.0"
        torch_cuda = "cu129"
        flash_cuda = "cu129"
        vllm_cuda = "cu129"
    else:
        torch_cuda = "cu121"
        flash_cuda = "cu121"

    if is_colab:
        flash_cuda = "cu128"

    if not evals:
        pass

    if evals and is_colab:
        # packages.append({"package": "flash-attn==2.8.3", "extra_args": ["--upgrade", "--no-build-isolation"]})
        pass

    
    ## TODO: re-enable for fit once trl has fix
    if evals and not is_colab and cuda_major >= 12:
        
        print(f"\n🎯 Detected CUDA {cuda_major}.{cuda_minor}, using {torch_cuda}")
        
        packages.append({"package": f"torch=={torch_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"torchvision=={torchvision_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"torchaudio=={torchaudio_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"vllm=={vllm_version}", "extra_args": ["--upgrade"]})
        packages.append({"package": "flashinfer-python", "extra_args": []})
        packages.append({"package": "flashinfer-cubin", "extra_args": []})
        if cuda_major + (cuda_minor / 10.0) >= 12.8:
            packages.append({"package": "flashinfer-jit-cache", "extra_args": ["--upgrade","--index-url", f"https://flashinfer.ai/whl/{flash_cuda}"]})
        packages.append({"package": "flash-attn==2.8.3", "extra_args": ["--upgrade", "--no-build-isolation"]})
        # packages.append({"package": "https://github.com/RapidFireAI/faiss-wheels/releases/download/v1.13.0/rf_faiss_gpu_12_8-1.13.0-cp39-abi3-manylinux_2_34_x86_64.whl", "extra_args": []})
        # Re-install torch, torchvision, and torchaudio to ensure compatibility
        packages.append({"package": f"torch=={torch_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"torchvision=={torchvision_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"torchaudio=={torchaudio_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})

    # TODO: re-enable for fit once flash-attn has fix

    for package_info in packages:
        try:
            package = package_info["package"]
            cmd = [sys.executable, "-m", "uv", "pip", "install", package] + package_info["extra_args"]
            print(f"   Installing {package}...")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}")
            print(f"   Error: {e}")
            if e.stdout:
                print(f"   Standard output: {e.stdout}")
            if e.stderr:
                print(f"   Standard error: {e.stderr}")
            print(f"   You may need to install {package} manually")
    return 0


def copy_tutorial_notebooks():
    """Copy the tutorial notebooks to the project."""
    print("Getting tutorial notebooks...")
    try:
        tutorial_path = os.getenv("RF_TUTORIAL_PATH", os.path.join(".", "tutorial_notebooks"))
        site_packages_path = site.getsitepackages()[0]
        source_path = os.path.join(site_packages_path, "tutorial_notebooks")
        print(f"Copying tutorial notebooks from {source_path} to {tutorial_path}...")
        os.makedirs(tutorial_path, exist_ok=True)
        shutil.copytree(source_path, tutorial_path, dirs_exist_ok=True)
        print(f"✅ Successfully copied notebooks to {tutorial_path}")
    except Exception as e:
        print(f"❌ Failed to copy notebooks to {tutorial_path}")
        print(f"   Error: {e}")
        print("   You may need to copy notebooks manually")
        return 1
    return 0


def run_init(evals: bool = False):
    """Run the init command to initialize the project."""
    print("🔧 Initializing RapidFire AI project...")
    print("-" * 30)
    print("Initializing project...")
    install_packages(evals)
    copy_tutorial_notebooks()

    return 0

def copy_test_notebooks():
    """Copy the test notebooks to the project."""
    print("Getting test notebooks...")
    try:
        test_path = os.getenv("RF_TEST_PATH", os.path.join(".", "tutorial_notebooks", "tests"))
        site_packages_path = site.getsitepackages()[0]
        source_path = os.path.join(site_packages_path, "tests", "notebooks")
        print(f"Copying test notebooks from {source_path} to {test_path}...")
        os.makedirs(test_path, exist_ok=True)
        shutil.copytree(source_path, test_path, dirs_exist_ok=True)
        print(f"✅ Successfully copied test notebooks to {test_path}")
    except Exception as e:
        print(f"❌ Failed to copy test notebooks to {test_path} from {source_path}")
        print(f"   Error: {e}")
        print("   You may need to copy test notebooks manually")
        return 1
    return 0

def run_jupyter():
    """ Run the Jupyter notebook server. """
    from jupyter_server.serverapp import ServerApp
    import logging
    import io
    from contextlib import redirect_stdout, redirect_stderr

    # Suppress all logging
    logging.getLogger().setLevel(logging.CRITICAL)
    for name in logging.root.manager.loggerDict:
        logging.getLogger(name).setLevel(logging.CRITICAL)

    app = ServerApp()
    app.open_browser = False
    app.port = int(os.getenv("RF_JUPYTER_PORT", "8850"))
    app.allow_origin = '*'
    app.websocket_ping_interval = 90000
    app.log_level = 'CRITICAL'
    app.token = ""
    app.password = ""
    app.default_url = "/tree"

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            app.initialize(argv=['--ServerApp.custom_display_url='])
        
        dispatcher_port = DispatcherConfig.PORT

        if os.getenv("TERM_PROGRAM") == "vscode":
            print(f"VSCode detected, port {app.port} should automatically be forwarded to localhost")
            print(f"Manually forward port {dispatcher_port} to localhost, using the Ports tab in VSCode/Cursor/etc.")
        else:
            os_username = os.getenv("USER", os.getenv("LOGNAME", "username"))
            print(f"Manually forward port {app.port} to localhost")
            print(f"Manually forward port {dispatcher_port} to localhost")
            print(f"For example using ssh:")
            print(f"    ssh -L {app.port}:localhost:{app.port} -L {dispatcher_port}:localhost:{dispatcher_port} {os_username}@{get_ip_address()}")
        print("If there is a problem, try running jupyter manually with:")
        print(f"   jupyter notebook --no-browser --port={app.port} --ServerApp.allow_origin='*' --ServerApp.default_url='/tree' --ServerApp.token=''")
        print("\n\nAfter forwarding the ports above, access the Jupyter notebook at:")
        print(f"http://localhost:{app.port}/tree?token={app.token}")
        
        # Don't redirect anything during start - let prompts through
        app.start()
        
    except Exception as e:
        print("ERROR occurred during Jupyter server startup:", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        
        stdout_output = stdout_capture.getvalue()
        stderr_output = stderr_capture.getvalue()
        
        if stdout_output:
            print("   Standard output:", file=sys.stderr)
            print(stdout_output, file=sys.stderr)
        
        if stderr_output:
            print("   Standard error:", file=sys.stderr)
            print(stderr_output, file=sys.stderr)
        
        print("=" * 60, file=sys.stderr)
        print(f"Exception: {e}", file=sys.stderr)
        print("Try running jupyter manually with:")
        print(f"   jupyter notebook --no-browser --port={app.port} --ServerApp.allow_origin='*' --ServerApp.default_url='/tree' --ServerApp.token=''")
        raise

def main():
    """Main entry point for the rapidfireai command."""
    parser = argparse.ArgumentParser(description="RapidFire AI - Start/stop/manage services", prog="rapidfireai",
    epilog="""
Examples:
  # Basic initialization
  rapidfireai init
  
  # Initialize with evaluation dependencies
  rapidfireai init --evals
  
  # Start services
  rapidfireai start
  
  # Stop services
  rapidfireai stop

For more information, visit: https://github.com/RapidFireAI/rapidfireai
        """
    )

    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "stop", "status", "restart", "setup", "doctor", "init", "jupyter"],
        help="Command to execute (default: start)",
    )

    parser.add_argument("--version", action="version", version=f"RapidFire AI {__version__}")

    parser.add_argument(
        "--tracking-backend",
        choices=["mlflow", "tensorboard", "both"],
        default=os.getenv("RF_TRACKING_BACKEND", "mlflow" if not is_running_in_colab() else "tensorboard"),
        help="Tracking backend to use for metrics (default: mlflow)",
    )

    parser.add_argument(
        "--tensorboard-log-dir",
        default=os.getenv("RF_TENSORBOARD_LOG_DIR", None),
        help="Directory for TensorBoard logs (default: {experiment_path}/tensorboard_logs)",
    )

    parser.add_argument(
        "--colab",
        action="store_true",
        help="Run in Colab mode (skips frontend, conditionally starts MLflow based on tracking backend)",
    )

    parser.add_argument(
        "--test-notebooks",
        action="store_true",
        help="Copy test notebooks to the tutorial_notebooks directory",
    )

    parser.add_argument("--evals", action="store_true", help="Initialize with evaluation dependencies")

    args = parser.parse_args()

    # Set environment variables from CLI args

    if args.tracking_backend:
        os.environ["RF_TRACKING_BACKEND"] = args.tracking_backend
    if args.tensorboard_log_dir:
        os.environ["RF_TENSORBOARD_LOG_DIR"] = args.tensorboard_log_dir
    if args.colab:
        os.environ["RF_COLAB_MODE"] = "true"
    elif is_running_in_colab():
        os.environ["RF_COLAB_MODE"] = "true"

    # Handle doctor command separately
    if args.command == "doctor":
        return run_doctor()

    # Handle init command separately
    if args.command == "init":
        return run_init(args.evals)
    
    if args.command == "jupyter":
        return run_jupyter()

    if args.test_notebooks:
        return copy_test_notebooks()

    # Run the script with the specified command
    return run_script([args.command])


if __name__ == "__main__":
    sys.exit(main())
