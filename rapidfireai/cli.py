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
from rapidfireai.evals.utils.get_ip_address import get_ip_address
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
    info = {}

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
        except (subprocess.CalledProcessError, ValueError):
            info["driver_version"] = "unknown"
            info["cuda_runtime"] = "unknown"

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
    else:
        info["driver_version"] = "N/A"
        info["cuda_runtime"] = "N/A"
        info["gpu_count"] = 0
        info["gpu_model"] = "N/A"
        info["gpu_memory_gb"] = "N/A"

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
        except subprocess.CalledProcessError:
            info["nvcc_version"] = "unknown"
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


def run_doctor():
    """Run the doctor command to diagnose system issues."""
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
            "vllm",
            "flash-attn",
            "flashinfer-python",
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
            print("⚠️ Not many packages installed, was rapidfireai init run (see installation instructions)?")
    else:
        print(pip_output)

    # GPU Information
    print("\n🚀 GPU & CUDA Information:")
    print("-" * 30)
    gpu_info = get_gpu_info()
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

    print("\n✅ Diagnostics complete!")
    return 0


def get_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi"""
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, check=True)
        match = re.search(r"release (\d+)\.(\d+)", result.stdout)
        if match:
            return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True)
            match = re.search(r"CUDA Version: (\d+)\.(\d+)", result.stdout)
            if match:
                return int(match.group(1))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return None


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


def install_packages(evals: bool = False):
    """Install packages for the RapidFire AI project."""
    packages = []
    # Generate CUDA requirements file
    cuda_major = get_cuda_version()
    compute_capability = get_compute_capability()
    python_info = get_python_info()
    site_packages = python_info["site_packages"]
    is_colab = os.getenv("COLAB_BACKEND_VERSION")
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

    if not evals:
        pass
        # Upgrading pytorch to 2.7.0 for fit
        # print("Upgrading pytorch to 2.7.0 for fit")
        # packages.append({"package": "torch==2.7.0", "extra_args": ["--upgrade","--index-url", "https://download.pytorch.org/whl/cu126"]})
        # packages.append({"package": "torchvision==0.22.0", "extra_args": ["--upgrade","--index-url", "https://download.pytorch.org/whl/cu126"]})
        # packages.append({"package": "torchaudio==2.7.0", "extra_args": ["--upgrade","--index-url", "https://download.pytorch.org/whl/cu126"]})
        # packages.append({"package": "transformers==4.57.1", "extra_args": ["--upgrade"]})

    ## TODO: re-enable for fit once trl has fix
    if evals and not is_colab and cuda_major == 12:
        print(f"\n🎯 Detected CUDA {cuda_major}.x")
        # packages.append({"package": "torch==2.5.1", "extra_args": ["--upgrade", "--index-url", "https://download.pytorch.org/whl/cu124"]})
        # packages.append({"package": "torchvision==0.20.1", "extra_args": ["--upgrade", "--index-url", "https://download.pytorch.org/whl/cu124"]})
        # packages.append({"package": "torchaudio==2.5.1", "extra_args": ["--upgrade", "--index-url", "https://download.pytorch.org/whl/cu124"]})
        packages.append({"package": "torch==2.8.0", "extra_args": ["--upgrade", "--index-url", "https://download.pytorch.org/whl/cu128"]})
        packages.append({"package": "torchvision==0.23.0", "extra_args": ["--upgrade", "--index-url", "https://download.pytorch.org/whl/cu128"]})
        packages.append({"package": "torchaudio==2.8.0", "extra_args": ["--upgrade", "--index-url", "https://download.pytorch.org/whl/cu128"]})
        # packages.append({"package": "vllm==0.7.2", "extra_args": ["--torch-backend=cu124"]})
        # packages.append({"package": "faiss-gpu-cu12==1.12.0", "extra_args": []})
        # packages.append({"package": "flashinfer-python==0.2.5", "extra_args": ["--index-url", "https://flashinfer.ai/whl/cu124/torch2.5/"]})
        # packages.append({"package": "flash-attn", "extra_args": ["--no-build-isolation"]})
        # packages.append({"package": "flashinfer-python", "extra_args": []})
        # packages.append({"package": "flashinfer-jit-cache", "extra_args": ["--index-url", "https://flashinfer.ai/whl/cu128"]})
    # elif cuda_major == 11:
    #     print(f"\n🎯 Detected CUDA {cuda_major}.x")
    #     packages.append({"package": "vllm==0.10.1.1", "extra_args": ["--torch-backend=cu118"]})
    # else:
    #     print("\n⚠️  CUDA version not detected or unsupported.")

    # TODO: re-enable for fit once flash-attn has fix
    # if cuda_major is not None:
    #     print(f"\n🎯 Detected CUDA {cuda_major}.x")

        # Determine flash-attn version based on CUDA version
        if evals and cuda_major < 8:
            # flash-attn 1.x for CUDA < 8.0
            print("Installing latest flash-attn 1.x for CUDA < 8.0")
            packages.append({"package": "flash-attn<2.0", "extra_args": ["--no-build-isolation"]})
        elif evals and cuda_major == 9:
            # flash-attn 3.x for CUDA 9.0 specifically
            print("Installing latest flash-attn 3.x for CUDA 9.0")
            packages.append({"package": "flash-attn>=3.0,<4.0", "extra_args": ["--no-build-isolation"]})
        elif evals and cuda_major >= 8:
            # flash-attn 2.x for CUDA >= 8.0 (but not 9.0)
            print("Installing flash-attn 2.8.3 for CUDA >= 8.0")
            packages.append({"package": "flash-attn==2.8.3", "extra_args": ["--no-build-isolation"]})
    # else:
    #     print("\n⚠️  CUDA version not detected.")
    #     print("Skipping flash-attn installation")

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
    app.log_level = 'CRITICAL'

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
        default=os.getenv("RF_TRACKING_BACKEND", "mlflow" if not os.getenv("COLAB_BACKEND_VERSION") else "tensorboard"),
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
    elif os.getenv("COLAB_BACKEND_VERSION"):
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
