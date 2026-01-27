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
from rapidfireai.utils.get_ip_address import get_ip_address
from rapidfireai.utils.python_info import get_python_info
from rapidfireai.utils.constants import DispatcherConfig, JupyterConfig, ColabConfig
from rapidfireai.utils.doctor import get_doctor_info
from rapidfireai.utils.constants import RF_EXPERIMENT_PATH, RF_HOME
from rapidfireai.utils.gpu_info import get_compute_capability

from .version import __version__


def get_script_path():
    """Get the path to the start.sh script.
    """
    # Get the directory where this package is installed
    package_dir = Path(__file__).parent

    # Try setup directory relative to package directory
    script_path = package_dir.parent / "setup" / "start.sh"

    if not script_path.exists():
        # Fallback: try to find it relative to the current working directory
        script_path = Path.cwd() / "setup" / "start.sh"
        if not script_path.exists():
            raise FileNotFoundError(f"Could not find start.sh script at {script_path}")

    return script_path


def run_script(args):
    """Run the start.sh script with the given arguments.

    Args:
        args: Command arguments (e.g., ["start"])
    """
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


def run_doctor(log_lines: int = 10):
    """Run the doctor command to diagnose system issues."""
    get_doctor_info(log_lines)
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


def install_packages(evals: bool = False, init_packages: list[str] | None = None):
    """Install packages for the RapidFire AI project."""
    packages = []
    # Generate CUDA requirements file
    mode_file = Path(RF_HOME) / "rf_mode.txt"
    if evals:
        mode_file.write_text("evals")
    else:
        mode_file.write_text("fit")
    cuda_major, cuda_minor = get_cuda_version()
    python_info = get_python_info()
    site_packages = python_info["site_packages"]
    setup_directory = None
    for site_package in site_packages.split(",") + ["."]:
        if os.path.exists(os.path.join(site_package.strip(), "setup", "fit")):
            setup_directory = Path(site_package) / "setup"
            break
    if not setup_directory:
        print("❌ Setup directory not found, skipping package installation")
        return 1
    if ColabConfig.ON_COLAB and evals:
        print("Colab environment detected, installing evals packages")
        requirements_file = setup_directory / "evals" / "requirements-colab.txt"
    elif ColabConfig.ON_COLAB and not evals:
        print("Colab environment detected, installing fit packages")
        requirements_file = setup_directory / "fit" / "requirements-colab.txt"
    elif not ColabConfig.ON_COLAB and evals:
        print("Non-Colab environment detected, installing evals packages")
        requirements_file = setup_directory / "evals" / "requirements-local.txt"
    elif not ColabConfig.ON_COLAB and not evals:
        print("Non-Colab environment detected, installing fit packages")
        requirements_file = setup_directory / "fit" / "requirements-local.txt"
    else:
        print("❌ Unknown environment detected, skipping package installation")
        return 1

    try:
        print(f"Installing packages from {requirements_file.absolute()}...")
        cmd = [sys.executable, "-m", "uv", "pip", "install", "-r", requirements_file.absolute()]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages from {requirements_file.absolute()}")
        print(f"   Error: {e}")
        if e.stdout:
            print(f"   Standard output: {e.stdout}")
        if e.stderr:
            print(f"   Standard error: {e.stderr}")
        print(f"   You may need to install {requirements_file.absolute()} manually")
        return 1
    print(f"✅ Successfully installed packages from {requirements_file.absolute()}")

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

    if ColabConfig.ON_COLAB:
        flash_cuda = "cu128"

    if not evals:
        pass

    if evals and ColabConfig.ON_COLAB:
        pass

    
    ## TODO: re-enable for fit once trl has fix
    if not ColabConfig.ON_COLAB and cuda_major >= 12:
        print(f"\n🎯 Detected CUDA {cuda_major}.{cuda_minor}, using {torch_cuda}")
        
        packages.append({"package": f"torch=={torch_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"torchvision=={torchvision_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": f"torchaudio=={torchaudio_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        if evals:
            packages.append({"package": f"vllm=={vllm_version}", "extra_args": ["--upgrade"]})
            packages.append({"package": "flashinfer-python", "extra_args": []})
            packages.append({"package": "flashinfer-cubin", "extra_args": []})
            if cuda_major + (cuda_minor / 10.0) >= 12.8:
                packages.append({"package": "flashinfer-jit-cache", "extra_args": ["--upgrade","--index-url", f"https://flashinfer.ai/whl/{flash_cuda}"]})
            if get_compute_capability() >= 8.0:
                packages.append({"package": "flash-attn>=2.8.3", "extra_args": ["--upgrade", "--no-build-isolation"]})
            # else:
            #     packages.append({"package": "flash-attn-triton", "extra_args": ["--upgrade"]})
            # packages.append({"package": "https://github.com/RapidFireAI/faiss-wheels/releases/download/v1.13.0/rf_faiss_gpu_12_8-1.13.0-cp39-abi3-manylinux_2_34_x86_64.whl", "extra_args": []})
            # Re-install torch, torchvision, and torchaudio to ensure compatibility
            packages.append({"package": f"torch=={torch_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
            packages.append({"package": f"torchvision=={torchvision_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
            packages.append({"package": f"torchaudio=={torchaudio_version}", "extra_args": ["--upgrade", "--index-url", f"https://download.pytorch.org/whl/{torch_cuda}"]})
        packages.append({"package": "numpy<2.3", "extra_args": ["--upgrade"]})

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
    app.port = JupyterConfig.PORT
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
  # Basic initialization for training
  rapidfireai init
  #or
  # Basic Initialize with evaluation dependencies
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
        "--tracking-backends",
        choices=["mlflow", "tensorboard", "trackio"],
        default=["mlflow"] if not ColabConfig.ON_COLAB else ["tensorboard"],
        help="Tracking backend to use for metrics (default: mlflow on Non-Google Colab and tensorboard on Google Colab)",
        nargs="*",
        action="extend"
    )

    parser.add_argument(
        "--tensorboard-log-dir",
        default=os.getenv("RF_TENSORBOARD_LOG_DIR", None),
        help=f"Directory for TensorBoard logs (default: {RF_EXPERIMENT_PATH}/tensorboard_logs)",
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

    parser.add_argument("--force", "-f", action="store_true", help="Force action without confirmation")

    parser.add_argument("--evals", action="store_true", help="Initialize with evaluation dependencies")

    parser.add_argument("--log-lines", type=int, default=10, help="Number of lines to log to the console")

    args = parser.parse_args()

    # Set environment variables from CLI args

    if args.tracking_backends:
        os.environ["RF_MLFLOW_ENABLED"] = "false"
        os.environ["RF_TENSORBOARD_ENABLED"] = "false"
        os.environ["RF_TRACKIO_ENABLED"] = "false"
        if "mlflow" in args.tracking_backends:
            os.environ["RF_MLFLOW_ENABLED"] = "true"
        if "tensorboard" in args.tracking_backends:
            os.environ["RF_TENSORBOARD_ENABLED"] = "true"
        if "trackio" in args.tracking_backends:
            os.environ["RF_TRACKIO_ENABLED"] = "true"
    if args.tensorboard_log_dir:
        os.environ["RF_TENSORBOARD_LOG_DIR"] = args.tensorboard_log_dir
    if args.colab:
        os.environ["RF_COLAB_MODE"] = "true"
    elif ColabConfig.ON_COLAB and os.getenv("RF_COLAB_MODE") is None:
        os.environ["RF_COLAB_MODE"] = "true"
    
    # Handle force command separately
    if args.force:
        os.environ["RF_FORCE"] = "true"

    # Handle doctor command separately
    if args.command == "doctor":
        return run_doctor(args.log_lines)

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
