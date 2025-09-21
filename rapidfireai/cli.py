#!/usr/bin/env python3
"""
Command-line interface for RapidFire AI
"""

import os
import sys
import subprocess
import argparse
import platform
import shutil
import re
import site
from pathlib import Path
from .version import __version__


def get_script_path():
    """Get the path to the start.sh script."""
    # Get the directory where this package is installed
    package_dir = Path(__file__).parent
    script_path = package_dir / "start.sh"
    
    if not script_path.exists():
        # Fallback: try to find it relative to the current working directory
        script_path = Path.cwd() / "rapidfireai" / "start.sh"
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
    info['version'] = sys.version
    info['implementation'] = platform.python_implementation()
    info['executable'] = sys.executable
    
    # Environment information
    info['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV', 'none')
    info['venv'] = 'yes' if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 'no'
    
    return info


def get_pip_packages():
    """Get list of installed pip packages."""
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "Failed to get pip packages"


def get_gpu_info():
    """Get comprehensive GPU and CUDA information."""
    info = {}
    
    # Check for nvidia-smi
    nvidia_smi_path = shutil.which('nvidia-smi')
    info['nvidia_smi'] = 'found' if nvidia_smi_path else 'not found'
    
    if nvidia_smi_path:
        try:
            # Get driver and CUDA runtime version from the full nvidia-smi output
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, check=True)
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                # Look for the header line that contains CUDA version
                for line in lines:
                    if 'CUDA Version:' in line:
                        # Extract CUDA version from line like "NVIDIA-SMI 535.183.06 Driver Version: 535.183.06 CUDA Version: 12.2"
                        cuda_version = line.split('CUDA Version:')[1].split()[0]
                        info['cuda_runtime'] = cuda_version
                        # Also extract driver version from the same line
                        if 'Driver Version:' in line:
                            driver_version = line.split('Driver Version:')[1].split('CUDA Version:')[0].strip()
                            info['driver_version'] = driver_version
                        break
                else:
                    info['driver_version'] = 'unknown'
                    info['cuda_runtime'] = 'unknown'
        except (subprocess.CalledProcessError, ValueError):
            info['driver_version'] = 'unknown'
            info['cuda_runtime'] = 'unknown'
        
        # Get GPU count, models, and VRAM
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=count,name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            if result.stdout.strip():
                lines = result.stdout.strip().split('\n')
                if lines:
                    count, name, memory = lines[0].split(', ')
                    info['gpu_count'] = int(count)
                    info['gpu_model'] = name.strip()
                    # Convert memory from MiB to GB
                    memory_mib = int(memory.split()[0])
                    memory_gb = memory_mib / 1024
                    info['gpu_memory_gb'] = f"{memory_gb:.1f}"
                    
                    # Get detailed info for multiple GPUs if present
                    if info['gpu_count'] > 1:
                        info['gpu_details'] = []
                        for line in lines:
                            count, name, memory = line.split(', ')
                            memory_mib = int(memory.split()[0])
                            memory_gb = memory_mib / 1024
                            info['gpu_details'].append({
                                'name': name.strip(),
                                'memory_gb': f"{memory_gb:.1f}"
                            })
        except (subprocess.CalledProcessError, ValueError):
            info['gpu_count'] = 0
            info['gpu_model'] = 'unknown'
            info['gpu_memory_gb'] = 'unknown'
    else:
        info['driver_version'] = 'N/A'
        info['cuda_runtime'] = 'N/A'
        info['gpu_count'] = 0
        info['gpu_model'] = 'N/A'
        info['gpu_memory_gb'] = 'N/A'
    
    # Check for nvcc (CUDA compiler)
    nvcc_path = shutil.which('nvcc')
    info['nvcc'] = 'found' if nvcc_path else 'not found'
    
    if nvcc_path:
        try:
            result = subprocess.run(['nvcc', '--version'], 
                                  capture_output=True, text=True, check=True)
            # Extract version from output like "Cuda compilation tools, release 11.8, V11.8.89"
            version_line = result.stdout.split('\n')[0]
            if 'release' in version_line:
                version = version_line.split('release')[1].split(',')[0].strip()
                info['nvcc_version'] = version
            else:
                info['nvcc_version'] = 'unknown'
        except subprocess.CalledProcessError:
            info['nvcc_version'] = 'unknown'
    else:
        info['nvcc_version'] = 'N/A'
    
    # Check CUDA installation paths
    cuda_paths = [
        '/usr/local/cuda',
        '/opt/cuda',
        '/usr/cuda',
        os.path.expanduser('~/cuda')
    ]
    
    cuda_installed = False
    for path in cuda_paths:
        if os.path.exists(path):
            cuda_installed = True
            break
    
    info['cuda_installation'] = 'present' if cuda_installed else 'not present'
    
    # Check if CUDA is on PATH
    cuda_on_path = any('cuda' in p.lower() for p in os.environ.get('PATH', '').split(os.pathsep))
    info['cuda_on_path'] = 'yes' if cuda_on_path else 'no'
    
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
    print(f"Conda Environment: {python_info['conda_env']}")
    print(f"Virtual Environment: {python_info['venv']}")
    
    # Pip Packages
    print("\n📦 Installed Packages:")
    print("-" * 30)
    pip_output = get_pip_packages()
    if pip_output != "Failed to get pip packages":
        # Show only relevant packages
        relevant_packages = ['rapidfireai', 'mlflow', 'torch', 'transformers', 'flask', 'gunicorn', 'peft', 'trl', 'bitsandbytes', 'nltk', 'evaluate', 'rouge-score', 'sentencepiece']
        lines = pip_output.split('\n')
        for line in lines:
            if any(pkg.lower() in line.lower() for pkg in relevant_packages):
                print(line)
        print("... (showing only relevant packages)")
    else:
        print(pip_output)
    
    # GPU Information
    print("\n🚀 GPU & CUDA Information:")
    print("-" * 30)
    gpu_info = get_gpu_info()
    print(f"nvidia-smi: {gpu_info['nvidia_smi']}")
    
    if gpu_info['nvidia_smi'] == 'found':
        print(f"Driver Version: {gpu_info['driver_version']}")
        print(f"CUDA Runtime: {gpu_info['cuda_runtime']}")
        print(f"GPU Count: {gpu_info['gpu_count']}")
        
        if gpu_info['gpu_count'] > 0:
            if 'gpu_details' in gpu_info:
                print("GPU Details:")
                for i, gpu in enumerate(gpu_info['gpu_details']):
                    print(f"  GPU {i}: {gpu['name']} ({gpu['memory_gb']} GB)")
            else:
                print(f"GPU Model: {gpu_info['gpu_model']}")
                print(f"Total VRAM: {gpu_info['gpu_memory_gb']} GB")
    
    print(f"nvcc: {gpu_info['nvcc']}")
    if gpu_info['nvcc'] == 'found':
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
    relevant_vars = ['CUDA_HOME', 'CUDA_PATH', 'LD_LIBRARY_PATH', 'PATH']
    for var in relevant_vars:
        value = os.environ.get(var, 'not set')
        if value != 'not set' and len(value) > 100:
            value = value[:100] + "..."
        print(f"{var}: {value}")
    
    print("\n✅ Diagnostics complete!")
    return 0

def get_cuda_version():
    """Detect CUDA version from nvcc or nvidia-smi"""
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, check=True)
        match = re.search(r'release (\d+)\.(\d+)', result.stdout)
        if match:
            return int(match.group(1))
    except (subprocess.CalledProcessError, FileNotFoundError):
        try:
            result = subprocess.run(['nvidia-smi'], 
                                  capture_output=True, text=True, check=True)
            match = re.search(r'CUDA Version: (\d+)\.(\d+)', result.stdout)
            if match:
                return int(match.group(1))
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    return None

def get_compute_capability():
    """Get compute capability from nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        match = re.search(r'(\d+)\.(\d+)', result.stdout)
        if match:
            major = int(match.group(1))
            minor = int(match.group(2))
            return major + minor / 10.0  # Return as float (e.g., 7.5, 8.0, 8.6)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def install_packages():
    """Install packages for the RapidFire AI project."""
    packages = []
    # Generate CUDA requirements file
    cuda_major = get_cuda_version()
    compute_capability = get_compute_capability()

    ## TODO: re-enable once trl has fix
    # if cuda_major == 12:
    #     print(f"\n🎯 Detected CUDA {cuda_major}.x")
    #     packages.append({"package": "vllm==0.10.1.1", "extra_args": ["--torch-backend=cu126"]})
    # elif cuda_major == 11:
    #     print(f"\n🎯 Detected CUDA {cuda_major}.x")
    #     packages.append({"package": "vllm==0.10.1.1", "extra_args": ["--torch-backend=cu118"]})
    # else:
    #     print("\n⚠️  CUDA version not detected or unsupported.")
    
    ## TODO: re-enable once flash-attn has fix
    # if cuda_major is not None:
    #     print(f"\n🎯 Detected CUDA {cuda_major}.x")
        
    #     # Determine flash-attn version based on CUDA version
    #     if cuda_major < 8:
    #         # flash-attn 1.x for CUDA < 8.0
    #         print("Installing latest flash-attn 1.x for CUDA < 8.0")
    #         packages.append({"package": "flash-attn<2.0", "extra_args": ["--no-build-isolation"]})
    #     elif cuda_major == 9:
    #         # flash-attn 3.x for CUDA 9.0 specifically
    #         print("Installing latest flash-attn 3.x for CUDA 9.0")
    #         packages.append({"package": "flash-attn>=3.0,<4.0", "extra_args": ["--no-build-isolation"]})
    #     elif cuda_major >= 8:
    #         # flash-attn 2.x for CUDA >= 8.0 (but not 9.0)
    #         print("Installing flash-attn 2.8.3 for CUDA >= 8.0")
    #         packages.append({"package": "flash-attn==2.8.3", "extra_args": ["--no-build-isolation"]})
    # else:
    #     print("\n⚠️  CUDA version not detected.")
    #     print("Skipping flash-attn installation")

    for package_info in packages:
        try:
            package = package_info["package"]
            cmd = [sys.executable, "-m", "uv", "pip", "install", package] + package_info["extra_args"]
            print(f"   Installing {package}...")
            subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"✅ Successfully installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}")
            print(f"   Error: {e}")
            print(f"   You may need to install {package} manually")
    return 0

def copy_tutorial_notebooks():
    """Copy the tutorial notebooks to the project."""
    print("Getting tutorial notebooks...")
    try:
        tutorial_path = os.getenv("RF_TUTORIAL_PATH", os.path.join(".", "tutorial_notebooks"))
        site_packages_path = site.getsitepackages()[0]
        source_path =os.path.join(site_packages_path, "tutorial_notebooks")
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


def run_init():
    """Run the init command to initialize the project."""
    print("🔧 Initializing RapidFire AI project...")
    print("-" * 30)
    print("Initializing project...")
    install_packages()
    copy_tutorial_notebooks()

    return 0

def main():
    """Main entry point for the rapidfireai command."""
    parser = argparse.ArgumentParser(
        description="RapidFire AI - Start/stop/manage services",
        prog="rapidfireai"
    )
    
    parser.add_argument(
        "command",
        nargs="?",
        default="start",
        choices=["start", "stop", "status", "restart", "setup", "doctor", "init"],
        help="Command to execute (default: start)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"RapidFire AI {__version__}"
    )
    
    args = parser.parse_args()
    
    # Handle doctor command separately
    if args.command == "doctor":
        return run_doctor()

    # Handle init command separately
    if args.command == "init":
        return run_init()
    
    # Run the script with the specified command
    return run_script([args.command])


if __name__ == "__main__":
    sys.exit(main()) 