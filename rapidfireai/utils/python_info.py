"""Utility functions for Python information."""

import os
import sys
import platform
import site
import subprocess

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