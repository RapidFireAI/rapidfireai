"""
RapidFire AI Platform Module

Provides platform/environment detection, diagnostics, and health check utilities.
"""

from rapidfireai.platform.colab import is_running_in_colab, get_colab_auth_token
from rapidfireai.platform.doctor import get_doctor_info
from rapidfireai.platform.get_ip_address import get_ip_address
from rapidfireai.platform.gpu_info import get_gpu_info, get_compute_capability, get_torch_version
from rapidfireai.platform.python_info import get_python_info, get_pip_packages
from rapidfireai.platform.ping import ping_server

__all__ = [
    "is_running_in_colab",
    "get_colab_auth_token",
    "get_doctor_info",
    "get_ip_address",
    "get_gpu_info",
    "get_compute_capability",
    "get_torch_version",
    "get_python_info",
    "get_pip_packages",
    "ping_server",
]
