"""
Constants for the RapidFire AI package
"""
import os

class DispatcherConfig:
    """Class to manage the dispatcher configuration"""

    HOST: str = os.getenv("RF_API_HOST", "127.0.0.1")
    PORT: int = int(os.getenv("RF_API_PORT", "8851"))
    URL: str = f"http://{HOST}:{PORT}"
