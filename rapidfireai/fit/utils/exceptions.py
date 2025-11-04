"""
This module contains custom exceptions for RapidFire.
"""


class ExperimentException(Exception):
    """Custom exception for experiment creation"""

    def __init__(self, message: str):
        super().__init__(message)


class DispatcherException(Exception):
    """Custom exception for dispatcher"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DBException(Exception):
    """Custom exception for database operations"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class DataPathException(Exception):
    """Custom exception for data path operations"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class NoGPUsFoundException(Exception):
    """Custom exception for no GPUs found"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class InitializeRunException(Exception):
    """Custom exception for initialize run"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ControllerException(Exception):
    """Custom exception for controller"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class WorkerException(Exception):
    """Custom exception for worker"""

    def __init__(self, message: str, traceback: str = None):
        self.message = message
        super().__init__(self.message)


class AutoMLException(Exception):
    """Custom exception for AutoML"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class InsufficientSharedMemoryException(Exception):
    """Custom exception for insufficient shared memory"""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)
