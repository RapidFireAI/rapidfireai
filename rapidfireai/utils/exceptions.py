"""
RapidFire AI Exceptions

Custom exceptions for RapidFire AI operations.
"""


class RFException(Exception):
    """Base exception for RapidFire AI."""

    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)


class ExperimentException(RFException):
    """Exception for experiment creation and management."""
    pass


class DispatcherException(RFException):
    """Exception for dispatcher operations."""
    pass


class DBException(RFException):
    """Exception for database operations."""
    pass


class DataPathException(RFException):
    """Exception for data path operations."""
    pass


class NoGPUsFoundException(RFException):
    """Exception for no GPUs found."""
    pass


class InitializeRunException(RFException):
    """Exception for run initialization."""
    pass


class ControllerException(RFException):
    """Exception for controller operations."""
    pass


class WorkerException(RFException):
    """Exception for worker operations."""

    def __init__(self, message: str, traceback: str = None):
        self.traceback_str = traceback
        super().__init__(message)


class AutoMLException(RFException):
    """Exception for AutoML operations."""
    pass


class InsufficientSharedMemoryException(RFException):
    """Exception for insufficient shared memory."""
    pass


class PipelineException(RFException):
    """Exception for pipeline operations (evals mode)."""
    pass


class ActorException(RFException):
    """Exception for actor operations (evals mode)."""
    pass


class RateLimitException(RFException):
    """Exception for rate limiting errors (evals mode)."""
    pass


__all__ = [
    "RFException",
    "ExperimentException",
    "DispatcherException",
    "DBException",
    "DataPathException",
    "NoGPUsFoundException",
    "InitializeRunException",
    "ControllerException",
    "WorkerException",
    "AutoMLException",
    "InsufficientSharedMemoryException",
    "PipelineException",
    "ActorException",
    "RateLimitException",
]
