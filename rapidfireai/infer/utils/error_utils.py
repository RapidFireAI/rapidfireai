"""
Error handling utilities for RF-Inferno.

Provides clean helpers for identifying and handling specific error types.
"""


def is_rate_limit_error(error: Exception) -> bool:
    """
    Check if an exception is a rate limit error.

    Handles rate limit errors from various APIs:
    - OpenAI API (openai.RateLimitError)
    - HTTP 429 errors
    - Generic rate limit exceptions

    Args:
        error: The exception to check

    Returns:
        True if the error is rate-limit related, False otherwise
    """
    # Check for OpenAI's RateLimitError
    error_type_name = type(error).__name__
    if error_type_name == "RateLimitError":
        return True

    # Check for HTTP 429 status code
    error_str = str(error).lower()
    if "429" in error_str or "rate limit" in error_str or "rate_limit" in error_str:
        return True

    # Check if it's an OpenAI error with rate_limit_exceeded code
    if hasattr(error, "code") and error.code == "rate_limit_exceeded":
        return True

    return False

