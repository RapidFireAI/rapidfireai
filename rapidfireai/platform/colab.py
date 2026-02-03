"""
Google Colab utilities for RapidFire AI.
"""

def is_running_in_colab() -> bool:
    """
    Check if code is running in Google Colab (not regular Jupyter).

    Returns:
        True if in Google Colab, False otherwise (including regular Jupyter notebooks)
    """
    try:
        # Check for google.colab module (only exists in Colab)
        import google.colab

        # Additional check: verify we can access Colab-specific APIs
        from google.colab.output import eval_js

        # If both succeed, we're in Colab
        return True
    except (ImportError, AttributeError):
        # Not in Colab (could be Jupyter, local Python, etc.)
        return False


def get_colab_auth_token() -> str | None:
    """
    Get the Colab authorization token for proxy requests.

    Returns:
        - In Google Colab: The authorization token string
        - In Jupyter/Local: None
    """
    if not is_running_in_colab():
        # Not in Colab (regular Jupyter, local, etc.) - no auth needed
        return None

    try:
        from google.colab.output import eval_js

        # Get the Colab auth token
        auth_token = eval_js("google.colab.kernel.accessAllowed")
        return auth_token
    except Exception as e:
        print(f"⚠️ Failed to get Colab auth token: {e}")
        return None
