"""
RapidFire AI
"""

from .version import __version__, __version_info__

__author__ = "RapidFire AI Inc."
__email__ = "support@rapidfire.ai"


try:
    from rapidfireai.experiment import Experiment
except ImportError:
    # Evals dependencies not available - create helpful placeholder
    class _ExperimentPlaceholder:
        """
        Placeholder for Experiment when evaluation dependencies are not installed.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "\n" + "="*70 + "\n"
                "RapidFire AI features are not available.\n\n"
                "Missing dependencies (one or more of: vllm, flash-attn, etc.)\n\n"
                "To install evaluation dependencies:\n"
                "  1: pip install rapidfireai\n"
                "  2: Run one of the following:\n"
                "    a: rapidfireai init\n"
                "    b: rapidfireai init --evals\n"
                "="*70
            )

        def __repr__(self):
            return "<Experiment: Not Available (missing dependencies)>"

    Experiment = _ExperimentPlaceholder


__all__ = [
    "Experiment",
    "__version__",
    "__version_info__",
]
