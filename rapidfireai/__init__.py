"""
RapidFire AI
"""

from .version import __version__, __version_info__

__author__ = "RapidFire AI Inc."
__email__ = "support@rapidfire.ai"

# Core imports - always available
from rapidfireai.experiment import Experiment

# Optional evals imports - gracefully handle missing dependencies
EVALS_AVAILABLE = False
EvalsExperiment = None

try:
    from rapidfireai.evals.experiment import Experiment as EvalsExperiment
    EVALS_AVAILABLE = True
except ImportError:
    # Evals dependencies not available - create helpful placeholder
    class _EvalsExperimentPlaceholder:
        """
        Placeholder for EvalsExperiment when evaluation dependencies are not installed.
        """

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "\n" + "="*70 + "\n"
                "RapidFire AI Evaluation features are not available.\n\n"
                "Missing dependencies (one or more of: vllm, flash-attn, etc.)\n\n"
                "To install evaluation dependencies:\n"
                "  Option 1: pip install rapidfireai[evals]\n"
                "  Option 2: rapidfireai init --evals\n"
                "="*70
            )

        def __repr__(self):
            return "<EvalsExperiment: Not Available (missing dependencies)>"

    EvalsExperiment = _EvalsExperimentPlaceholder


def coming_soon():
    """Placeholder function - full functionality coming soon."""
    return "RapidFire AI package is under development. Stay tuned!"


__all__ = ["Experiment", "__version__", "__version_info__", "EvalsExperiment", "EVALS_AVAILABLE"]
