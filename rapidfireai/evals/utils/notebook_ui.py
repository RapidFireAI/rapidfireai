"""
Evals-mode Interactive Run Controller panel.

Kept for backward compatibility; the panel itself lives in
:mod:`rapidfireai.utils.notebook_ui` and is shared with fit mode.
"""

from rapidfireai.utils.notebook_ui import NotebookUI as _NotebookUI


class NotebookUI(_NotebookUI):
    """Interactive Run Controller panel for evals experiments."""

    def __init__(
        self,
        dispatcher_url: str = "http://127.0.0.1:8851",
        refresh_rate_seconds: float = 3.0,
        auth_token: str | None = None,  # unused; accepted for backward compatibility
    ):
        super().__init__(mode="evals", dispatcher_url=dispatcher_url, refresh_rate_seconds=refresh_rate_seconds)
