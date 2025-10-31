"""
Dispatcher module for interactive control of RF-Inferno experiments.

Provides REST API for dynamic pipeline management:
- Stop, Resume, Delete pipelines
- Clone new pipelines from existing contexts

The dispatcher automatically starts when an Experiment is created and runs
in a background thread. It can also be run standalone for testing.
"""

from rapidfireai.evals.dispatcher.dispatcher import Dispatcher, run_dispatcher, start_dispatcher_thread

__all__ = ["Dispatcher", "run_dispatcher", "start_dispatcher_thread"]
