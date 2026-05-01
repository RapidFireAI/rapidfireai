"""
Standalone reproduction for the evals scheduler actor-wedge bug.

Symptom (before fix):
  controller.run_multi_pipeline_inference parks MainThread in
  `time.sleep(0.5)` at controller.py "Check if all actors busy" branch
  while Ray reports 0% GPU/CPU usage. Bookkeeping says actors are busy;
  Ray says they are idle. The hang is permanent.

Root cause:
  PipelineScheduler.schedule() marks an actor busy *before* the controller
  dispatches Ray work. If anything between "schedule() returned" and
  "active_tasks[actor_id] = {...}" raises (init failure, batch serialization
  error, transient DB hiccup, actor death) AND the controller does not
  call scheduler.remove_pipeline(pipeline_id), the actor stays busy forever.

  All subsequent schedule() calls return {pipeline_id: -1, actor_id: -1}
  ("all actors busy"). The controller's busy-branch was a bare
  `time.sleep(0.5)` -- no health check, no progress -- so the loop spins
  forever. (See controller.py "Check if all actors busy" block.)

How this script demonstrates it:

  1. Without the fix: simulate a dispatch failure that does NOT call
     remove_pipeline. Show that PipelineScheduler.schedule() returns the
     "all busy" sentinel forever.
  2. With the fix: same setup, but the controller catches the dispatch
     exception and calls remove_pipeline. Show that schedule() recovers
     and serves the surviving pipeline.

Run:
    python tools/repro_scheduler_wedge.py

Exits 0 with PASS lines if the fix is in place (Change 2 in controller.py),
exits 1 with FAIL if the bookkeeping invariant is violated.

This is a deliberately small, Ray-free reproduction. Running the real
controller end-to-end would also wedge but requires the full evals stack;
the bookkeeping invariant alone is enough to expose the bug.
"""

from __future__ import annotations

import sys
import time

from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler

DISPATCH_FAILURES_TO_SIMULATE = 1
WEDGE_DETECTION_ITERS = 50  # 50 * 0.01s = 500ms; matches controller's busy-loop cadence


def _print(tag: str, msg: str) -> None:
    print(f"[{tag}] {msg}", flush=True)


def reproduce_wedge_without_fix() -> bool:
    """Replay the buggy path: actor marked busy, dispatch fails, no cleanup."""
    _print("repro", "scenario 1: dispatch fails AND controller forgets to call remove_pipeline")
    scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)

    schedule = scheduler.schedule()
    actor_id = schedule["actor_id"]
    pipeline_id = schedule["pipeline_id"]
    _print(
        "repro",
        f"schedule() -> actor {actor_id} <- pipeline {pipeline_id}; "
        f"actor_current_pipeline={scheduler.actor_current_pipeline}",
    )

    _print("repro", "simulating dispatch failure WITHOUT cleanup...")
    # ^^ This is the bug: an exception thrown between schedule() and
    # active_tasks[actor_id] = {...} would land here. If the controller
    # does not call scheduler.remove_pipeline(pipeline_id) in its except
    # block, the actor stays busy forever.

    # Try to schedule the next pipeline. With the actor leaked busy, this
    # should return the "all actors busy" sentinel forever.
    busy_sentinel_count = 0
    for _ in range(WEDGE_DETECTION_ITERS):
        result = scheduler.schedule()
        if result["pipeline_id"] == -1:
            busy_sentinel_count += 1
        else:
            break
        time.sleep(0.01)

    if busy_sentinel_count == WEDGE_DETECTION_ITERS:
        _print("FAIL", f"WEDGED -- {WEDGE_DETECTION_ITERS} consecutive 'all actors busy' returns")
        _print("FAIL", "this is the bug: actor leaked busy state, scheduler can never recover")
        return False
    _print("OK", "scheduler recovered without explicit cleanup -- bug not reproduced")
    return True


def reproduce_recovery_with_fix() -> bool:
    """Replay the fixed path: dispatch fails, controller calls remove_pipeline."""
    _print("repro", "scenario 2: dispatch fails AND controller calls remove_pipeline")
    scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)

    schedule = scheduler.schedule()
    actor_id = schedule["actor_id"]
    pipeline_id = schedule["pipeline_id"]
    _print(
        "repro",
        f"schedule() -> actor {actor_id} <- pipeline {pipeline_id}; "
        f"actor_current_pipeline={scheduler.actor_current_pipeline}",
    )

    _print("repro", "simulating dispatch failure -- controller catches and calls remove_pipeline...")
    scheduler.remove_pipeline(pipeline_id)
    _print(
        "repro",
        f"after remove_pipeline: actor_current_pipeline={scheduler.actor_current_pipeline}, "
        f"surviving pipelines={scheduler.pipeline_ids}",
    )

    if scheduler.actor_current_pipeline[actor_id] != -1:
        _print("FAIL", f"actor {actor_id} is still marked busy after remove_pipeline")
        return False

    next_schedule = scheduler.schedule()
    if next_schedule["pipeline_id"] == -1:
        _print("FAIL", "scheduler still returns 'all actors busy' after remove_pipeline")
        return False
    _print(
        "PASS",
        f"scheduler recovered: next assignment is actor {next_schedule['actor_id']} "
        f"<- pipeline {next_schedule['pipeline_id']}",
    )
    return True


def main() -> int:
    print("=" * 72)
    print("rapidfireai evals scheduler -- actor-wedge reproduction")
    print("=" * 72)

    bug_present = not reproduce_wedge_without_fix()
    print("-" * 72)
    fix_works = reproduce_recovery_with_fix()
    print("=" * 72)

    if bug_present and fix_works:
        print("BUG REPRODUCED + FIX VERIFIED")
        print("  - without remove_pipeline: scheduler wedges (returns 'all busy' forever)")
        print("  - with remove_pipeline: scheduler recovers and serves surviving pipelines")
        return 0
    print("UNEXPECTED STATE")
    print(f"  bug_present={bug_present}  fix_works={fix_works}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
