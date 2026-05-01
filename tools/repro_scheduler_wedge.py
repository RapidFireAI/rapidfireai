"""
Reproduction for the evals scheduler actor-wedge bug.

PipelineScheduler.schedule() marks an actor busy before the controller
dispatches Ray work. If dispatch raises and the controller doesn't call
scheduler.remove_pipeline(pipeline_id), the actor leaks busy state and
schedule() returns {-1, -1, -1} forever.

Run:
    python tools/repro_scheduler_wedge.py
"""

from __future__ import annotations

import sys
import time

from rapidfireai.evals.scheduling.pipeline_scheduler import PipelineScheduler

WEDGE_DETECTION_ITERS = 50


def reproduce_wedge_without_cleanup() -> bool:
    """Dispatch fails, controller does not call remove_pipeline. Scheduler wedges."""
    scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)

    schedule = scheduler.schedule()
    actor_id = schedule["actor_id"]
    pipeline_id = schedule["pipeline_id"]
    print(f"scheduled actor {actor_id} <- pipeline {pipeline_id}")
    print("simulating dispatch failure with no cleanup")

    busy_sentinel_count = 0
    for _ in range(WEDGE_DETECTION_ITERS):
        result = scheduler.schedule()
        if result["pipeline_id"] == -1:
            busy_sentinel_count += 1
        else:
            break
        time.sleep(0.01)

    if busy_sentinel_count == WEDGE_DETECTION_ITERS:
        print(f"wedged: {WEDGE_DETECTION_ITERS} consecutive 'all actors busy' returns")
        return False
    print("recovered without cleanup (bug not reproduced)")
    return True


def reproduce_recovery_with_cleanup() -> bool:
    """Dispatch fails, controller calls remove_pipeline. Scheduler recovers."""
    scheduler = PipelineScheduler(pipeline_ids=[1, 2], num_actors=1, num_shards=1)

    schedule = scheduler.schedule()
    actor_id = schedule["actor_id"]
    pipeline_id = schedule["pipeline_id"]
    print(f"scheduled actor {actor_id} <- pipeline {pipeline_id}")

    scheduler.remove_pipeline(pipeline_id)
    print(f"removed pipeline {pipeline_id}; surviving={scheduler.pipeline_ids}")

    if scheduler.actor_current_pipeline[actor_id] != -1:
        print(f"actor {actor_id} still busy after remove_pipeline")
        return False

    next_schedule = scheduler.schedule()
    if next_schedule["pipeline_id"] == -1:
        print("scheduler still returns 'all actors busy' after remove_pipeline")
        return False
    print(f"recovered: actor {next_schedule['actor_id']} <- pipeline {next_schedule['pipeline_id']}")
    return True


def main() -> int:
    print("scenario 1: no cleanup")
    bug_present = not reproduce_wedge_without_cleanup()
    print()
    print("scenario 2: with remove_pipeline")
    fix_works = reproduce_recovery_with_cleanup()
    print()

    if bug_present and fix_works:
        print("bug reproduced and fix verified")
        return 0
    print(f"unexpected: bug_present={bug_present} fix_works={fix_works}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
