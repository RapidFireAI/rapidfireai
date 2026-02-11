#!/usr/bin/env python3
"""Validate rapidfire experiment log: setup (workers/GPUs), run completion (total_steps), worker termination."""

import argparse
import re


def validate_log(
    log_path: str,
    number_of_runs: int,
    total_steps: int,
    num_gpus: int,
) -> bool:
    with open(log_path) as f:
        content = f.read()
    lines = content.splitlines()

    all_pass = True

    # --- Setup validations (use num_gpus for workers and worker processes) ---
    workers_found = re.search(rf"Found {num_gpus} workers/GPUs", content)
    print(f"Setup: Found {num_gpus} workers/GPUs: {'PASS' if workers_found else 'FAIL'}")
    if not workers_found:
        all_pass = False

    if "Controller initialized" in content:
        print("Setup: Controller initialized: PASS")
    else:
        print("Setup: Controller initialized: FAIL")
        all_pass = False

    created_runs = re.search(rf"Created {number_of_runs} runs", content)
    print(f"Setup: Created {number_of_runs} runs: {'PASS' if created_runs else 'FAIL'}")
    if not created_runs:
        all_pass = False

    started_workers = re.search(rf"Started {num_gpus} worker processes successfully", content)
    print(f"Setup: Started {num_gpus} worker processes successfully: {'PASS' if started_workers else 'FAIL'}")
    if not started_workers:
        all_pass = False

    created_workers = re.search(rf"Created {num_gpus} workers", content)
    print(f"Setup: Created {num_gpus} workers: {'PASS' if created_workers else 'FAIL'}")
    if not created_workers:
        all_pass = False

    # --- Run completion validations (each run completed total_steps) ---
    steps_pattern = re.escape(f"{total_steps}/{total_steps}")
    for run_id in range(1, number_of_runs + 1):
        # Match "Run N completed steps - total_steps/total_steps" or "Run N has completed all its epochs - steps total_steps/total_steps"
        run_completed = (
            re.search(rf"Run {run_id} completed steps - {steps_pattern}", content)
            or re.search(rf"Run {run_id} has completed all its epochs - steps {steps_pattern}", content)
        )
        print(f"Run {run_id} completed {total_steps} steps: {'PASS' if run_completed else 'FAIL'}")
        if not run_completed:
            all_pass = False

    # --- Worker termination validation ---
    if "All workers shutdown gracefully" in content:
        print("Termination: All workers shutdown gracefully: PASS")
    else:
        print("Termination: All workers shutdown gracefully: FAIL")
        all_pass = False

    return all_pass


def main():
    parser = argparse.ArgumentParser(description="Validate rapidfire experiment log")
    parser.add_argument("--number-of-runs", type=int, required=True, help="Expected number of runs (e.g. 4)")
    parser.add_argument("--total-steps", type=int, required=True, help="Expected total steps per run (e.g. 16)")
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Expected number of GPUs/workers (used to validate worker and worker process counts)",
    )
    parser.add_argument("--log", required=True, help="Log file path")
    args = parser.parse_args()

    result = validate_log(args.log, args.number_of_runs, args.total_steps, args.num_gpus)
    print()
    print("Overall:", "PASS" if result else "FAIL")
    return result


if __name__ == "__main__":
    exit(0 if main() else 1)
