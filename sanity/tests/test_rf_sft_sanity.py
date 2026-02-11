"""Pytest test: run RF SFT sanity script then validate its log. PASS if script exits 0 and validation returns True."""

import logging
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Paths relative to sanity/ directory layout: sanity/scripts, sanity/scripts/validations, sanity/tests, sanity/logs
TESTS_DIR = Path(__file__).resolve().parent
SANITY_DIR = TESTS_DIR.parent
SCRIPTS_DIR = SANITY_DIR / "scripts"
VALIDATIONS_DIR = SCRIPTS_DIR / "validations"
LOGS_DIR = SANITY_DIR / "logs"
LOGS_BASE = Path("/home/ubuntu/rapidfireai/logs")  # RapidFire experiment logs (unchanged)

logger = logging.getLogger("test_rf_sft_sanity")


def _log_block(text: str) -> None:
    """Log a block of text in one call. Preserves newlines; no extra formatting."""
    if text.strip():
        logger.info("%s", text.rstrip())


def setup_test_log(
    logs_dir: Path,
    log_prefix: str = "test_rf_sanity",
    log_instance: logging.Logger | None = None,
) -> logging.FileHandler:
    """Create a timestamped log file and attach a file handler. Caller must remove and close the handler when done.

    Returns the FileHandler so the test can call logger.removeHandler(handler); handler.close() in finally.
    """
    logs_dir = Path(logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"{log_prefix}_{timestamp}.log"
    handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    target = log_instance if log_instance is not None else logger
    target.setLevel(logging.INFO)
    target.addHandler(handler)
    return handler


def test_rf_sft_sanity_run_then_validate():
    """Run RF SFT sanity script with a fixed experiment name; then validate its log. PASS if both succeed."""
    rf_script = SCRIPTS_DIR / "rf-tutorial-sft-chatqa-sanity-1.py"
    validate_script = VALIDATIONS_DIR / "validate_log_sft.py"
    experiment_name = f"exp1-chatqa-sanity-{random.randint(100, 999)}"
    # Expected by validate_log_sft.py for this experiment
    num_runs = 4
    total_steps = 8
    num_gpus = 1
    log_path = LOGS_BASE / experiment_name / "rapidfire.log"
    file_handler = setup_test_log(LOGS_DIR, log_prefix="test_rf_sft_sanity")
    test_passed = False

    try:
        # 1) Run training script — must exit 0
        _log_block(f"""\
{"=" * 60}
Test run at {datetime.now(timezone.utc).isoformat()}
Experiment: {experiment_name}
RapidFire log: {log_path}
{"=" * 60}
""")
        rf_cmd = ["python", str(rf_script), "--experiment-name", experiment_name]
        result = subprocess.run(
            rf_cmd,
            cwd=str(SCRIPTS_DIR),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        rf_out = f"""--- RF script (training) ---
Command: {" ".join(rf_cmd)}

Exit code: {result.returncode}
"""
        if result.stdout:
            rf_out += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            rf_out += f"stderr:\n{result.stderr}\n"
        _log_block(rf_out)

        assert result.returncode == 0, (
            f"RF script failed (exit {result.returncode}). stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        )

        # 2) Validate log — must exit 0 (validate_log returns True)
        assert log_path.exists(), f"Log file not found: {log_path}"
        validate_cmd = [
            "python", str(validate_script),
            "--number-of-runs", str(num_runs),
            "--total-steps", str(total_steps),
            "--num-gpus", str(num_gpus),
            "--log", str(log_path),
        ]
        validate_result = subprocess.run(
            validate_cmd,
            cwd=str(SANITY_DIR),
            capture_output=True,
            text=True,
        )
        val_out = f"""--- Validate log script ---
Command: {" ".join(validate_cmd)}

Exit code: {validate_result.returncode}
"""
        if validate_result.stdout:
            val_out += f"stdout:\n{validate_result.stdout}\n"
        if validate_result.stderr:
            val_out += f"stderr:\n{validate_result.stderr}\n"
        _log_block(val_out)

        assert validate_result.returncode == 0, (
            f"Log validation failed (exit {validate_result.returncode}). "
            f"stdout:\n{validate_result.stdout}\nstderr:\n{validate_result.stderr}"
        )

        test_passed = True
    finally:
        _log_block(f"""\
{"=" * 60}
RESULT: {"PASS" if test_passed else "FAIL"}
{"=" * 60}
""")
        logger.removeHandler(file_handler)
        file_handler.close()
