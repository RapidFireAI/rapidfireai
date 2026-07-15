import os
import subprocess
import sys


def test_cli_version_no_side_effects(tmp_path):
    """Test that --version does not create RF_HOME or log directories."""
    rf_home = tmp_path / "rf-test"

    env = os.environ.copy()
    env["RF_HOME"] = str(rf_home)

    result = subprocess.run(
        [sys.executable, "-m", "rapidfireai.cli", "--version"], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0
    assert "RapidFire AI" in result.stdout
    assert not rf_home.exists(), f"RF_HOME ({rf_home}) should not have been created by --version"


def test_cli_help_no_side_effects(tmp_path):
    """Test that --help does not create RF_HOME or log directories."""
    rf_home = tmp_path / "rf-test"

    env = os.environ.copy()
    env["RF_HOME"] = str(rf_home)

    result = subprocess.run(
        [sys.executable, "-m", "rapidfireai.cli", "--help"], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0
    assert not rf_home.exists(), f"RF_HOME ({rf_home}) should not have been created by --help"


def test_cli_invalid_command_has_side_effects(tmp_path):
    """Test that a non-pure-info command DOES create the directories (sanity check).
    We pass an invalid command so it exits quickly, but since it's not --help/--version,
    it should trigger directory scaffolding before exiting with an argparse error.
    """
    rf_home = tmp_path / "rf-test"

    env = os.environ.copy()
    env["RF_HOME"] = str(rf_home)

    result = subprocess.run(
        [sys.executable, "-m", "rapidfireai.cli", "invalidcommand"], env=env, capture_output=True, text=True
    )

    # argparse exits with code 2 on invalid choices
    assert result.returncode == 2
    assert rf_home.exists(), "RF_HOME should be created for non-pure-info commands"
    assert (rf_home / "logs").exists(), "RF_LOG_PATH should be created for non-pure-info commands"
