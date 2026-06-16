"""
Tests for installed-mode reading and validation (rapidfireai/utils/mode_utils.py).

These guard against the IC Ops "disabled panel" bug: when RapidFire is installed
in one mode (e.g. evals) but a notebook runs the other (e.g. fit), the dispatcher
is bound to the wrong database and IC Ops silently stays disabled. The validation
blocks that mismatch early with an actionable error.
"""

import pytest

from rapidfireai.utils import mode_utils
from rapidfireai.utils.mode_utils import assert_mode_matches, get_installed_mode


@pytest.fixture
def rf_home(tmp_path, monkeypatch):
    """Point mode_utils at a temporary RF_HOME and return its path."""
    monkeypatch.setattr(mode_utils, "RF_HOME", str(tmp_path))
    return tmp_path


class TestGetInstalledMode:
    @pytest.mark.parametrize("mode", ["fit", "evals"])
    def test_reads_mode_from_file(self, rf_home, mode):
        (rf_home / "rf_mode.txt").write_text(mode)
        assert get_installed_mode() == mode

    def test_strips_whitespace(self, rf_home):
        (rf_home / "rf_mode.txt").write_text("  evals\n")
        assert get_installed_mode() == "evals"

    def test_missing_file_returns_none(self, rf_home):
        assert get_installed_mode() is None

    def test_empty_file_returns_none(self, rf_home):
        (rf_home / "rf_mode.txt").write_text("   \n")
        assert get_installed_mode() is None


class TestAssertModeMatches:
    @pytest.mark.parametrize("mode", ["fit", "evals"])
    def test_match_does_not_raise(self, mode):
        assert assert_mode_matches(mode, mode) is None

    @pytest.mark.parametrize(
        ("required", "installed"),
        [("fit", "evals"), ("evals", "fit")],
    )
    def test_conflict_raises_with_remedy(self, required, installed):
        with pytest.raises(ValueError) as excinfo:
            assert_mode_matches(required, installed)
        msg = str(excinfo.value)
        assert installed in msg
        assert required in msg
        # Message must guide the user to re-initialize.
        assert "rapidfireai init" in msg

    def test_missing_mode_allows_fit(self):
        # start.sh defaults to the fit dispatcher when rf_mode.txt is absent, so
        # run_fit() must not be blocked just because the mode file is missing.
        assert assert_mode_matches("fit", None) is None

    def test_missing_mode_blocks_evals_with_remedy(self):
        # With no mode file, services run in the default fit mode, so run_evals()
        # would be uncontrollable and must still be blocked.
        with pytest.raises(ValueError) as excinfo:
            assert_mode_matches("evals", None)
        msg = str(excinfo.value)
        assert "fit" in msg
        assert "rapidfireai init --evals" in msg

    def test_init_command_matches_required_mode(self):
        with pytest.raises(ValueError) as fit_err:
            assert_mode_matches("fit", "evals")
        assert "rapidfireai init --evals" not in str(fit_err.value)

        with pytest.raises(ValueError) as evals_err:
            assert_mode_matches("evals", "fit")
        assert "rapidfireai init --evals" in str(evals_err.value)
