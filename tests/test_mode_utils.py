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

    def test_empty_file_returns_empty_string(self, rf_home):
        # Distinct from a missing file (None): start.sh only defaults to evals when
        # cat fails, so a present-but-blank file must not be read as "missing".
        (rf_home / "rf_mode.txt").write_text("   \n")
        assert get_installed_mode() == ""


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

    def test_missing_mode_allows_evals(self):
        # start.sh defaults to the evals dispatcher when rf_mode.txt is absent, so
        # run_evals() must not be blocked just because the mode file is missing.
        assert assert_mode_matches("evals", None) is None

    def test_missing_mode_blocks_fit_with_remedy(self):
        # With no mode file, services run in the default evals mode, so run_fit()
        # would be uncontrollable and must still be blocked.
        with pytest.raises(ValueError) as excinfo:
            assert_mode_matches("fit", None)
        msg = str(excinfo.value)
        assert "evals" in msg
        assert "rapidfireai init --train" in msg

    @pytest.mark.parametrize("required", ["fit", "evals"])
    def test_empty_mode_blocks_with_remedy(self, required):
        # A present-but-blank rf_mode.txt is NOT the missing-file default:
        # start.sh would build a broken dispatcher path, so the guard must fail
        # even for run_evals().
        with pytest.raises(ValueError) as excinfo:
            assert_mode_matches(required, "")
        msg = str(excinfo.value)
        assert "rf_mode.txt" in msg
        assert "rapidfireai init" in msg

    def test_init_command_matches_required_mode(self):
        # fit is opt-in via `--train`; evals is the bare `init` default.
        with pytest.raises(ValueError) as fit_err:
            assert_mode_matches("fit", "evals")
        assert "rapidfireai init --train" in str(fit_err.value)

        with pytest.raises(ValueError) as evals_err:
            assert_mode_matches("evals", "fit")
        assert "rapidfireai init --train" not in str(evals_err.value)
