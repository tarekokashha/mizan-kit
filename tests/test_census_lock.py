"""Two census runs must not share one output directory.

Found the hard way. A first run was launched, believed dead, and a second
started against the same out-dir. Both were alive and both appended, so the
ledger ended up with 438 records for 223 datasets, 215 of them written
twice. Nothing complained. For a job whose whole point is to run unattended
for hours and be resumable, a silently doubled ledger is the worst kind of
failure: it looks like data.
"""

from __future__ import annotations

import pytest

from ledger.census import CensusLockError, census_lock


def test_lock_is_held_and_released(tmp_path):
    with census_lock(tmp_path):
        assert (tmp_path / ".census.lock").exists()
    assert not (tmp_path / ".census.lock").exists()


def test_a_second_run_is_refused_while_the_first_holds_the_lock(tmp_path):
    with census_lock(tmp_path):
        with pytest.raises(CensusLockError) as e, census_lock(tmp_path):
            pass
        assert "already running" in str(e.value).lower()


def test_the_lock_names_the_holding_process(tmp_path):
    import os

    with census_lock(tmp_path):
        body = (tmp_path / ".census.lock").read_text(encoding="utf-8")
        assert str(os.getpid()) in body


def test_the_lock_is_released_even_if_the_run_raises(tmp_path):
    with pytest.raises(ValueError), census_lock(tmp_path):
        raise ValueError("boom")
    assert not (tmp_path / ".census.lock").exists()


def test_a_stale_lock_can_be_broken_deliberately(tmp_path):
    """A crashed run leaves its lock behind. The operator must be able to
    clear it, but only on purpose, never silently."""
    (tmp_path / ".census.lock").write_text("pid=999999 started=whenever", encoding="utf-8")
    with pytest.raises(CensusLockError), census_lock(tmp_path):
        pass
    with census_lock(tmp_path, force=True):
        assert (tmp_path / ".census.lock").exists()
