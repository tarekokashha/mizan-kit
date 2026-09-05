import numpy as np
import pytest
from ledger.checks import (
    REGISTRY, AGGREGATE_FLAGS, EpisodeStats, run_checks, episode_stats,
    xcorr_lag, ts_nonmonotonic, frame_gap,
)
from ledger.synth import make_episodes


def _stats(defect="", **kw):
    df = make_episodes(defect=defect, seed=0, **kw)
    ep = df[df["episode_index"] == 0]
    return episode_stats(ep, fps=30.0)


def test_registry_is_populated():
    assert {"frac_bad_dt", "stuck_state_frac", "identity_frac",
            "ts_nonmonotonic", "frame_gap"} <= set(REGISTRY)
    assert "dup_episode_frac" not in REGISTRY


def test_aggregate_flags_disjoint_from_registry():
    assert len(AGGREGATE_FLAGS) > 0
    assert set(AGGREGATE_FLAGS).isdisjoint(set(REGISTRY))


def test_clean_episode_is_quiet():
    v = run_checks(_stats(""))
    assert v["identity_frac"] == 0.0
    assert v["stuck_state_frac"] < 0.05
    assert v["frac_bad_dt"] == 0.0


def test_identity_defect_is_seen():
    assert run_checks(_stats("identity"))["identity_frac"] == 1.0


def test_stuck_defect_is_seen():
    assert run_checks(_stats("stuck"))["stuck_state_frac"] > 0.2


def test_xcorr_recovers_positive_lag():
    df = make_episodes(lag=3, T=400, n_eps=1, seed=2)
    a = np.stack(df["action"].to_numpy())
    s = np.stack(df["observation.state"].to_numpy())
    best, r0, rb = xcorr_lag(a, s, range(-5, 11))
    assert best == 3
    assert rb >= r0


def test_xcorr_returns_nan_for_short_series():
    a = np.zeros((5, 6)); s = np.zeros((5, 6))
    best, r0, rb = xcorr_lag(a, s, range(-5, 11))
    assert np.isnan(r0) and np.isnan(rb)


def test_ts_nonmonotonic_flags_non_increasing_timestamp():
    s = EpisodeStats(n_frames=4, fps=30.0, ts=np.array([0.0, 0.033, 0.033, 0.1]),
                     frame_index=np.arange(4), action=None, state=None)
    assert ts_nonmonotonic(s) == 1.0


def test_ts_nonmonotonic_is_quiet_on_increasing_timestamp():
    s = EpisodeStats(n_frames=4, fps=30.0, ts=np.array([0.0, 0.033, 0.066, 0.1]),
                     frame_index=np.arange(4), action=None, state=None)
    assert ts_nonmonotonic(s) == 0.0


def test_frame_gap_flags_missing_frame():
    s = EpisodeStats(n_frames=3, fps=30.0, ts=np.array([0.0, 0.033, 0.066]),
                     frame_index=np.array([0, 1, 3]), action=None, state=None)
    assert frame_gap(s) == 1.0


def test_frame_gap_is_quiet_on_consecutive_frames():
    s = EpisodeStats(n_frames=3, fps=30.0, ts=np.array([0.0, 0.033, 0.066]),
                     frame_index=np.array([0, 1, 2]), action=None, state=None)
    assert frame_gap(s) == 0.0
