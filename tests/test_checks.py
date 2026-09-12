import numpy as np
import pandas as pd

from ledger.checks import (
    AGGREGATE_FLAGS,
    REGISTRY,
    EpisodeStats,
    episode_stats,
    frame_gap,
    run_checks,
    ts_nonmonotonic,
    xcorr_lag,
)
from ledger.synth import make_episodes


def _stats(defect="", **kw):
    df = make_episodes(defect=defect, seed=0, **kw)
    ep = df[df["episode_index"] == 0]
    return episode_stats(ep, fps=30.0)


def test_registry_is_populated():
    assert {
        "frac_bad_dt",
        "stuck_state_frac",
        "identity_frac",
        "ts_nonmonotonic",
        "frame_gap",
    } <= set(REGISTRY)
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
    a = np.zeros((5, 6))
    s = np.zeros((5, 6))
    best, r0, rb = xcorr_lag(a, s, range(-5, 11))
    assert np.isnan(r0) and np.isnan(rb)


def test_ts_nonmonotonic_flags_non_increasing_timestamp():
    s = EpisodeStats(
        n_frames=4,
        fps=30.0,
        ts=np.array([0.0, 0.033, 0.033, 0.1]),
        frame_index=np.arange(4),
        action=None,
        state=None,
    )
    assert ts_nonmonotonic(s) == 1.0


def test_ts_nonmonotonic_is_quiet_on_increasing_timestamp():
    s = EpisodeStats(
        n_frames=4,
        fps=30.0,
        ts=np.array([0.0, 0.033, 0.066, 0.1]),
        frame_index=np.arange(4),
        action=None,
        state=None,
    )
    assert ts_nonmonotonic(s) == 0.0


def test_frame_gap_flags_missing_frame():
    s = EpisodeStats(
        n_frames=3,
        fps=30.0,
        ts=np.array([0.0, 0.033, 0.066]),
        frame_index=np.array([0, 1, 3]),
        action=None,
        state=None,
    )
    assert frame_gap(s) == 1.0


def test_frame_gap_is_quiet_on_consecutive_frames():
    s = EpisodeStats(
        n_frames=3,
        fps=30.0,
        ts=np.array([0.0, 0.033, 0.066]),
        frame_index=np.array([0, 1, 2]),
        action=None,
        state=None,
    )
    assert frame_gap(s) == 0.0


# --------------------------------------------------------------------------- #
# IMPORTANT 4: a ragged or wrong-dtype action/state column must be visibly
# distinguished from a column that was never there, not silently collapsed
# to the same "no data" None every downstream check already treats as nan.
# --------------------------------------------------------------------------- #
def test_episode_stats_records_a_stack_error_for_a_ragged_action_column():
    ep = pd.DataFrame(
        {
            "timestamp": [0.0, 0.033, 0.066],
            "frame_index": [0, 1, 2],
            "action": [[0.0, 1.0], [0.0, 1.0, 2.0], [0.0]],  # ragged: differing lengths
            "observation.state": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        }
    )
    stats = episode_stats(ep, fps=30.0)
    assert stats.action is None
    assert stats.stack_error is True


def test_episode_stats_has_no_stack_error_when_the_column_is_simply_absent():
    ep = pd.DataFrame(
        {
            "timestamp": [0.0, 0.033, 0.066],
            "frame_index": [0, 1, 2],
        }
    )
    stats = episode_stats(ep, fps=30.0)
    assert stats.action is None
    assert stats.state is None
    assert stats.stack_error is False


def test_episode_stats_has_no_stack_error_on_a_clean_episode():
    ep = pd.DataFrame(
        {
            "timestamp": [0.0, 0.033, 0.066],
            "frame_index": [0, 1, 2],
            "action": [[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]],
            "observation.state": [[0.0, 1.0], [0.1, 1.0], [0.2, 1.0]],
        }
    )
    stats = episode_stats(ep, fps=30.0)
    assert stats.action is not None
    assert stats.stack_error is False


# --------------------------------------------------------------------------- #
# stuck_while_commanded. Found 2026-09-12: stuck_state fires on bit identical
# consecutive observations, which has two very different causes. An encoder
# reporting the same quantised value while the arm deliberately holds still is
# innocent and expected in teleoperation data. A state that does not follow a
# changing command is a defect. Of 10 flagged datasets sampled from the census,
# 8 were consistent with the innocent cause. Neither stuck_state_frac nor the
# run length structure separated them; only comparing the action channel did.
# --------------------------------------------------------------------------- #
def _stats_from(action, state, fps=30.0):
    import pandas as pd

    n = len(action)
    df = pd.DataFrame(
        {
            "timestamp": np.arange(n, dtype=np.float32) / fps,
            "frame_index": np.arange(n, dtype=np.int64),
            "episode_index": np.zeros(n, dtype=np.int64),
            "action": list(action),
            "observation.state": list(state),
        }
    )
    return episode_stats(df, fps)


def test_a_deliberate_hold_is_not_stuck_while_commanded():
    """Action frozen and state frozen together: the arm was told to hold."""
    n = 200
    a = np.zeros((n, 6))
    s = np.zeros((n, 6))
    v = run_checks(_stats_from(a, s))
    assert v["stuck_state_frac"] > 0.9, "the state really is frozen here"
    assert v["stuck_while_commanded"] == 0.0, (
        "a frozen state under a frozen command is a hold, not a fault"
    )


def test_a_state_that_ignores_a_changing_command_is_flagged():
    """Action moving, state frozen: the arm was commanded and did not follow."""
    n = 200
    a = np.cumsum(np.full((n, 6), 0.01), axis=0)  # commanded to move every frame
    s = np.zeros((n, 6))  # state never changes
    v = run_checks(_stats_from(a, s))
    assert v["stuck_while_commanded"] > 0.9, (
        f"a frozen state under a moving command must be flagged, got {v['stuck_while_commanded']}"
    )


def test_healthy_data_is_not_stuck_while_commanded():
    from ledger.synth import make_episodes

    df = make_episodes(n_eps=1, T=300, seed=0)
    ep = df[df["episode_index"] == 0]
    v = run_checks(episode_stats(ep, 30.0))
    assert v["stuck_while_commanded"] == 0.0


def test_stuck_while_commanded_is_nan_when_widths_differ():
    """2 of the 10 sampled datasets had incomparable action and state widths,
    a condition the flag itself cannot see. nan, not a number."""
    a = np.zeros((100, 7))
    s = np.zeros((100, 8))
    v = run_checks(_stats_from(a, s))
    assert np.isnan(v["stuck_while_commanded"])
