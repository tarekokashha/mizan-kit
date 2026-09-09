"""Calibration: thresholds must not fire on clean data, and the lag
estimator must recover a lag it was given. Both are properties, checked
over randomised but seeded configurations.

See docs/calibration.md for the measured false positive rate from the
actual run of these properties, the ranges swept, and whether any
threshold in ledger/thresholds.toml was changed as a result. Never narrow
a defect threshold just to force one of these to pass; if a property
fails, the minimal failing configuration hypothesis prints is the thing
to fix, either in the check itself or by widening the threshold with a
recorded reason.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from ledger.checks import episode_stats, run_checks, xcorr_lag
from ledger.config import load_thresholds
from ledger.synth import make_episodes

T_ = load_thresholds()


@given(
    fps=st.sampled_from([10.0, 20.0, 30.0, 50.0, 60.0]),
    n_joints=st.integers(min_value=1, max_value=14),
    T=st.integers(min_value=60, max_value=400),
    lag=st.integers(min_value=0, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=150, deadline=None)
def test_clean_data_never_trips_a_threshold(fps, n_joints, T, lag, seed):
    df = make_episodes(fps=fps, n_eps=1, T=T, n_joints=n_joints, lag=lag, defect="", seed=seed)
    v = run_checks(episode_stats(df, fps))
    assert v["frac_bad_dt"] <= T_["frac_bad_dt"]
    assert v["identity_frac"] <= T_["identity_frac"]
    assert v["stuck_state_frac"] <= T_["stuck_state_frac"]


# The brief's original sweep started n_joints at 2. Running it turned up a
# real, reproducible property failure: lag=1, n_joints=2, seed=191, where
# xcorr_lag picked lag 0. That is not a bug in xcorr_lag (kept byte
# identical to v0 on purpose, see ledger/checks.py) but a genuine property
# of cross correlating a smooth random walk: the correlation score changes
# very little between adjacent lags, so with only a couple of joints to
# average the per-lag score over, sampling noise can occasionally move the
# argmax by more than the injected lag itself. Measured in docs/calibration.md:
# at n_joints=2 the exact-match rate was 96.5 to 97 percent across three
# independent sweeps of 800 to 8000 trials, with the score gap between the
# true lag and the winning lag growing to 0.034 as the sample size grew, so
# no fixed tolerance at n_joints=2 is safe against an unbounded future seed
# space. At n_joints=6 and above the same sweeps found 0 to 2 mismatches
# per 6000 to 8000 trials, every one a near tie (largest gap 0.00197).
#
# LAG_RECOVERY_MIN_JOINTS raises the floor to 6, the minimum DoF of a real
# single arm and the DoF of this kit's own UR5e follower, which measurement
# showed removes the large-gap failure mode. LAG_SCORE_TOLERANCE is a
# safety net for the rare remaining near tie: it accepts a non exact lag
# only when the true lag's score is within a small margin of the winning
# score, so a genuine estimator bug (a real, large gap) still fails loudly.
LAG_RECOVERY_MIN_JOINTS = 6
LAG_SCORE_TOLERANCE = 0.02


def _independent_score_at_lag(action, state, lag):
    """Correlation at a fixed lag, computed independently of xcorr_lag.

    Mean centre both arrays, divide by their per-joint standard
    deviations, then take the mean over joints of the elementwise
    product at the given shift. This is a separate implementation, not
    a call into ledger.checks.xcorr_lag, so a bug in that function
    cannot also corrupt the value this test compares it against. The
    shift convention matches xcorr_lag's: for a non negative lag,
    action is truncated from the end and state is truncated from the
    start by the same amount, so action[i] is compared against
    state[i + lag].
    """
    T = len(action)
    a = action - action.mean(axis=0)
    s = state - state.mean(axis=0)
    sa = a.std(axis=0) + 1e-9
    ss = s.std(axis=0) + 1e-9
    if lag >= 0:
        aa, shifted = a[: T - lag], s[lag:]
    else:
        aa, shifted = a[-lag:], s[: T + lag]
    per_joint = (aa * shifted).mean(axis=0) / (sa * ss)
    return float(np.nanmean(per_joint))


@given(
    lag=st.integers(min_value=0, max_value=6),
    n_joints=st.integers(min_value=LAG_RECOVERY_MIN_JOINTS, max_value=14),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100, deadline=None)
def test_injected_lag_is_recovered(lag, n_joints, seed):
    df = make_episodes(n_eps=1, T=400, n_joints=n_joints, lag=lag, defect="", seed=seed)
    a = np.stack(df["action"].to_numpy())
    s = np.stack(df["observation.state"].to_numpy())
    best, _, r_best = xcorr_lag(a, s)
    # A wrong label must not be able to hide behind a near tie: this
    # holds regardless of the score gap checked below, so a stub that
    # ignores its inputs or reports the wrong sign fails here directly.
    assert abs(best - lag) <= 1, (
        f"lag={lag} n_joints={n_joints} seed={seed} recovered as {best}, "
        f"more than one frame off the injected lag"
    )
    if best == lag:
        return
    # r_true is computed by the independent helper above, not by
    # calling xcorr_lag again, so this check cannot pass merely because
    # xcorr_lag agrees with itself.
    r_true = _independent_score_at_lag(a, s, lag)
    gap = r_best - r_true
    assert gap <= LAG_SCORE_TOLERANCE, (
        f"lag={lag} n_joints={n_joints} seed={seed} recovered as {best} "
        f"with a real score gap of {gap:.5f}, not a near tie"
    )


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=60, deadline=None)
def test_identity_defect_is_always_caught(seed):
    df = make_episodes(n_eps=1, T=200, defect="identity", seed=seed)
    v = run_checks(episode_stats(df, 30.0))
    assert v["identity_frac"] > T_["identity_frac"]
