"""One function per temporal defect, held in a registry.

Adding a check is adding a function. The report schema, the CSV columns
and the flag names are all derived from REGISTRY, so they cannot drift
apart from the checks that fill them.

REGISTRY holds only genuine per-episode scalar checks, each computed from
a single episode's EpisodeStats. Two measurements do not fit that shape
and are therefore not registry entries.

The lag triple, best lag plus the action-state correlation at lag zero
and at the best lag, compares a whole episode's action against its state
rather than reducing to one defect fraction. It is exposed here as
xcorr_lag, a plain function, unchanged from v0.

Duplicate episode detection needs every episode's action hash at once to
find repeats, so a single EpisodeStats cannot answer it alone. This
module exposes the per-episode half of that work, action_head_hash, and
leaves the cross-episode comparison to the caller.

AGGREGATE_FLAGS names the flags that come from those two measurements.
They are computed in report.py from all episodes together, not by this
registry.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

LAGS = tuple(range(-5, 11))

# Flags derived from cross-episode aggregates rather than from a REGISTRY
# check. report.py computes these once it has every episode's stats in
# hand. Kept here, next to the registry, so the two namespaces can be
# checked against each other instead of drifting apart silently.
AGGREGATE_FLAGS = ("large_lag", "negative_lag", "duplicate_episodes")


@dataclass(frozen=True)
class Check:
    name: str
    fn: Callable[[EpisodeStats], float]
    threshold_key: str | None
    higher_is_worse: bool


REGISTRY: dict[str, Check] = {}


def register(name: str, threshold_key: str | None = None, higher_is_worse: bool = True):
    def deco(fn):
        REGISTRY[name] = Check(name, fn, threshold_key or name, higher_is_worse)
        return fn

    return deco


@dataclass
class EpisodeStats:
    n_frames: int
    fps: float
    ts: np.ndarray | None
    frame_index: np.ndarray | None
    action: np.ndarray | None
    state: np.ndarray | None
    # IMPORTANT 4: True when "action" or "observation.state" was present
    # on this episode but could not be stacked into a (T, D) array (a
    # ragged column, a wrong dtype, or a stack that came out with the
    # wrong number of dimensions). episode_stats sets this only when the
    # column existed and _stack failed on it, never for a column that
    # was simply absent, so a caller can tell the two apart instead of
    # both collapsing to the same action/state=None every downstream
    # check already reports as nan.
    stack_error: bool = False


def _stack(col) -> np.ndarray | None:
    """Stack a column of per-frame vectors into one (T, D) array.

    Only ValueError and TypeError are caught: a ragged column (frames of
    differing length) raises ValueError from np.stack itself, and a
    value np.asarray(..., dtype=float64) cannot convert raises
    TypeError or ValueError from the conversion. Anything else
    propagates rather than being swallowed here, so a genuine bug
    elsewhere cannot silently masquerade as "no data". Returns None on
    a caught failure or when the stack does not come out 2D, exactly as
    before; episode_stats is what turns that into a visible, counted
    signal (stack_error) rather than treating it the same as a column
    that was never there.
    """
    try:
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in col])
    except (ValueError, TypeError):
        return None
    return arr if arr.ndim == 2 else None


def episode_stats(ep, fps: float) -> EpisodeStats:
    ts = np.asarray(ep["timestamp"], dtype=np.float64) if "timestamp" in ep else None
    fi = np.asarray(ep["frame_index"], dtype=np.int64) if "frame_index" in ep else None
    stack_error = False
    if "action" in ep:
        ac = _stack(ep["action"])
        if ac is None:
            stack_error = True
    else:
        ac = None
    if "observation.state" in ep:
        st = _stack(ep["observation.state"])
        if st is None:
            stack_error = True
    else:
        st = None
    return EpisodeStats(len(ep), fps, ts, fi, ac, st, stack_error)


def xcorr_lag(action, state, lags: Iterable[int] = LAGS):
    T = len(action)
    if T < 30:
        return 0, float("nan"), float("nan")
    a = action - action.mean(0)
    s = state - state.mean(0)
    sa = a.std(0) + 1e-9
    ss = s.std(0) + 1e-9
    scores = {}
    for k in lags:
        if k >= 0:
            aa, st = a[: T - k], s[k:]
        else:
            aa, st = a[-k:], s[: T + k]
        if len(aa) < 20:
            continue
        scores[k] = float(np.nanmean(((aa * st).mean(0)) / (sa * ss)))
    if not scores:
        return 0, float("nan"), float("nan")
    best = max(scores, key=scores.get)
    return best, scores.get(0, float("nan")), scores[best]


def action_head_hash(s: EpisodeStats, n: int = 50) -> str | None:
    """Hash of the first n action frames. v0's duplicate detector.

    Kept for the v0 equivalence tests and reachable by passing n. It is not
    what the duplicate flag should use: see action_episode_hash.
    """
    if s.action is None:
        return None
    return hashlib.md5(np.round(s.action[:n], 4).tobytes()).hexdigest()


def action_episode_hash(s: EpisodeStats) -> str | None:
    """Hash of the WHOLE action array, which is what duplication means.

    action_head_hash looked at the first 50 frames only, so episodes that
    begin from a shared home pose hashed identically however differently
    they ended. Robot episodes routinely start from a home pose, so that is
    a common false positive rather than an exotic one. Measured on
    2026-09-12: four fully divergent episodes sharing a 60 frame home pose
    gave dup_episode_frac 0.750 and raised the flag.

    Two episodes are duplicates when they are the same episode, which is a
    statement about all of their frames. Hashing all of them is both the
    correct check and no more expensive, since the array is already in
    memory.

    Length is folded in first, so two episodes of different length can never
    collide on a shared prefix.
    """
    if s.action is None:
        return None
    h = hashlib.md5()
    h.update(str(len(s.action)).encode())
    h.update(np.round(s.action, 4).tobytes())
    return h.hexdigest()


@register("frac_bad_dt")
def frac_bad_dt(s: EpisodeStats) -> float:
    if s.ts is None or len(s.ts) < 2 or not (s.fps and s.fps > 0):
        return float("nan")
    expected = 1.0 / s.fps
    dt = np.diff(s.ts)
    return float(np.sum(np.abs(dt - expected) > 0.25 * expected) / len(dt))


@register("stuck_state_frac")
def stuck_state_frac(s: EpisodeStats) -> float:
    if s.state is None or len(s.state) < 2:
        return float("nan")
    same = np.all(s.state[1:] == s.state[:-1], axis=1)
    return float(same.sum() / len(same))


@register("stuck_while_commanded")
def stuck_while_commanded(s: EpisodeStats) -> float:
    """Fraction of frames where the state is frozen but the action is not.

    This is the companion measurement stuck_state_frac needs, and the one
    that actually discriminates. A bit identical consecutive observation has
    two very different causes:

    - The arm was commanded to hold still and did. An encoder reporting the
      same quantised value during a pause before a grasp is expected in real
      teleoperation data, and is not a defect.
    - The arm was commanded to move and the state did not follow. That is a
      stalled sensor or a frozen recording pipeline, and it is a defect.

    stuck_state_frac cannot tell these apart, and neither can the run length
    structure of the stuck frames. Measured on 2026-09-12 across 10 datasets
    sampled from the census, 8 were consistent with the innocent cause, and
    values from 0.20 to 0.99 appeared on both sides. The clearest few-long-runs
    shape in the sample turned out innocent once the action channel was read.
    Only this comparison sorted them. See
    results/2026-09-12-stuck-state-check.md.

    Returns nan when the two channels cannot be compared, either because one
    is absent or because their widths differ. Two of the ten sampled datasets
    were in that position, which is a structural condition the flag itself
    cannot see, so it is reported as nan rather than as a number.
    """
    if s.state is None or s.action is None:
        return float("nan")
    if s.state.shape != s.action.shape or len(s.state) < 2:
        return float("nan")
    state_frozen = np.all(s.state[1:] == s.state[:-1], axis=1)
    action_moving = ~np.all(s.action[1:] == s.action[:-1], axis=1)
    return float((state_frozen & action_moving).sum() / len(state_frozen))


@register("identity_frac")
def identity_frac(s: EpisodeStats) -> float:
    if s.state is None or s.action is None or s.state.shape != s.action.shape:
        return float("nan")
    eq = np.all(np.isclose(s.action, s.state, atol=0.0), axis=1)
    return float(eq.sum() / len(eq))


@register("ts_nonmonotonic")
def ts_nonmonotonic(s: EpisodeStats) -> float:
    """True when any timestamp diff in the episode is zero or negative.

    Ported unchanged from the per episode body of audit_frame in
    ledger/audit.py, where an episode with fewer than two timestamps is
    simply not counted rather than treated as unknown, so this returns
    0.0 or 1.0 and never nan.
    """
    if s.ts is None or len(s.ts) < 2:
        return 0.0
    return float(np.any(np.diff(s.ts) <= 0))


@register("frame_gap")
def frame_gap(s: EpisodeStats) -> float:
    """True when the episode's frame_index skips or repeats a value.

    Ported unchanged from the per episode body of audit_frame in
    ledger/audit.py. Same convention as ts_nonmonotonic: too little data
    to judge counts as 0.0, not nan.
    """
    if s.frame_index is None or len(s.frame_index) < 2:
        return 0.0
    return float(np.any(np.diff(s.frame_index) != 1))


def run_checks(stats: EpisodeStats) -> dict[str, float]:
    return {name: c.fn(stats) for name, c in REGISTRY.items()}
