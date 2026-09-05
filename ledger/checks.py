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
from dataclasses import dataclass
from typing import Callable, Iterable

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
    fn: Callable[["EpisodeStats"], float]
    threshold_key: str | None
    higher_is_worse: bool


REGISTRY: dict[str, Check] = {}


def register(name: str, threshold_key: str | None = None,
             higher_is_worse: bool = True):
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


def _stack(col) -> np.ndarray | None:
    try:
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in col])
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def episode_stats(ep, fps: float) -> EpisodeStats:
    ts = np.asarray(ep["timestamp"], dtype=np.float64) if "timestamp" in ep else None
    fi = np.asarray(ep["frame_index"], dtype=np.int64) if "frame_index" in ep else None
    ac = _stack(ep["action"]) if "action" in ep else None
    st = _stack(ep["observation.state"]) if "observation.state" in ep else None
    return EpisodeStats(len(ep), fps, ts, fi, ac, st)


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
            aa, st = a[:T - k], s[k:]
        else:
            aa, st = a[-k:], s[:T + k]
        if len(aa) < 20:
            continue
        scores[k] = float(np.nanmean(((aa * st).mean(0)) / (sa * ss)))
    if not scores:
        return 0, float("nan"), float("nan")
    best = max(scores, key=scores.get)
    return best, scores.get(0, float("nan")), scores[best]


def action_head_hash(s: EpisodeStats, n: int = 50) -> str | None:
    if s.action is None:
        return None
    return hashlib.md5(np.round(s.action[:n], 4).tobytes()).hexdigest()


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
