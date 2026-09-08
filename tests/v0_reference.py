"""Frozen snapshot of mizan-kit v0 as shipped on 2026-09-02.

This is a verbatim copy of the check and aggregation code that lived in
ledger/audit.py before the M-02 LEDGER census refactor: DatasetReport,
THRESHOLDS, LAGS, _stack, _xcorr_lag, audit_frame, summarise and
_synthetic. It is not imported from ledger.audit and it does not import
ledger.audit, on purpose.

Why this file exists (RULING 10): tests/test_report.py asserts that the
new ledger.report pipeline reproduces v0 exactly, for every synthetic
defect. That assertion is only meaningful if the thing it compares
against cannot change when ledger/audit.py is refactored. Once
ledger/audit.py is rewritten into a thin CLI over the new modules, its
own audit_frame and summarise become re exports of ledger.report, so
comparing against ledger.audit at that point would compare
ledger.report to itself: always equal, no matter what the refactor
does. This file is the fix: a copy frozen at the exact behaviour v0
shipped with, kept solely to prove the refactor did not change that
behaviour.

This file must never be edited to make a test pass. If a test against
this file fails, the bug is in the new code, not in this file. The only
legitimate reason to touch this file is discovering the copy was not
actually verbatim when it was written.
"""
from __future__ import annotations

import hashlib

import numpy as np
from dataclasses import dataclass

THRESHOLDS = {
    "frac_bad_dt": 0.05,
    "stuck_state_frac": 0.20,
    "identity_frac": 0.50,
    "lag_large": 3,
    "dup_episode_frac": 0.0,
}

LAGS = list(range(-5, 11))


@dataclass
class DatasetReport:
    repo: str
    codebase: str = ""
    fps: float = float("nan")
    episodes_sampled: int = 0
    frames_sampled: int = 0
    ts_nonmonotonic_eps: int = 0
    frac_bad_dt: float = float("nan")
    dt_jitter_ratio: float = float("nan")
    frame_gap_eps: int = 0
    stuck_state_frac: float = float("nan")
    identity_frac: float = float("nan")
    lag_frames: float = float("nan")
    r_lag0: float = float("nan")
    r_best: float = float("nan")
    dup_episode_frac: float = float("nan")
    flags: str = ""
    note: str = ""


def _stack(col) -> np.ndarray | None:
    try:
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in col])
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def _xcorr_lag(action: np.ndarray, state: np.ndarray) -> tuple[int, float, float]:
    """Return (best_lag, r_at_lag0, r_at_best) with r averaged over dimensions."""
    T = len(action)
    if T < 30:
        return 0, float("nan"), float("nan")
    a = action - action.mean(0)
    s = state - state.mean(0)
    sa = a.std(0) + 1e-9
    ss = s.std(0) + 1e-9
    scores = {}
    for k in LAGS:
        if k >= 0:
            aa, st = a[: T - k], s[k:]
        else:
            aa, st = a[-k:], s[: T + k]
        if len(aa) < 20:
            continue
        r = ((aa * st).mean(0)) / (sa * ss)
        scores[k] = float(np.nanmean(r))
    if not scores:
        return 0, float("nan"), float("nan")
    best = max(scores, key=scores.get)
    return best, scores.get(0, float("nan")), scores[best]


def audit_frame(df, fps: float) -> dict:
    """Run every check on a dataframe holding one or more episodes."""
    out = dict(episodes=0, frames=int(len(df)), ts_nonmono=0, bad_dt=0, n_dt=0, dts=[],
               frame_gap=0, stuck=0, n_stuck=0, ident=0, n_ident=0, lags=[], r0=[], rb=[], hashes=[])
    if "episode_index" not in df:
        df = df.assign(episode_index=0)
    has_state = "observation.state" in df
    has_action = "action" in df
    expected = 1.0 / fps if fps and fps > 0 else float("nan")
    for _, ep in df.groupby("episode_index", sort=True):
        out["episodes"] += 1
        if "timestamp" in ep:
            ts = np.asarray(ep["timestamp"], dtype=np.float64)
            if len(ts) > 1:
                dt = np.diff(ts)
                if np.any(dt <= 0):
                    out["ts_nonmono"] += 1
                if np.isfinite(expected):
                    out["bad_dt"] += int(np.sum(np.abs(dt - expected) > 0.25 * expected))
                    out["n_dt"] += len(dt)
                    out["dts"].extend(dt.tolist())
        if "frame_index" in ep:
            fi = np.asarray(ep["frame_index"], dtype=np.int64)
            if len(fi) > 1 and np.any(np.diff(fi) != 1):
                out["frame_gap"] += 1
        st = _stack(ep["observation.state"]) if has_state else None
        ac = _stack(ep["action"]) if has_action else None
        if st is not None and len(st) > 1:
            same = np.all(st[1:] == st[:-1], axis=1)
            out["stuck"] += int(same.sum())
            out["n_stuck"] += len(same)
        if st is not None and ac is not None and st.shape == ac.shape:
            eq = np.all(np.isclose(ac, st, atol=0.0), axis=1)
            out["ident"] += int(eq.sum())
            out["n_ident"] += len(eq)
            lag, r0, rb = _xcorr_lag(ac, st)
            if np.isfinite(rb):
                out["lags"].append(lag)
                out["r0"].append(r0)
                out["rb"].append(rb)
        if ac is not None:
            head = np.round(ac[:50], 4).tobytes()
            out["hashes"].append(hashlib.md5(head).hexdigest())
    return out


def summarise(repo: str, info: dict | None, parts: list[dict]) -> DatasetReport:
    rep = DatasetReport(repo=repo)
    if info:
        rep.codebase = str(info.get("codebase_version", ""))
        rep.fps = float(info.get("fps", float("nan")))
    if not parts:
        rep.note = "no data sampled"
        return rep
    eps = sum(p["episodes"] for p in parts)
    rep.episodes_sampled = eps
    rep.frames_sampled = sum(p["frames"] for p in parts)
    rep.ts_nonmonotonic_eps = sum(p["ts_nonmono"] for p in parts)
    n_dt = sum(p["n_dt"] for p in parts)
    rep.frac_bad_dt = (sum(p["bad_dt"] for p in parts) / n_dt) if n_dt else float("nan")
    dts = np.concatenate([np.asarray(p["dts"]) for p in parts if p["dts"]]) if any(p["dts"] for p in parts) else None
    if dts is not None and np.isfinite(rep.fps) and rep.fps > 0:
        mad = float(np.median(np.abs(dts - np.median(dts))))
        rep.dt_jitter_ratio = mad * rep.fps
    rep.frame_gap_eps = sum(p["frame_gap"] for p in parts)
    n_stuck = sum(p["n_stuck"] for p in parts)
    rep.stuck_state_frac = (sum(p["stuck"] for p in parts) / n_stuck) if n_stuck else float("nan")
    n_ident = sum(p["n_ident"] for p in parts)
    rep.identity_frac = (sum(p["ident"] for p in parts) / n_ident) if n_ident else float("nan")
    lags = sum((p["lags"] for p in parts), [])
    if lags:
        rep.lag_frames = float(np.median(lags))
        rep.r_lag0 = float(np.nanmean(sum((p["r0"] for p in parts), [])))
        rep.r_best = float(np.nanmean(sum((p["rb"] for p in parts), [])))
    hashes = sum((p["hashes"] for p in parts), [])
    if hashes:
        rep.dup_episode_frac = 1.0 - len(set(hashes)) / len(hashes)
    flags = []
    if rep.ts_nonmonotonic_eps:
        flags.append("ts_nonmonotonic")
    if np.isfinite(rep.frac_bad_dt) and rep.frac_bad_dt > THRESHOLDS["frac_bad_dt"]:
        flags.append("bad_dt")
    if rep.frame_gap_eps:
        flags.append("frame_gaps")
    if np.isfinite(rep.stuck_state_frac) and rep.stuck_state_frac > THRESHOLDS["stuck_state_frac"]:
        flags.append("stuck_state")
    if np.isfinite(rep.identity_frac) and rep.identity_frac > THRESHOLDS["identity_frac"]:
        flags.append("action_equals_state")
    if np.isfinite(rep.lag_frames):
        if rep.lag_frames < 0:
            flags.append("negative_lag")
        elif rep.lag_frames >= THRESHOLDS["lag_large"]:
            flags.append("large_lag")
    if np.isfinite(rep.dup_episode_frac) and rep.dup_episode_frac > THRESHOLDS["dup_episode_frac"]:
        flags.append("duplicate_episodes")
    rep.flags = "|".join(flags)
    return rep


def _synthetic(fps: float = 30.0, n_eps: int = 6, T: int = 300, lag: int = 2, defect: str = "") -> "pandas.DataFrame":
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for e in range(n_eps):
        t = np.arange(T) / fps
        # smooth random joint trajectory as the commanded action
        base = np.cumsum(rng.normal(0, 0.02, size=(T + lag + 5, 6)), axis=0)
        action = base[lag: T + lag]                                      # command at t
        state = base[: T] + rng.normal(0, 1e-3, size=(T, 6))            # reaches it `lag` frames later
        if defect == "identity":
            action = state.copy()
        if defect == "drops" and e % 2 == 0:
            keep = np.ones(T, bool); keep[rng.choice(T, size=T // 8, replace=False)] = False
            t, action, state = t[keep], action[keep], state[keep]
        if defect == "stuck" and e % 2 == 0:
            state[50:200] = state[50]
        if defect == "swapped":
            action, state = state, action
        if defect == "duplicate" and e > 0:
            action = rows_first_action
        if e == 0:
            rows_first_action = action
        for i in range(len(t)):
            rows.append(dict(timestamp=np.float32(t[i]), frame_index=i, episode_index=e,
                             action=action[i].astype(np.float32), **{"observation.state": state[i].astype(np.float32)}))
    return pd.DataFrame(rows)
