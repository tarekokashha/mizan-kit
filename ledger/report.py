"""
ledger.report  (M-02 LEDGER)
=============================

Aggregates per-episode measurements from ledger.checks into a per-dataset
DatasetReport, and adds the provisional-findings bookkeeping the census
needs on top of the v0 audit: a confirmed column a human fills in, plus
source and revision so a report can be traced back to where its data came
from, and an error field so a failed fetch is recorded rather than
crashing the run.

audit_frame(df, fps) mirrors ledger.audit.audit_frame exactly: given a
dataframe holding one or more episodes, it returns one counts dict for
that dataframe, not a ratio and not a list. summarise(repo, info, parts,
thresholds=None) takes a list of those dicts, one per sampled file, and
pools them the same way v0 does: sums of numerators over sums of
denominators, never a mean of per-part or per-episode ratios.

    frac_bad_dt       = sum(bad_dt) / sum(n_dt)
    stuck_state_frac  = sum(stuck) / sum(n_stuck)
    identity_frac     = sum(ident) / sum(n_ident)

each falling back to nan when its denominator is zero. A mean of
per-episode fractions gives a different, wrong, answer whenever episodes
have unequal length, which the "drops" defect and real Hub data both
produce. That is why audit_frame returns counts (numerators and
denominators) rather than the ratios ledger.checks.run_checks computes
per episode: ratios do not pool by averaging, so the numerator and
denominator behind each one are carried and summed here instead.

Per-episode parsing (episode_stats) and the two measurements that need a
whole episode or a whole dataset rather than one scalar (xcorr_lag,
action_head_hash) come from ledger.checks, per its own docstring: lag
and duplicate detection are not REGISTRY entries, and AGGREGATE_FLAGS
documents that they are computed here instead. ts_nonmonotonic and
frame_gap are already per-episode booleans in the registry, so summing
run_checks()'s 0.0/1.0 across episodes reproduces the v0 episode counts
directly.

Flags describe measurements, not findings. PROVISIONAL_HEADER says so
and is written as the first line of every CSV. This module never labels
a dataset as having a confirmed problem; DatasetReport.confirmed is
where a human records that after opening the data.

Two of those flags, large_lag and negative_lag, come from lag_frames,
which docs/calibration.md shows xcorr_lag does not reliably resolve
below 6 action dimensions (96.75 percent exact match at n_joints=2,
versus 0 to 2 mismatches per 6000 to 8000 trials, all near ties, at
n_joints 6 and above). Below 6 action dimensions, treat those two flags
as uninformative and confirm by hand rather than trusting the flag.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ledger.checks import action_head_hash, episode_stats, run_checks, xcorr_lag
from ledger.config import DEFAULT_THRESHOLDS

PROVISIONAL_HEADER = (
    "# LEDGER M-02 automated audit. Flags are provisional measurements, "
    "not findings. No dataset should be treated as having a confirmed "
    "problem until a human has opened it and filled in the confirmed "
    "column."
)


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
    confirmed: str = ""
    source: str = ""
    revision: str = ""
    error: str = ""


def audit_frame(df, fps: float) -> dict:
    """Run every check on a dataframe holding one or more episodes.

    Returns one counts dict for the whole dataframe, matching
    ledger.audit.audit_frame's shape and meaning exactly (RULING 2: this
    is not renamed and does not return a list). Build one of these per
    sampled parquet file, then pass the list of them to summarise().

    ts_nonmono and frame_gap are accumulated straight from
    run_checks()'s ts_nonmonotonic/frame_gap, since those are already
    per-episode 0.0/1.0 flags and summing them across episodes is an
    episode count, exactly like v0. frac_bad_dt, stuck_state_frac and
    identity_frac are per-episode ratios, and ratios do not pool by
    averaging, so their numerators (bad_dt, stuck, ident) and
    denominators (n_dt, n_stuck, n_ident) are counted here directly from
    the same EpisodeStats fields ledger.checks parses, using the same
    conditions the registry checks use, rather than multiplied back out
    of a ratio. lags/r0/rb and hashes come straight from ledger.checks'
    xcorr_lag and action_head_hash, the two measurements that are not
    REGISTRY entries because they need a whole episode (lag) or the
    whole dataset (duplicates) rather than one scalar.
    """
    out = dict(episodes=0, frames=int(len(df)), ts_nonmono=0, bad_dt=0, n_dt=0, dts=[],
               frame_gap=0, stuck=0, n_stuck=0, ident=0, n_ident=0, lags=[], r0=[], rb=[], hashes=[])
    if "episode_index" not in df:
        df = df.assign(episode_index=0)
    for _, ep in df.groupby("episode_index", sort=True):
        out["episodes"] += 1
        stats = episode_stats(ep, fps)
        flags = run_checks(stats)
        out["ts_nonmono"] += int(flags["ts_nonmonotonic"])
        out["frame_gap"] += int(flags["frame_gap"])
        if stats.ts is not None and len(stats.ts) > 1 and stats.fps and stats.fps > 0:
            expected = 1.0 / stats.fps
            dt = np.diff(stats.ts)
            out["bad_dt"] += int(np.sum(np.abs(dt - expected) > 0.25 * expected))
            out["n_dt"] += len(dt)
            out["dts"].extend(dt.tolist())
        if stats.state is not None and len(stats.state) > 1:
            same = np.all(stats.state[1:] == stats.state[:-1], axis=1)
            out["stuck"] += int(same.sum())
            out["n_stuck"] += len(same)
        if stats.state is not None and stats.action is not None and stats.state.shape == stats.action.shape:
            eq = np.all(np.isclose(stats.action, stats.state, atol=0.0), axis=1)
            out["ident"] += int(eq.sum())
            out["n_ident"] += len(eq)
            lag, r0, rb = xcorr_lag(stats.action, stats.state)
            if np.isfinite(rb):
                out["lags"].append(lag)
                out["r0"].append(r0)
                out["rb"].append(rb)
        h = action_head_hash(stats, n=50)
        if h is not None:
            out["hashes"].append(h)
    return out


def summarise(repo: str, info: dict | None, parts: list[dict],
             thresholds: dict | None = None) -> DatasetReport:
    """Pool a list of audit_frame() counts dicts into one DatasetReport.

    Aggregation is pooled exactly as ledger.audit.summarise does: sums of
    numerators over sums of denominators, not means of per-part ratios.
    See RULING 6 in the task dispatch and the module docstring above.

    thresholds defaults to ledger.config.DEFAULT_THRESHOLDS, the same
    keys ledger.audit.THRESHOLDS used in v0 (frac_bad_dt,
    stuck_state_frac, identity_frac, lag_large, dup_episode_frac), so a
    caller can pass a calibrated set (see docs/calibration.md) without
    editing this function.
    """
    th = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
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
    if np.isfinite(rep.frac_bad_dt) and rep.frac_bad_dt > th["frac_bad_dt"]:
        flags.append("bad_dt")
    if rep.frame_gap_eps:
        flags.append("frame_gaps")
    if np.isfinite(rep.stuck_state_frac) and rep.stuck_state_frac > th["stuck_state_frac"]:
        flags.append("stuck_state")
    if np.isfinite(rep.identity_frac) and rep.identity_frac > th["identity_frac"]:
        flags.append("action_equals_state")
    if np.isfinite(rep.lag_frames):
        if rep.lag_frames < 0:
            flags.append("negative_lag")
        elif rep.lag_frames >= th["lag_large"]:
            flags.append("large_lag")
    if np.isfinite(rep.dup_episode_frac) and rep.dup_episode_frac > th["dup_episode_frac"]:
        flags.append("duplicate_episodes")
    rep.flags = "|".join(flags)
    return rep


def write_csv(reports: list[DatasetReport], path) -> None:
    """Write reports as CSV, with PROVISIONAL_HEADER as the first line.

    The header line is a comment (it starts with '#'), so it does not
    become a spurious extra column when the CSV is read back with
    pandas or csv.DictReader; both skip a leading '#' line by default
    only if told to, so a reader of this file should pass
    comment='#' (pandas) or skip the first line itself.
    """
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(PROVISIONAL_HEADER + "\n")
        if not reports:
            return
        fieldnames = list(asdict(reports[0]).keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in reports:
            w.writerow(asdict(r))


def append_jsonl(report: DatasetReport, path) -> None:
    """Append one report as a JSON line, for a long-running census run."""
    path = Path(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(report)) + "\n")


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
