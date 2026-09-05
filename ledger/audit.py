"""
ledger.audit  (M-02 LEDGER, v0)
===============================

Temporal-integrity audit for LeRobot datasets on the Hugging Face Hub.

Run on the most-downloaded datasets:
    python -m ledger.audit --top 20 --files 1 --out ledger_report.csv

Run on named repositories:
    python -m ledger.audit --repos lerobot/svla_so101_pickplace,lerobot/aloha_sim_insertion_human

Run the synthetic demo (no network), which injects known defects and shows
that the checks catch them:
    python -m ledger.audit --demo

What it checks, per episode, then aggregated per dataset
--------------------------------------------------------
ts_nonmonotonic_eps   episodes whose timestamp is not strictly increasing
frac_bad_dt           fraction of frame intervals outside +/-25% of 1/fps
                      (dropped frames, duplicated frames, or a wrong fps)
dt_jitter_ratio       median absolute deviation of dt divided by 1/fps
frame_gap_eps         episodes whose frame_index has holes
stuck_state_frac      fraction of consecutive frames whose observation.state
                      is bit-identical (a stalled sensor stream)
identity_frac         fraction of frames where action == observation.state
                      exactly (the "action is just the state" recording bug)
lag_frames            the lag k (frames) at which action[t] best matches
                      state[t+k]; healthy position-controlled arms show a
                      small positive lag; 0 with identity_frac high is a bug;
                      negative means the columns are probably swapped
r_lag0 / r_best       action-state correlation at lag 0 and at the best lag
dup_episode_frac      fraction of sampled episodes whose first 50 action
                      frames are identical to another episode's

A `flags` column summarises which of these crossed the thresholds in
THRESHOLDS below. Thresholds are deliberately loose for v0; calibrate them
on known-good data (DROID, your own M-03 recordings) before naming anyone.

This is v0. It samples the first parquet file(s) of each dataset, so the
numbers are estimates of prevalence, not a full census. The paper's
prevalence table needs `--files all` and the full Hub list, which the
overnight loop can run.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict, field

import numpy as np

THRESHOLDS = {
    "frac_bad_dt": 0.05,
    "stuck_state_frac": 0.20,
    "identity_frac": 0.50,
    "lag_large": 3,
    "dup_episode_frac": 0.0,
}

LAGS = list(range(-5, 11))


# --------------------------------------------------------------------------- #
# Hub access (all optional so the demo runs offline)
# --------------------------------------------------------------------------- #
def list_top(n: int) -> list[str]:
    url = f"https://huggingface.co/api/datasets?filter=LeRobot&sort=downloads&direction=-1&limit={n}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return [d["id"] for d in json.load(r)]


def load_info(repo: str) -> dict | None:
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return json.load(r)
    except Exception as e:  # gated, missing, or not a LeRobot dataset
        print(f"  [skip] {repo}: {e}", file=sys.stderr)
        return None


def sample_parquet_paths(repo: str, max_files: int, max_mb: float) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        top = list(api.list_repo_tree(repo, repo_type="dataset", path_in_repo="data"))
        chunks = sorted(e.path for e in top if not e.path.endswith(".parquet"))
        entries = [e for e in top if e.path.endswith(".parquet")]
        if chunks:  # descend into the first chunk directory only; cheap on huge repos
            entries += list(api.list_repo_tree(repo, repo_type="dataset", path_in_repo=chunks[0]))
    except Exception as e:
        print(f"  [skip] {repo}: cannot list tree ({e})", file=sys.stderr)
        return []
    files = [(e.path, getattr(e, "size", 0) or 0) for e in entries if e.path.endswith(".parquet")]
    files.sort()  # chunk-000/file-000 (v3) or chunk-000/episode_000000 (v2.x) first
    small = [p for p, s in files if s <= max_mb * 1e6]
    if not small:
        print(f"  [skip] {repo}: smallest parquet exceeds --max-mb {max_mb:.0f}; raise it or run in the overnight loop", file=sys.stderr)
        return []
    return small[:max_files]


def download(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def read_table(local_path: str):
    import pyarrow.parquet as pq

    return pq.read_table(local_path).to_pandas()


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #
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


def audit_repo(repo: str, max_files: int, max_mb: float) -> DatasetReport:
    info = load_info(repo)
    if info is None:
        return DatasetReport(repo=repo, note="no meta/info.json (gated, missing, or not LeRobot)")
    fps = float(info.get("fps", 0) or 0)
    paths = sample_parquet_paths(repo, max_files, max_mb)
    parts = []
    for p in paths:
        try:
            local = download(repo, p)
            df = read_table(local)
            parts.append(audit_frame(df, fps))
        except Exception as e:
            print(f"  [warn] {repo}/{p}: {e}", file=sys.stderr)
    return summarise(repo, info, parts)


# --------------------------------------------------------------------------- #
# Synthetic demo: inject defects, prove the checks see them
# --------------------------------------------------------------------------- #
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


def demo() -> None:
    print("Synthetic demo: each row injects one defect; the flags column should name it.\n")
    rows = []
    for defect in ["", "identity", "drops", "stuck", "swapped", "duplicate"]:
        df = _synthetic(defect=defect)
        rep = summarise(f"synthetic/{defect or 'clean'}", {"codebase_version": "demo", "fps": 30}, [audit_frame(df, 30.0)])
        rows.append(rep)
    _print_table(rows)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_table(reps: list[DatasetReport]) -> None:
    cols = ["repo", "codebase", "fps", "episodes_sampled", "frac_bad_dt", "stuck_state_frac",
            "identity_frac", "lag_frames", "r_lag0", "r_best", "dup_episode_frac", "flags"]
    widths = {c: max(len(c), *(len(_fmt(getattr(r, c))) for r in reps)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in reps:
        print("  ".join(_fmt(getattr(r, c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.3f}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--top", type=int, default=0, help="audit the N most-downloaded LeRobot datasets")
    ap.add_argument("--repos", type=str, default="", help="comma-separated repo ids to audit")
    ap.add_argument("--files", type=int, default=1, help="parquet files to sample per dataset")
    ap.add_argument("--max-mb", type=float, default=150.0, help="prefer parquet files under this size")
    ap.add_argument("--out", type=str, default="ledger_report.csv")
    ap.add_argument("--demo", action="store_true", help="run the offline synthetic demo")
    args = ap.parse_args(argv)

    if args.demo:
        demo()
        return 0
    repos = [r for r in args.repos.split(",") if r]
    if args.top:
        repos = list_top(args.top) + repos
    if not repos:
        ap.error("give --top N, --repos a,b or --demo")
    reps: list[DatasetReport] = []
    for i, repo in enumerate(repos, 1):
        t0 = time.time()
        print(f"[{i}/{len(repos)}] {repo}", file=sys.stderr)
        reps.append(audit_repo(repo, args.files, args.max_mb))
        print(f"      done in {time.time() - t0:.1f}s  flags={reps[-1].flags or '-'}", file=sys.stderr)
    _print_table(reps)
    import csv

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(reps[0]).keys()))
        w.writeheader()
        for r in reps:
            w.writerow(asdict(r))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
