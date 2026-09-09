"""
ledger.audit  (M-02 LEDGER)
===========================

Thin CLI over ledger.report, ledger.census, ledger.sources and
ledger.hubclient. This module holds no audit arithmetic of its own:
audit_frame and summarise are re exports of ledger.report (RULING 2 and
RULING 10 in the plan record why they keep their v0 names and shapes),
and the two tier census logic lives entirely in ledger.census. What
stays here is argument parsing, the small amount of legacy Hub glue the
quick "--top N" / "--repos a,b" workflow still needs (list_top,
load_info, sample_parquet_paths, download, read_table), and the
offline synthetic demo.

Quick workflow, unchanged from v0 (samples the first parquet file(s) of
each named dataset; --max-mb filters by file size; --files all removes
the file cap entirely):
    python -m ledger.audit --top 20 --files 1 --out ledger_report.csv
    python -m ledger.audit --repos lerobot/svla_so101_pickplace,lerobot/aloha_sim_insertion_human
    python -m ledger.audit --demo

Census workflow, the full two tier Hub survey described in
docs/superpowers/specs/2026-09-05-ledger-census-design.md and
implemented in ledger.census:
    python -m ledger.audit --census both --sample-size 800 --seed 20260905 --out-dir census_out
    python -m ledger.audit --census deep --source local --local-root fixtures --repos acme/x --out-dir out

--source selects what ledger.census samples parquet data through:
stream (ledger.sources.StreamingSource, the default), download
(ledger.sources.DownloadSource), or local (ledger.sources.LocalSource,
which needs --local-root and is how this workflow stays testable
offline). --resume loads whichever tier's output file already exists
under --out-dir and skips repos already recorded there, keyed by
repo@revision (ledger.hubclient.HubClient captures the real Hub
revision from the X-Repo-Commit header, Ruling 9, so a dataset that
changed since the last run is re audited rather than silently skipped).
A record whose error names a rate limit is not treated as done, so a
later --resume run retries it instead of treating an exhausted backoff
as finished work (CRITICAL 2).

--files all is not the same guarantee on both layouts (IMPORTANT 3). A
per-episode (v2.x) dataset's file count is in info.json, so all is
exact. A packed (v3.x) dataset's file count is not, so all instead
probes file indices through the source's own existence check
(ledger.sources.*.exists; stream, download and local all have one) and
stops at the first missing one; a source that cannot check existence
falls back to one file and the report's note records that sample as
partial.

What the audit checks, per episode, then aggregated per dataset, is
documented in ledger.checks and ledger.report; see those modules for
the eight measurements (frac_bad_dt, stuck_state_frac, identity_frac,
lag_frames plus r_lag0/r_best, dup_episode_frac, ts_nonmonotonic_eps,
frame_gap_eps) and the flags they can raise. Flags are provisional
measurements, never a finding a dataset is "defective"; see
ledger.report.PROVISIONAL_HEADER.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np

from ledger.census import (
    CensusConfig,
    build_frame,
    draw_sample,
)
from ledger.census import (
    run_census as run_census_tiers,
)
from ledger.hubclient import API, HubClient, RateLimited
from ledger.report import DatasetReport, audit_frame, summarise, write_csv
from ledger.sources import DownloadSource, LocalSource, StreamingSource

__all__ = ["main", "parse_files", "demo", "_synthetic", "audit_frame", "summarise", "DatasetReport"]


# --------------------------------------------------------------------------- #
# Legacy Hub glue for the quick "--top" / "--repos" workflow. Kept as its
# own code path rather than folded into ledger.sources: --max-mb size
# filtering and the tree API traversal it uses have no equivalent in
# ledger.paths.derive_paths, which trades that away deliberately (see
# ledger/paths.py) because the tree endpoint is the one the anonymous
# rate limit punishes hardest. All optional so --demo runs offline.
# --------------------------------------------------------------------------- #
class TreeListingError(RuntimeError):
    """Listing a repository's data tree failed.

    Distinct from a listing that genuinely returned no parquet. v0 caught
    every exception here and returned an empty list, so a throttled repo
    produced a report that read as a dataset which had been audited and
    found to contain nothing. The tree endpoint is the call measured
    getting throttled hardest, so that was not a rare path.
    """


# The quick audit downloads whole parquet files. --files all parses to
# None, and a v2.0 dataset stores one parquet per episode: one real Hub
# dataset has 209,880 of them. Without a bound, --files all fills the disk.
# Lift it deliberately with --no-file-cap.
QUICK_AUDIT_FILE_CAP = 25


def _hf_api():
    """The HfApi used for tree listing. Indirected so tests can replace it."""
    from huggingface_hub import HfApi

    return HfApi()


def list_top(n: int, client: HubClient | None = None) -> list[str]:
    """The n most downloaded LeRobot datasets.

    Goes through HubClient so it inherits Retry-After compliance and
    backoff. v0 called urllib directly, so a single 429 ended the run.
    """
    client = client or HubClient()
    url = f"{API}/datasets?filter=LeRobot&sort=downloads&direction=-1&limit={n}"
    return [d["id"] for d in client.get_json(url)]


def load_info(repo: str, client: HubClient | None = None) -> dict | None:
    """The parsed meta/info.json for repo, or None if it is genuinely absent.

    None means a genuine 401, 403 or 404. A rate limit that outlives the
    backoff propagates as RateLimited rather than being folded into the
    same None, because "we were throttled" and "this dataset is gated or
    missing" are different facts and a human reading the report must be
    able to tell them apart.
    """
    return (client or HubClient()).get_info(repo)


def sample_parquet_paths(
    repo: str, max_files: int | None, max_mb: float, cap: bool = True
) -> list[str]:
    """Parquet paths for the quick audit, smallest chunk first.

    Raises TreeListingError if the listing itself failed. Returns an empty
    list only when the listing succeeded and there was nothing to audit,
    or when every file exceeds max_mb.
    """
    api = _hf_api()
    try:
        top = list(api.list_repo_tree(repo, repo_type="dataset", path_in_repo="data"))
        chunks = sorted(e.path for e in top if not e.path.endswith(".parquet"))
        entries = [e for e in top if e.path.endswith(".parquet")]
        if chunks:  # descend into the first chunk directory only; cheap on huge repos
            entries += list(api.list_repo_tree(repo, repo_type="dataset", path_in_repo=chunks[0]))
    except Exception as e:
        raise TreeListingError(f"cannot list the data tree for {repo}: {e}") from e
    files = [(e.path, getattr(e, "size", 0) or 0) for e in entries if e.path.endswith(".parquet")]
    files.sort()  # chunk-000/file-000 (v3) or chunk-000/episode_000000 (v2.x) first
    small = [p for p, size in files if size <= max_mb * 1e6]
    if files and not small:
        print(
            f"  [skip] {repo}: smallest parquet exceeds --max-mb {max_mb:.0f}; "
            f"raise it or run the census instead",
            file=sys.stderr,
        )
        return []
    if max_files is None:
        if cap and len(small) > QUICK_AUDIT_FILE_CAP:
            print(
                f"  [cap] {repo}: {len(small)} parquet files, auditing the first "
                f"{QUICK_AUDIT_FILE_CAP}. Pass --no-file-cap to audit every one, "
                f"or use the census for a whole population survey.",
                file=sys.stderr,
            )
            return small[:QUICK_AUDIT_FILE_CAP]
        return small
    return small[:max_files]


def download(repo: str, path: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, path, repo_type="dataset")


def read_table(local_path: str):
    import pyarrow.parquet as pq

    return pq.read_table(local_path).to_pandas()


def audit_repo(repo: str, max_files: int | None, max_mb: float, cap: bool = True) -> DatasetReport:
    """Audit one named dataset the quick way, by downloading its parquet.

    Every failure is recorded on the report rather than raised, so one bad
    repository cannot end a run over many. A rate limit and a failed tree
    listing are recorded in `error`, which keeps them distinguishable from
    the `note` a genuinely absent dataset gets.
    """
    try:
        info = load_info(repo)
    except RateLimited as e:
        return DatasetReport(repo=repo, error=f"RateLimited: {e}")
    if info is None:
        return DatasetReport(repo=repo, note="no meta/info.json (gated, missing, or not LeRobot)")
    fps = float(info.get("fps", 0) or 0)
    try:
        paths = sample_parquet_paths(repo, max_files, max_mb, cap=cap)
    except TreeListingError as e:
        return DatasetReport(repo=repo, error=f"TreeListingError: {e}")
    parts = []
    for path in paths:
        local = None
        try:
            local = download(repo, path)
            parts.append(audit_frame(read_table(local), fps))
        except Exception as e:
            print(f"  [warn] {repo}/{path}: {e}", file=sys.stderr)
        finally:
            # Best effort. v0 left every downloaded parquet in the cache
            # forever, which is what made --files all a disk hazard.
            if local:
                with contextlib.suppress(OSError):
                    Path(local).unlink()
    return summarise(repo, info, parts)


# --------------------------------------------------------------------------- #
# Synthetic demo: inject defects, prove the checks see them. Kept here
# rather than in ledger.synth: ledger.synth.make_episodes is the shared
# generator every other test in the suite draws on, with its own
# n_joints/seed/noise knobs; _synthetic is v0's own narrower signature,
# preserved verbatim (down to its fixed seed 0 and 6 joints) so that
# tests/test_audit.py, which imports it by name, keeps testing the
# actual thing v0 shipped rather than a reimplementation of it.
# --------------------------------------------------------------------------- #
def _synthetic(
    fps: float = 30.0, n_eps: int = 6, T: int = 300, lag: int = 2, defect: str = ""
) -> pandas.DataFrame:  # noqa: F821
    import pandas as pd

    rng = np.random.default_rng(0)
    rows = []
    for e in range(n_eps):
        t = np.arange(T) / fps
        # smooth random joint trajectory as the commanded action
        base = np.cumsum(rng.normal(0, 0.02, size=(T + lag + 5, 6)), axis=0)
        action = base[lag : T + lag]  # command at t
        state = base[:T] + rng.normal(0, 1e-3, size=(T, 6))  # reaches it `lag` frames later
        if defect == "identity":
            action = state.copy()
        if defect == "drops" and e % 2 == 0:
            keep = np.ones(T, bool)
            keep[rng.choice(T, size=T // 8, replace=False)] = False
            t, action, state = t[keep], action[keep], state[keep]
        if defect == "stuck" and e % 2 == 0:
            state[50:200] = state[50]
        if defect == "swapped":
            action, state = state, action
        # The next two lines carry two defects that v0 shipped and that are
        # preserved on purpose, because tests/test_audit.py imports this
        # function by name to test what v0 actually did. rows_first_action is
        # referenced before any binding a reader can see, and the annotation
        # above names pandas before it is imported. Neither fires: e == 0 is
        # always the first iteration, so the name is bound before any e > 0
        # reads it, and the annotation is a string under
        # from __future__ import annotations. Fixing them here would make this
        # copy diverge from the v0 it exists to preserve. The corrected
        # generator is ledger/synth.py.
        if defect == "duplicate" and e > 0:
            action = rows_first_action  # noqa: F821
        if e == 0:
            rows_first_action = action  # noqa: F841
        for i in range(len(t)):
            rows.append(
                dict(
                    timestamp=np.float32(t[i]),
                    frame_index=i,
                    episode_index=e,
                    action=action[i].astype(np.float32),
                    **{"observation.state": state[i].astype(np.float32)},
                )
            )
    return pd.DataFrame(rows)


def demo(out: str | None = None) -> list[DatasetReport]:
    """Run the offline synthetic demo and print the flags table.

    When `out` is given, also writes the reports there via
    ledger.report.write_csv, PROVISIONAL_HEADER and all. This is how
    tests/golden/demo_report.csv is produced; a bare --demo with no
    --out, exactly like v0, only prints and touches no disk.
    """
    print("Synthetic demo: each row injects one defect; the flags column should name it.\n")
    rows = []
    for defect in ["", "identity", "drops", "stuck", "swapped", "duplicate"]:
        df = _synthetic(defect=defect)
        rep = summarise(
            f"synthetic/{defect or 'clean'}",
            {"codebase_version": "demo", "fps": 30},
            [audit_frame(df, 30.0)],
        )
        rows.append(rep)
    _print_table(rows)
    if out:
        write_csv(rows, out)
        print(f"\nwrote {out}")
    return rows


# --------------------------------------------------------------------------- #
# Census workflow: argument driven wiring over ledger.census.
# --------------------------------------------------------------------------- #
def _build_source(args):
    if args.source == "local":
        if not args.local_root:
            raise SystemExit("--source local requires --local-root")
        return LocalSource(args.local_root)
    if args.source == "download":
        return DownloadSource()
    return StreamingSource()


def run_census(args) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    source = _build_source(args)

    if args.repos:
        frame = [r for r in args.repos.split(",") if r]
    elif args.source == "local":
        print(
            "error: --source local requires --repos (there is no Hub to crawl for a frame)",
            file=sys.stderr,
        )
        return 2
    else:
        client = getattr(source, "client", None) or HubClient()
        frame = build_frame(client)

    sample = draw_sample(frame, args.sample_size, args.seed)
    # tier lives only on cfg from here on: which tiers actually run is
    # decided by ledger.census.run_census reading cfg.tier, not by this
    # function branching on args.census itself.
    cfg = CensusConfig(
        out_dir=out_dir,
        sample_size=args.sample_size,
        seed=args.seed,
        files_per_dataset=args.files,
        tier=args.census,
    )

    written = run_census_tiers(
        sample, source, cfg, out_dir=out_dir, resume=args.resume, force_unlock=args.force_unlock
    )

    print(f"census: {written} record(s) written under {out_dir}")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _print_table(reps: list[DatasetReport]) -> None:
    cols = [
        "repo",
        "codebase",
        "fps",
        "episodes_sampled",
        "frac_bad_dt",
        "stuck_state_frac",
        "identity_frac",
        "lag_frames",
        "r_lag0",
        "r_best",
        "dup_episode_frac",
        "flags",
    ]
    widths = {c: max(len(c), *(len(_fmt(getattr(r, c))) for r in reps)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in reps:
        print("  ".join(_fmt(getattr(r, c)).ljust(widths[c]) for c in cols))


def _fmt(v) -> str:
    if isinstance(v, float):
        return "nan" if not np.isfinite(v) else f"{v:.3f}"
    return str(v)


def parse_files(value: str) -> int | None:
    """argparse type for --files: the literal "all" means no cap
    (None, threaded through to sample_parquet_paths / ledger.census as
    "sample every file"); anything else must parse as an int. Raises
    argparse.ArgumentTypeError, not ValueError, so argparse reports it
    as a usage error rather than a traceback.
    """
    if value == "all":
        return None
    try:
        return int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"--files must be an integer or 'all', got {value!r}"
        ) from e


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--top", type=int, default=0, help="audit the N most-downloaded LeRobot datasets"
    )
    ap.add_argument(
        "--repos",
        type=str,
        default="",
        help="comma-separated repo ids; the quick audit's target list, or the census frame when given",
    )
    ap.add_argument(
        "--files",
        type=parse_files,
        default=1,
        help="parquet files to sample per dataset; an integer, or 'all' for every file. "
        "In the census workflow, 'all' is exact for a per-episode (v2.x) layout "
        "but for a packed (v3.x) layout it probes file indices via the source's "
        "existence check and stops at the first missing one, falling back to one "
        "file (noted as partial on the report) if the source cannot check",
    )
    ap.add_argument(
        "--max-mb",
        type=float,
        default=150.0,
        help="prefer parquet files under this size (quick workflow only)",
    )
    ap.add_argument(
        "--no-file-cap",
        action="store_true",
        help=(
            f"lift the quick audit's {QUICK_AUDIT_FILE_CAP} file ceiling. The quick "
            "workflow downloads whole parquet files and a v2.x dataset stores one per "
            "episode, so '--files all' without this is capped to protect the disk. "
            "For a whole population survey use --census instead."
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="CSV path for the quick workflow (default ledger_report.csv); "
        "with --demo, only written if this is given",
    )
    ap.add_argument("--demo", action="store_true", help="run the offline synthetic demo")

    census = ap.add_argument_group("census", "the two tier Hub survey; see ledger.census")
    census.add_argument(
        "--census",
        choices=("metadata", "deep", "both"),
        default=None,
        help="run the census instead of the quick audit",
    )
    census.add_argument(
        "--sample-size", type=int, default=800, help="datasets drawn from the frame"
    )
    census.add_argument("--seed", type=int, default=20260905, help="seed for the sample draw")
    census.add_argument(
        "--resume", action="store_true", help="skip repos already recorded under --out-dir"
    )
    census.add_argument(
        "--force-unlock",
        action="store_true",
        help=(
            "break a stale census lock on --out-dir. A crashed run leaves its lock "
            "behind on purpose, because two runs sharing an out-dir append to the "
            "same ledger and duplicate every record"
        ),
    )
    census.add_argument(
        "--out-dir",
        type=str,
        default="census_out",
        help="directory for metadata.jsonl / deep.jsonl",
    )
    census.add_argument(
        "--source",
        choices=("stream", "download", "local"),
        default="stream",
        help="where the census samples parquet data from",
    )
    census.add_argument(
        "--local-root", type=str, default=None, help="fixture root for --source local"
    )

    args = ap.parse_args(argv)

    if args.demo:
        demo(args.out)
        return 0
    if args.census:
        return run_census(args)

    repos = [r for r in args.repos.split(",") if r]
    if args.top:
        repos = list_top(args.top) + repos
    if not repos:
        ap.error("give --top N, --repos a,b, --demo or --census")
    reps: list[DatasetReport] = []
    for i, repo in enumerate(repos, 1):
        t0 = time.time()
        print(f"[{i}/{len(repos)}] {repo}", file=sys.stderr)
        reps.append(audit_repo(repo, args.files, args.max_mb, cap=not args.no_file_cap))
        print(
            f"      done in {time.time() - t0:.1f}s  flags={reps[-1].flags or '-'}", file=sys.stderr
        )
    _print_table(reps)
    out = args.out or "ledger_report.csv"
    write_csv(reps, out)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
