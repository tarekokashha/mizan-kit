"""Show the evidence behind a flag, so a human can confirm or refute it.

A LEDGER flag is a measurement. `CLAUDE.md` requires a human to open the
data before it becomes a finding, and this is the tool for doing that in a
minute rather than an afternoon. It fetches one dataset's first parquet,
prints the columns the flag was raised on, and says nothing about whether
the dataset is defective. That judgement is the human's, and it belongs in
the `confirmed` column.

    python tools/inspect_flagged.py qingshu123/Airbot_MMK2_storage_egg_white_box
    python tools/inspect_flagged.py --all-flagged results/2026-09-11-census-400-deep.jsonl
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from ledger.hubclient import HubClient
from ledger.paths import derive_paths
from ledger.report import read_jsonl
from ledger.sources import StreamingSource

CONFIRMABLE = ("action_equals_state", "negative_lag", "duplicate_episodes")


def inspect(repo: str, source: StreamingSource) -> int:
    print(f"\n{'=' * 72}\n{repo}\nhttps://huggingface.co/datasets/{repo}\n{'=' * 72}")
    info = source.info(repo)
    if info is None:
        print("  no info.json: gated, private, or removed since the census")
        return 1
    print(
        f"  codebase {info.get('codebase_version')}  fps {info.get('fps')}  "
        f"episodes {info.get('total_episodes')}"
    )

    paths = derive_paths(info, limit=1)
    if not paths:
        print("  no parquet path derivable")
        return 1
    df = source.frames(repo, paths[0])
    print(f"  read {paths[0]}: {len(df)} frames, columns {list(df.columns)}")

    if "action" not in df or "observation.state" not in df:
        print("  action or observation.state absent, nothing to compare")
        return 0

    a = np.stack([np.asarray(v, dtype=np.float64) for v in df["action"]])
    s = np.stack([np.asarray(v, dtype=np.float64) for v in df["observation.state"]])

    print(f"\n  action dims {a.shape[1]}, state dims {s.shape[1]}")
    if a.shape != s.shape:
        # ledger.checks guards this too and returns nan rather than comparing.
        # A dataset whose action and state have different widths cannot be
        # compared elementwise, and the mismatch is itself worth a look.
        print("  action and observation.state have DIFFERENT widths, so they")
        print("  cannot be compared frame by frame. The audit reports nan for")
        print("  identity on this dataset rather than a number. The mismatch")
        print("  is itself the thing to investigate.")
        return 0
    identical = np.all(np.isclose(a, s, atol=0.0), axis=1)
    print(
        f"  frames where action == observation.state EXACTLY: "
        f"{identical.sum()} of {len(identical)} ({identical.mean():.3f})"
    )
    print("\n  first 3 frames, action vs state:")
    for i in range(min(3, len(a))):
        print(f"    [{i}] action {np.round(a[i][:6], 5).tolist()}")
        print(f"        state  {np.round(s[i][:6], 5).tolist()}")
        print(f"        equal: {bool(identical[i])}")

    if a.shape[1] < 6:
        print(f"\n  NOTE: {a.shape[1]} action dimensions. docs/calibration.md shows the")
        print("  lag estimator is unreliable below 6, so any lag flag on this")
        print("  dataset is uninformative and should not be confirmed on that basis.")

    print("\n  This tool asserts nothing about whether this dataset is defective.")
    print("  Record your judgement in the confirmed column.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", nargs="?", help="a single dataset to inspect")
    ap.add_argument(
        "--all-flagged",
        metavar="LEDGER",
        help="inspect every dataset in LEDGER carrying a confirmable flag",
    )
    args = ap.parse_args(argv)

    source = StreamingSource(HubClient(min_interval=1.0))
    if args.all_flagged:
        rows = read_jsonl(args.all_flagged)
        todo = [
            r["repo"]
            for r in rows
            if not r["error"] and any(f in r["flags"].split("|") for f in CONFIRMABLE)
        ]
        print(f"{len(todo)} dataset(s) carrying a confirmable flag", file=sys.stderr)
        for repo in todo:
            inspect(repo, source)
        return 0
    if not args.repo:
        ap.error("give a repo id or --all-flagged LEDGER")
    return inspect(args.repo, source)


if __name__ == "__main__":
    raise SystemExit(main())
