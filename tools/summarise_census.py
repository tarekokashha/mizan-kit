"""Summarise a metadata census with confidence intervals.

Every proportion goes through cairo_protocol.wilson_interval, per the
kit's rule that a rate is never reported bare.
"""

import sys
from collections import Counter
from pathlib import Path

from cairo_protocol.stats import wilson_interval
from ledger.report import read_jsonl

rows = read_jsonl(Path(sys.argv[1] if len(sys.argv) > 1 else "census_run/metadata.jsonl"))
n = len(rows)
repos = [r["repo"] for r in rows]
dupes = len([r for r in set(repos) if repos.count(r) > 1])

print(f"records {n}, unique repos {len(set(repos))}, duplicated {dupes}")
print("seed 20260905, sample drawn from the full LeRobot frame\n")


def table(title, counter, total):
    print(f"{title}")
    print(f"  {'value':<24} {'n':>5} {'rate':>7}   95% Wilson interval")
    for k, c in counter.most_common():
        lo, hi = wilson_interval(c, total)
        print(f"  {str(k):<24} {c:>5} {c / total:>7.3f}   [{lo:.3f}, {hi:.3f}]")
    print()


ok = [r for r in rows if not r["error"]]
errs = [r for r in rows if r["error"]]
if errs:
    lo, hi = wilson_interval(len(errs), n)
    print(
        f"reachable {len(ok)}/{n}; errored {len(errs)} rate {len(errs) / n:.3f} [{lo:.3f}, {hi:.3f}]"
    )
    table("error kinds", Counter(e["error"].split(":")[0] for e in errs), n)

table("codebase_version", Counter(r["codebase"] or "(absent)" for r in ok), len(ok))
table("layout_family", Counter(r["layout_family"] or "(absent)" for r in ok), len(ok))
table("fps", Counter(r["fps"] for r in ok), len(ok))

empty = [r for r in ok if (r["total_episodes"] or 0) == 0]
lo, hi = wilson_interval(len(empty), len(ok))
print(
    f"datasets declaring zero episodes: {len(empty)}/{len(ok)} "
    f"rate {len(empty) / len(ok):.3f} [{lo:.3f}, {hi:.3f}]"
)

eps = sorted(r["total_episodes"] or 0 for r in ok)
frames = sorted(r["total_frames"] or 0 for r in ok)


def q(xs, p):
    return xs[min(len(xs) - 1, int(p * len(xs)))]


print(f"\nepisodes  median {q(eps, 0.5)}  p90 {q(eps, 0.9)}  max {max(eps)}")
print(f"frames    median {q(frames, 0.5)}  p90 {q(frames, 0.9)}  max {max(frames)}")
