"""
ledger.census  (M-02 LEDGER)
=============================

Two tier Hub census. The metadata tier fetches only each repo's
info.json, cheap enough to run over the whole frame first and confirm a
repo is reachable and looks like a LeRobot dataset before the deep tier
spends any bandwidth on it. The deep tier samples actual parquet data
through a SampleSource and runs the full audit via ledger.report, one
DatasetReport per dataset.

Sampling is seeded and sorts the frame before drawing, so a draw does
not depend on the order the Hub happened to return datasets in: only
the sorted frame and the seed determine the result. That is what makes
the resulting prevalence table a pre-registered claim, in the same
spirit as protocols/PROTOCOL_TEMPLATE.md, rather than a post hoc one.
Every reported rate carries a Wilson confidence interval from
cairo_protocol.stats; prevalence() never returns a bare rate.

Both tiers write to a JSONL ledger keyed by repo@revision. Every record
is flushed to disk immediately after it is computed, not batched to the
end, so a crash costs at most the one dataset being processed when it
happened. load_done() reads that ledger back into the set of keys
already written, tolerating a truncated final line, exactly what a
crash mid write leaves behind, by skipping it rather than raising. A
caller passes that set back in as `done` on the next run to resume.

Graceful degradation is mandatory in the deep tier: a missing repo, an
absent info.json, an unreadable parquet file, or any other raised
exception is caught and recorded into the report's error field rather
than raised, so one bad dataset can never end the run. Neither tier
ever labels a dataset as having a confirmed problem; that stays a human
judgement recorded on DatasetReport.confirmed, per ledger.report.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cairo_protocol.stats import wilson_interval
from ledger.report import DatasetReport, append_jsonl, audit_frame, summarise


@dataclass
class CensusConfig:
    out_dir: Path
    sample_size: int = 800
    seed: int = 20260905
    files_per_dataset: int = 1
    max_datasets: int | None = None
    tier: str = "both"


def _key(repo: str, revision: str = "") -> str:
    """The repo@revision key both the ledger and load_done use."""
    return f"{repo}@{revision}"


def build_frame(client, max_datasets: int | None = None) -> list[str]:
    """Every dataset repo id the Hub reports, optionally capped.

    This is the frame draw_sample later samples from. Built once per
    census run; the seeded sample it feeds is what makes the eventual
    prevalence table a pre-registered draw rather than a post hoc one.
    """
    repos = []
    for dataset in client.iter_datasets():
        repos.append(dataset["id"])
        if max_datasets is not None and len(repos) >= max_datasets:
            break
    return repos


def draw_sample(frame: list[str], size: int, seed: int) -> list[str]:
    """Deterministic, order independent sample of `size` ids from frame.

    The frame is sorted before sampling, so the draw does not depend on
    the order the Hub happened to return datasets in, only on the
    sorted frame and the seed. Returning fewer than `size` items when
    the frame itself is smaller is correct, not an error.
    """
    ordered = sorted(frame)
    if size >= len(ordered):
        return ordered
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ordered), size=size, replace=False)
    return [ordered[i] for i in idx]


def load_done(path) -> set[str]:
    """Read a JSONL ledger and return the repo@revision keys already
    written, so a re-run can skip them.

    A truncated final line, exactly what a crash mid write leaves
    behind, is skipped rather than raised on: a crash at dataset N
    costs one dataset, not the run.
    """
    path = Path(path)
    done: set[str] = set()
    if not path.exists():
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(_key(row.get("repo", ""), row.get("revision", "")))
    return done


def run_metadata_tier(repos: list[str], source, out_path, done: set[str] | None = None) -> int:
    """Cheap tier: fetch each repo's info.json only, no parquet reads.

    Confirms a repo is reachable and looks like a LeRobot dataset
    before the deep tier spends bandwidth on it. Same resume by key and
    per dataset flush discipline as run_deep_tier: a metadata pass can
    be interrupted and resumed exactly like a deep pass, and any error
    fetching info is recorded rather than raised.
    """
    done = done if done is not None else set()
    n = 0
    for repo in repos:
        rep = DatasetReport(repo=repo, source=type(source).__name__)
        key = _key(rep.repo, rep.revision)
        if key in done:
            continue
        try:
            info = source.info(repo)
            if info is None:
                rep.error = "no info.json"
            else:
                rep.codebase = str(info.get("codebase_version", ""))
                rep.fps = float(info.get("fps", float("nan")))
        except Exception as e:  # noqa: BLE001 - graceful degradation is the point
            rep.error = f"{type(e).__name__}: {e}"
        append_jsonl(rep, out_path)
        n += 1
    return n


def run_deep_tier(repos: list[str], source, cfg: CensusConfig, out_path,
                  done: set[str] | None = None) -> int:
    """Full tier: sample parquet data for each repo and run the audit.

    Every dataset is wrapped in its own try block. A missing repo, an
    absent info.json, an unreadable parquet file, or any other raised
    exception is recorded into the report's error field rather than
    raised, so one bad dataset can never end the run. Resume works the
    same way as run_metadata_tier: a repo already present in `done` is
    skipped, and every record is flushed to out_path immediately after
    it is computed, never batched to the end.
    """
    done = done if done is not None else set()
    n = 0
    for repo in repos:
        key = _key(repo)
        if key in done:
            continue
        try:
            info = source.info(repo)
            if info is None:
                raise ValueError(f"no info.json for {repo}")
            fps = float(info.get("fps", float("nan")))
            paths = source.parquet_paths(repo, info, cfg.files_per_dataset)
            if not paths:
                raise ValueError(f"no parquet files found for {repo}")
            parts = [audit_frame(source.frames(repo, p), fps) for p in paths]
            rep = summarise(repo, info, parts)
            rep.source = type(source).__name__
        except Exception as e:  # noqa: BLE001 - graceful degradation is the point
            rep = DatasetReport(repo=repo, source=type(source).__name__,
                                error=f"{type(e).__name__}: {e}")
        append_jsonl(rep, out_path)
        n += 1
    return n


def prevalence(reports: list[DatasetReport], flag: str) -> tuple[float, float, float]:
    """Fraction of reports carrying `flag`, with its Wilson interval.

    Never returns a bare rate: cairo_protocol.stats.wilson_interval
    supplies the lower and upper bound alongside it, so every
    prevalence claim this module produces carries its own uncertainty.
    """
    n = len(reports)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    successes = sum(1 for r in reports if flag in r.flags.split("|"))
    rate = successes / n
    lo, hi = wilson_interval(successes, n)
    return rate, lo, hi
