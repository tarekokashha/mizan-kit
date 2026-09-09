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

Both tiers write to a JSONL ledger keyed by repo@revision. The revision
half of that key comes from source.revision(repo) when the source has
one (StreamingSource and DownloadSource proxy HubClient.get_revision,
which captures the Hub's X-Repo-Commit header); a source with no such
method, LocalSource and SyntheticSource among them, leaves it empty,
same as before. Every record is flushed to disk immediately after it
is computed, not batched to the end, so a crash costs at most the one
dataset being processed when it happened. load_done() reads that
ledger back into the set of keys already written, tolerating a
truncated final line, exactly what a crash mid write leaves behind, by
skipping it rather than raising. A caller passes that set back in as
`done` on the next run to resume, and a repo whose revision changed on
the Hub since the ledger was written no longer matches its old key, so
it is re audited rather than silently skipped.

Graceful degradation is mandatory in the deep tier: a missing repo, an
absent info.json, an unreadable parquet file, or any other raised
exception is caught and recorded into the report's error field rather
than raised, so one bad dataset can never end the run. Neither tier
ever labels a dataset as having a confirmed problem; that stays a human
judgement recorded on DatasetReport.confirmed, per ledger.report.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cairo_protocol.stats import wilson_interval
from ledger.paths import layout_family, packed_sample_is_partial
from ledger.report import DatasetReport, append_jsonl, audit_frame, summarise


@dataclass
class CensusConfig:
    out_dir: Path
    sample_size: int = 800
    seed: int = 20260905
    files_per_dataset: int | None = 1  # None means every file ("--files all")
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


def _is_rate_limited_error(error: str) -> bool:
    """True when a record's error field was produced by a propagated
    ledger.hubclient.RateLimited (CRITICAL 2).

    Both tier runners record every exception as f"{type(e).__name__}: {e}"
    (run_metadata_tier, run_deep_tier), so the exception class's own name
    is a stable, distinguishing marker: a genuine 401/403/404 is recorded
    as "no info.json" or "no info.json for <repo>", never this.
    """
    return error.startswith("RateLimited:")


def load_done(path) -> set[str]:
    """Read a JSONL ledger and return the repo@revision keys already
    written, so a re-run can skip them.

    A truncated final line, exactly what a crash mid write leaves
    behind, is skipped rather than raised on: a crash at dataset N
    costs one dataset, not the run.

    CRITICAL 2: a record whose error indicates an exhausted-backoff rate
    limit is excluded from the returned set, even though it was written
    to the ledger. That is not genuinely done work the way a success or
    a real 401/403/404 is; it is the one failure mode this whole client
    exists to survive, so a later --resume run must retry it rather than
    treat the exhausted backoff as permanent.
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
            if _is_rate_limited_error(row.get("error", "")):
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

    Ruling 9: a source that can report a revision (StreamingSource and
    DownloadSource, via HubClient.get_revision) has it read both before
    and after the info() call. The pre-call read only ever finds
    something on a source whose revision was already warmed earlier in
    this process, for instance a deep tier pass sharing the same client
    right after this one; a cold call always reads "" there, since the
    revision is only known once info() itself has returned. Either way,
    the value read after info() is what gets recorded on the report, so
    the key a subsequent load_done() sees is repo@<sha>, not repo@, for
    every source that can supply one. A source with no revision method,
    LocalSource and SyntheticSource among them, behaves exactly as
    before: the key stays repo@.

    CRITICAL 1: a record also carries total_episodes, total_frames,
    chunk_size, layout_family and a compact feature schema
    (feature_names, the sorted feature keys joined by ";", plus
    n_features), the fields the metadata tier has always been documented
    to record (design doc section 3, README). Each is read from info
    with .get(key, default), so a key info.json happens not to carry
    leaves that one field at its DatasetReport default rather than
    raising; the rest of the record is unaffected. layout_family comes
    from ledger.paths.layout_family, which raises ValueError on a
    data_path template it does not recognise; that is caught locally and
    recorded as "unknown" rather than being allowed to fail the whole
    dataset the way letting it propagate to the broad except below would.
    """
    done = done if done is not None else set()
    get_revision = getattr(source, "revision", None)
    n = 0
    for repo in repos:
        pre_revision = get_revision(repo) if get_revision else ""
        if _key(repo, pre_revision) in done:
            continue
        rep = DatasetReport(repo=repo, source=type(source).__name__)
        try:
            info = source.info(repo)
            if info is None:
                rep.error = "no info.json"
            else:
                rep.codebase = str(info.get("codebase_version", ""))
                rep.fps = float(info.get("fps", float("nan")))
                rep.total_episodes = int(info.get("total_episodes", 0) or 0)
                rep.total_frames = int(info.get("total_frames", 0) or 0)
                rep.chunk_size = int(info.get("chunks_size", 0) or 0)
                features = info.get("features")
                if isinstance(features, dict) and features:
                    names = sorted(features.keys())
                    rep.feature_names = ";".join(names)
                    rep.n_features = len(names)
                try:
                    rep.layout_family = layout_family(info)
                except ValueError:
                    rep.layout_family = "unknown"
        except Exception as e:  # noqa: BLE001 - graceful degradation is the point
            rep.error = f"{type(e).__name__}: {e}"
        if get_revision:
            rep.revision = get_revision(repo)
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

    Ruling 9: the resume key is read from source.revision(repo), when
    the source has one, both before and after info() is fetched, for
    the same reason and with the same fallback to "" documented on
    run_metadata_tier. The revision recorded on a successful report is
    always the post-info() read, which is the one a subsequent
    load_done() will see.

    IMPORTANT 3: cfg.files_per_dataset=None ("--files all") on a packed
    dataset only samples every file when source.parquet_paths can
    actually discover them (ledger.paths.derive_paths's `exists`
    parameter; LocalSource, StreamingSource and DownloadSource all
    supply one). When the source cannot answer that,
    ledger.paths.packed_sample_is_partial says so, and the report's
    note records the sample as partial rather than silently claiming
    full coverage it does not have.
    """
    done = done if done is not None else set()
    get_revision = getattr(source, "revision", None)
    n = 0
    for repo in repos:
        pre_revision = get_revision(repo) if get_revision else ""
        key = _key(repo, pre_revision)
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
            has_exists = callable(getattr(source, "exists", None))
            if packed_sample_is_partial(info, cfg.files_per_dataset, has_exists):
                rep.note = "partial: packed layout, source has no existence check, sampled 1 file only"
        except Exception as e:  # noqa: BLE001 - graceful degradation is the point
            rep = DatasetReport(repo=repo, source=type(source).__name__,
                                error=f"{type(e).__name__}: {e}")
        if get_revision:
            rep.revision = get_revision(repo)
        append_jsonl(rep, out_path)
        n += 1
    return n


TIERS = ("metadata", "deep", "both")


def run_census(repos: list[str], source, cfg: CensusConfig, out_dir=None,
               resume: bool = False) -> int:
    """Single entry point for the two tier census: cfg.tier is the only
    switch that decides which tiers run.

    This exists so the tier decision lives in one place. Before this,
    ledger.audit built a CensusConfig with a tier field on it and then
    separately branched on args.census to decide what to run; a caller
    building a CensusConfig directly, tier and all, still got both
    tiers regardless of what tier said, since nothing ever read it.
    run_census is what audit.py now delegates to, so args.census
    reaches run_metadata_tier / run_deep_tier only by way of cfg.tier,
    and any other caller gets the same guarantee.

    Runs run_metadata_tier when cfg.tier is "metadata" or "both", and
    run_deep_tier when it is "deep" or "both", writing to
    <out_dir>/metadata.jsonl and <out_dir>/deep.jsonl, the same paths
    the CLI has always written. Both tiers are the unchanged building
    blocks; this function only decides which of them to call. out_dir
    defaults to cfg.out_dir. resume mirrors the CLI's --resume flag:
    when true, each tier that runs loads its own existing output file
    with load_done() first and skips repos already recorded there.

    Raises ValueError, naming the bad value, if cfg.tier is not one of
    "metadata", "deep" or "both", rather than silently running nothing.
    """
    if cfg.tier not in TIERS:
        raise ValueError(
            f"unknown census tier {cfg.tier!r}, expected one of {TIERS}"
        )
    out_dir = Path(out_dir) if out_dir is not None else Path(cfg.out_dir)
    written = 0
    if cfg.tier in ("metadata", "both"):
        meta_path = out_dir / "metadata.jsonl"
        done = load_done(meta_path) if resume else set()
        n = run_metadata_tier(repos, source, meta_path, done=done)
        written += n
        print(f"metadata tier: {n} record(s) -> {meta_path}", file=sys.stderr)
    if cfg.tier in ("deep", "both"):
        deep_path = out_dir / "deep.jsonl"
        done = load_done(deep_path) if resume else set()
        n = run_deep_tier(repos, source, cfg, deep_path, done=done)
        written += n
        print(f"deep tier: {n} record(s) -> {deep_path}", file=sys.stderr)
    return written


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
