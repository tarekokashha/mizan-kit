"""
ledger.census  (M-02 LEDGER)
=============================

Two tier Hub census. The metadata tier fetches only each repo's
info.json, cheap enough to run over the whole frame first and confirm a
repo is reachable and looks like a LeRobot dataset before the deep tier
spends any bandwidth on it. The deep tier samples actual parquet data
through a SampleSource and runs the full audit via ledger.report, one
DatasetReport per dataset.

Sampling ranks each repo by a hash of its own id and the seed and takes
the lowest ranks, so membership depends only on the id and the seed:
not on how many other datasets exist, and not on the order the Hub
returned them. That is what makes the resulting prevalence table a pre
registered claim, in the same spirit as protocols/PROTOCOL_TEMPLATE.md,
rather than a post hoc one. An earlier version drew indices into the
sorted frame, which silently failed that promise as the Hub population
grew; see draw_sample for the measurement that exposed it.
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

import contextlib
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

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


class CensusLockError(RuntimeError):
    """Another census run owns this output directory."""


@contextlib.contextmanager
def census_lock(out_dir, force: bool = False):
    """Hold an exclusive lock on a census output directory.

    A census appends to its ledger for hours and is built to be resumable,
    so two runs sharing one out_dir is not an exotic accident. It is the
    obvious operator mistake: start a run, believe it died, start another.
    Without a lock both processes append and the ledger silently doubles.
    That was observed in practice during the first live run of this tool,
    438 records for 223 datasets, and a doubled ledger is worse than a
    crash because it still looks like data.

    The lock is a file created with O_EXCL, so acquiring it is atomic. A
    crashed run leaves its lock behind deliberately. Clearing it is the
    operator's decision, taken with force=True, never something this code
    guesses at by reading a timestamp and deciding a run looks old enough
    to be dead.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / ".census.lock"
    if force:
        with contextlib.suppress(OSError):
            path.unlink()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        held = ""
        with contextlib.suppress(OSError):
            held = path.read_text(encoding="utf-8").strip()
        raise CensusLockError(
            f"a census is already running against {out_dir} "
            f"({held or 'no holder recorded'}). Two runs appending to one "
            f"ledger duplicate every record. If that run is dead, delete "
            f"{path} or pass force to break the lock."
        ) from None
    try:
        os.write(fd, f"pid={os.getpid()} started={time.time():.0f}".encode())
        os.close(fd)
        yield path
    finally:
        with contextlib.suppress(OSError):
            path.unlink()


def write_sample_record(out_dir, frame: list[str], size: int, seed: int) -> dict:
    """Record which datasets a run drew, and what it drew them from.

    A seed alone never identified a sample. It identified a sample given a
    frame, and the frame is a live population that grows daily and was
    thrown away after each run. That is what let the 2026-09-11 sampler
    bug stay invisible: two runs declared the same seed and size, drew
    almost disjoint samples, and nothing on disk contradicted them.

    So a run leaves behind the seed, the size, the frame's size and
    sha256, and the drawn sample in full. The hash is over the sorted
    unique ids, so it is a fingerprint of the population itself rather
    than of the order the Hub happened to return it in: two runs of the
    same population agree, and a population that gained or lost a dataset
    is visibly different. Together these make a prevalence claim auditable
    by someone who was not there, which is the point of pre-registering
    the draw at all.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    canonical = "\n".join(sorted(set(frame))).encode()
    record = {
        "seed": seed,
        "size": size,
        "frame_size": len(set(frame)),
        "frame_sha256": hashlib.sha256(canonical).hexdigest(),
        "drawn_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sample": draw_sample(frame, size, seed),
    }
    (out_dir / "sample.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


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


def _sample_rank(repo: str, seed: int) -> bytes:
    """Where a repo sorts in a draw. A pure function of its id and the seed."""
    return hashlib.blake2b(f"{seed}:{repo}".encode(), digest_size=16).digest()


def draw_sample(frame: list[str], size: int, seed: int) -> list[str]:
    """Stable, order independent sample of `size` ids from frame.

    Each repo is ranked by a hash of its own id and the seed, and the
    lowest `size` ranks are taken. Whether a given dataset is in the
    sample therefore depends only on its id and the seed, never on how
    many other datasets exist or on the order the Hub returned them.

    This replaces an index draw, which was wrong in a way that mattered.
    The previous version sorted the frame and asked a seeded generator
    for `size` indices into it. Indices are positions, so when the Hub
    population grew between two runs every position addressed a
    different repo and the same seed produced an almost entirely
    different sample. Measured on 2026-09-11: a tier 1 run and a deep
    run, both declaring "seed 20260905, size 400", overlapped in 3 of
    400 datasets. A seed that does not identify a sample cannot support
    a pre-registered prevalence claim, which is the whole reason the
    draw is seeded.

    Three properties follow, each covered by a test:

    - Growth safe. Adding datasets to the Hub cannot evict an existing
      member except at the selection boundary, where a newcomer may
      outrank it. Turnover is proportional to how many new ids hash
      below the cutoff, not total.
    - Nested. draw_sample(f, 50, s) is exactly the first 50 of
      draw_sample(f, 100, s), so a run that stops early is a genuine
      prefix of the planned sample rather than a different one.
    - Order free. Only ids and the seed matter, so the Hub's ordering
      cannot influence the draw.

    Hash order is unrelated to any property of a dataset, so a prefix of
    an interrupted run is still a random subsample. Duplicate ids in the
    frame are collapsed: a frame is a set of candidates, not a multiset.
    Returning fewer than `size` when the frame is smaller is correct.
    """
    return sorted(set(frame), key=lambda repo: _sample_rank(repo, seed))[:size]


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
    with open(path, encoding="utf-8") as f:
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


def run_deep_tier(
    repos: list[str], source, cfg: CensusConfig, out_path, done: set[str] | None = None
) -> int:
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
                rep.note = (
                    "partial: packed layout, source has no existence check, sampled 1 file only"
                )
        except Exception as e:  # noqa: BLE001 - graceful degradation is the point
            rep = DatasetReport(
                repo=repo, source=type(source).__name__, error=f"{type(e).__name__}: {e}"
            )
        if get_revision:
            rep.revision = get_revision(repo)
        append_jsonl(rep, out_path)
        n += 1
    return n


TIERS = ("metadata", "deep", "both")


def run_census(
    repos: list[str],
    source,
    cfg: CensusConfig,
    out_dir=None,
    resume: bool = False,
    force_unlock: bool = False,
) -> int:
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
        raise ValueError(f"unknown census tier {cfg.tier!r}, expected one of {TIERS}")
    out_dir = Path(out_dir) if out_dir is not None else Path(cfg.out_dir)
    written = 0
    # Held for the whole run, both tiers. Two runs appending to one ledger
    # duplicate every record and nothing downstream can tell afterwards.
    with census_lock(out_dir, force=force_unlock):
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
