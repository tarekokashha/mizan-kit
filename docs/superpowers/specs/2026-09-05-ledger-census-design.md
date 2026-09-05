# M-02 LEDGER census: design

Date: 2026-09-05
Owner: Tarek
Status: approved for planning
Extends: `ledger/audit.py` v0

## 1. Problem

`ledger/audit.py` v0 audits a handful of named LeRobot datasets by
downloading whole parquet files. The README names the next step: a
prevalence table over the Hub, run with `--files all` and the full
dataset list. Neither exists. `--files` is typed `int`, the Hub listing
is capped at one page, there is no resume, the CSV is written only at
the end, and downloaded files are never deleted.

## 2. Measurements taken before this design

All numbers below were measured on 2026-09-05 from this machine. None
are estimated.

| finding | value | consequence |
|---------|-------|-------------|
| Baseline suite | 8 pytest, 17 doctests pass | v0 is sound on pandas 3.0.5, numpy 2.4.6, pyarrow 25.0.1 |
| LeRobot datasets on the Hub | at least 12,000 | exhaustive audit is not finite on one machine |
| Hub list pagination | Link header cursor, 1000 per page | the full frame is reachable cheaply |
| `meta/info.json` | carries a `data_path` format template | every parquet path is derivable, zero tree calls |
| v2.0 layout | one parquet per episode | `kuka_lerobot` alone is 209,880 files |
| v3.0 layout | episodes packed into large files | file count needs `meta/episodes/`, not derivable from episode count |
| Anonymous rate limit | HTTP 429 within tens of requests, on both `api/` and `resolve/` | a token is mandatory for any real run |
| Column projection | audit columns are 2.7 percent of compressed bytes | streaming transfers 37x fewer bytes and uses 40x less RAM |

The rate limit and the per episode file counts are the two findings that
forced a scope change. An exhaustive crawl cannot be completed, and a
crawl that stops partway has no sampling frame, so it supports no
prevalence claim at all.

## 3. Scope

Two tiers.

**Tier 1, metadata census.** One `meta/info.json` per dataset over the
entire LeRobot population. Records codebase version, fps, episode and
frame counts, chunk size, feature schema, and layout family. Cheap, and
it yields a Hub wide descriptive table that does not exist today.

**Tier 2, deep temporal audit.** The existing per frame checks, run over
a seeded random sample drawn from the tier 1 frame. Default sample size
800, which pins any prevalence to plus or minus 2.8 points at 95 percent
by `cairo_protocol.wilson_interval`. Sample size and seed are config.
Per dataset file limits, including all files, are supported for deep
dives.

Prevalence is reported only with a confidence interval computed by
`cairo_protocol`. That is the reason the sample is drawn and recorded
before the run rather than accumulated opportunistically. It is the same
pre registration discipline `protocols/PROTOCOL_TEMPLATE.md` already
requires of a hardware experiment.

## 4. Architecture

`ledger/audit.py` today mixes Hub access, checks, aggregation, CLI and
the synthetic demo in one 470 line module. It is split along seams that
already exist inside it.

    ledger/
      hubclient.py     token aware HTTP: retry, Retry-After, backoff, jitter
      paths.py         derive parquet paths from the info.json data_path template
      sources.py       SampleSource: Streaming, Download, Local, Synthetic
      checks.py        named check registry, one function per defect
      report.py        DatasetReport, flagging, CSV and JSONL writers
      census.py        two tier run loop, sampling frame, resume ledger
      audit.py         thin CLI over the above
      thresholds.toml  calibrated thresholds, config not code

Each module answers one question. `sources.py` answers where frames come
from, `checks.py` answers what is wrong with them, `census.py` answers
over what population, `report.py` answers how it is written down.

### 4.1 SampleSource

The central abstraction. High level code never learns whether frames
came from the Hub, a local fixture, or a generator.

    class SampleSource(Protocol):
        def info(self, repo) -> DatasetInfo | None
        def parquet_paths(self, repo, info, limit) -> list[str]
        def frames(self, repo, path) -> DataFrame

Four implementations. `StreamingSource` opens the file over HTTP range
reads and projects to the five audit columns. `DownloadSource` is the v0
path plus deletion after use, kept as the fallback when a range read
fails. `LocalSource` reads fixtures from disk. `SyntheticSource` wraps
the existing generator.

`census.py` tries streaming and degrades to download on failure,
recording which path each dataset used, so one malformed repo cannot end
a run. This abstraction is also what makes the system testable with no
network, which is the main reason v0 has only two audit tests.

### 4.2 Check registry

`audit_frame` is currently a fixed sequence of `if` blocks and
`summarise` a fixed sequence of flag appends. Adding a check means
editing both, in agreement, by hand. Instead a registry:

    @register("stuck_state", threshold="stuck_state_frac")
    def stuck_state(ep) -> float

Adding a check becomes adding one function. The flag list, the CSV
columns and the report schema are all derived from the registry, so they
cannot drift apart. Every existing check moves across unchanged in
behaviour, pinned by the current tests.

### 4.3 Resume

An append only JSONL ledger, one record per dataset, keyed by repo and
revision sha. On start the runner reads it into a set and skips what it
already holds, so a re run is idempotent and a crash at dataset 3,000
costs one dataset rather than the run. The revision sha is part of the
key, so a dataset that changed on the Hub is audited again rather than
skipped. Records are flushed per dataset, not at the end.

### 4.4 Rate limiting

One chokepoint, `hubclient.py`. The token is read from `HF_TOKEN` or the
standard CLI cache, never written to the repo and never logged. The
client honours `Retry-After`, applies exponential backoff with jitter on
429 and 5xx, and holds a global minimum interval between requests so the
census stays a polite client of a free service.

### 4.5 Provisional findings

`CLAUDE.md` forbids naming a dataset defective until a human has opened
it. The report therefore separates `flags`, which are automated and
provisional, from `confirmed`, which stays empty until a human fills it
in. Every written report carries a header saying so, and the summary
tool prints a list headed flagged for review, never defective.

## 5. Testing

Following the robotics testing pyramid, widest at the bottom.

- **Unit.** Every check against synthetic frames carrying one known
  injected defect. The existing five defect tests are the floor.
- **Property based, hypothesis.** The calibration the README asks for.
  Two properties carry most of the weight. A clean generated dataset
  must never raise a flag, across randomised fps, episode counts, joint
  counts and noise levels. An injected lag of k frames must be recovered
  as lag k. This is how thresholds become justified rather than guessed.
- **Integration.** Each source against local parquet fixtures written in
  both the v2.0 and v3.0 layouts, including the failure cases v0 never
  handled: missing columns, empty file, single frame episode, non
  numeric state, gated repo, absent `info.json`.
- **Golden file.** The census report over a fixed fixture set, so a
  threshold change appears as a reviewable diff rather than a silent
  shift in results.
- **Live smoke.** A small number of real datasets behind a network
  marker, deselected by default so the suite runs offline.

Determinism is a requirement, not an aspiration. Every generator takes a
seed and every test fixes it.

## 6. What this design does not do

- It does not audit video files. The checks are temporal and read the
  parquet columns only.
- It does not attempt an exhaustive crawl. Section 2 explains why.
- It does not touch `cairo_protocol` or `lerobot_ur`. M-03 stays
  untouched, honouring the one active front rule.
- It does not change any existing check's behaviour. The v0 numbers in
  the README must remain reproducible.

## 7. Definition of done

Per `CLAUDE.md`: the code runs from a clean checkout with the documented
command, `pytest` passes, every reported metric carries its confidence
interval or confidence sequence, and nothing outside the named files
changed. In addition the existing 8 tests and 17 doctests still pass,
and `python -m ledger.audit --demo` produces the flags it produces
today.
