# Limitations

This document is the honest account of what the M-02 LEDGER programme does
not yet support. It is written to be harder on this work than a reviewer
would be, per the review standard in `CLAUDE.md`. Read this before using any
number in `paper/DRAFT.md` for anything beyond describing the sample that
produced it.

All numbers below are read from the files named next to them. None is
estimated. Where a number could not be sourced, this document says so
instead of guessing.

## 1. The deep tier is incomplete

`census_run/deep.jsonl` is the deep tier ledger. It is a live file: a
background census process (`census_run/.census.lock` was present and held a
process id while this document was written) is actively appending to it. Two
counts taken minutes apart during the drafting of this document were 53 and
58 raw lines. The count used in this document is a single snapshot, not a
final number.

At that snapshot: 61 raw JSON lines, 56 unique repository keys (5 repository
names appear twice in the ledger; whether that is a legitimate re-audit
after a revision change on the Hub or a resume gap is not determined here).
All 61 records used `StreamingSource`. 7 records carry a non-empty `error`
field. 34 of the 61 records carry at least one non-empty automated flag.

This is far short of either sample size on record for a deep tier run:
`docs/superpowers/specs/2026-09-05-ledger-census-design.md` states a default
deep tier sample size of 800, and `ledger/census.py`'s `CensusConfig` also
defaults `sample_size` to 800. No completed deep tier run, at any sample
size, exists in this repository as of this draft. No prevalence rate and no
Wilson interval for any deep tier flag is reported anywhere in this
programme's paper. Any such number would be provisional twice over: once
because the sample is partial, and again because a partial, non-random-feeling
prefix of an in-progress run is not the same as a completed seeded draw.

A further gap: the completed tier 1 census (`results/2026-09-09-metadata-census-400.jsonl`,
400 repositories) and the in-progress deep tier ledger share zero
repositories, checked directly (0 of 56 unique deep tier repos appear among
the 400 tier 1 repos). The deep tier is therefore not, at least not yet
verifiably, a deeper look at the same 400 datasets tier 1 already
described. It appears to be a separate seeded draw against a separately
fetched Hub frame, run with a different sample size than tier 1's completed
400. This means the tier 1 descriptive numbers in `DRAFT.md` and the deep
tier flags in this section cannot currently be joined dataset by dataset.

## 2. Zero human confirmations, so zero findings

`ledger/report.py` keeps two things apart on purpose. `DatasetReport.flags`
is written by `summarise()` from automated threshold comparisons.
`DatasetReport.confirmed` is a separate field that stays empty until a human
opens the dataset and fills it in; `PROVISIONAL_HEADER` in that same file
states this in writing on every CSV export: "Flags are provisional
measurements, not findings. No dataset should be treated as having a
confirmed problem until a human has opened it and filled in the confirmed
column." `CLAUDE.md` states the same rule at the programme level: "Never
name a dataset in a LEDGER report as defective until a human has opened it
and confirmed the finding."

Checked directly against both ledgers: `confirmed` is empty on all 400
records in `results/2026-09-09-metadata-census-400.jsonl` and on all 61
records in `census_run/deep.jsonl`, as of the same snapshot described above.

Tier 1 additionally never populates `flags` at all (0 of 400 records carry a
non-empty `flags` value), because the metadata tier records declarations,
not the per-frame measurements the flags are computed from. The 34 flagged
records described in section 1 come entirely from the partial deep tier.

The distinction matters because a flag is cheap and a finding is not. A flag
is one automated threshold comparison against a number computed from
whatever sample of frames the run happened to read. A finding, in this
programme's own terms, requires a person to open the data and look. Zero of
those have happened. There are, at the time of this draft, zero findings in
the M-02 LEDGER programme. There are measurements, and there are provisional
flags on a partial sample, and neither is a finding.

## 3. The lag estimator is unreliable below 6 action dimensions

`docs/calibration.md` measured this directly, and the numbers below are
copied from there, not re-derived.

`large_lag` and `negative_lag`, two of the eight flags `ledger/report.py`
can write, are computed from `lag_frames`, the median of `xcorr_lag`'s
best-lag estimate across an episode's sampled frames (`ledger/checks.py`).
`xcorr_lag` averages a per-joint correlation score across the action
dimension. With few joints to average over, the averaging noise is large
enough that the argmax can pick the wrong lag.

The calibration document's own sweeps: at `n_joints=2`, a 400-trial bucket
exact-matched the injected lag only 387 of 400 times (96.75 percent), with
mismatches up to 6 frames off the true lag, and a dedicated 8000-trial sweep
at `n_joints=2` alone found the worst-case score gap between the true lag
and the winning lag growing to 0.0343 as more trials were drawn. At
`n_joints=6` and above, the same bucketed sweep found 0 mismatches in 400
trials per joint count, and a dedicated 6000-trial sweep at `n_joints=6`
found only 2 mismatches (0.033 percent), both near ties rather than a real
miss (score gaps of 0.00114 and 0.00197, against a 0.02 tolerance). This is
why `tests/test_calibration.py` sweeps lag recovery over `n_joints` 6 to 14
rather than 2 to 14: 6 is the measured boundary, and it is also the DoF of
this kit's own UR5e follower in `lerobot_ur/`.

Consequence for the census: any dataset with fewer than 6 action dimensions
(a single gripper task, a 2 or 3 DoF planar arm, and similar recordings are
common on the Hub per `docs/calibration.md`) should have its `large_lag` and
`negative_lag` flags treated as uninformative, not as evidence, until a
human opens that dataset and inspects the action and state traces directly.
The other six flags (`ts_nonmonotonic`, `bad_dt`, `frame_gaps`,
`stuck_state`, `action_equals_state`, `duplicate_episodes`) do not depend on
`xcorr_lag` and are not affected by this limitation.

This limitation currently cannot be applied per record. `ledger/report.py`
documents that the tier 1 metadata fields on `DatasetReport`, including
`n_features`, are additive fields that a deep tier report leaves at their
defaults, because `run_deep_tier` and `summarise()` do not populate them.
Checked directly: every one of the 61 deep tier records has `n_features: 0`
and `total_episodes: 0`. The deep tier ledger, as currently written, does
not carry the action dimensionality needed to know, per dataset, whether a
given `large_lag` or `negative_lag` flag falls inside or outside the
regime `docs/calibration.md` validated. Of the 32 `large_lag` and
`negative_lag` flags observed across the 61 records in section 1, none can
currently be checked against this limitation from the ledger alone; each
would need the dataset's `info.json` re-read by hand.

## 4. About one percent of datasets declare a layout no reader here can follow

From the completed tier 1 census: 4 of 400 datasets (0.010, Wilson 95
percent interval [0.004, 0.026]) declare a `data_path` template that
`ledger.paths` cannot resolve to a file this codebase can read. These are
excluded from any claim in this programme about parquet-level content,
because no parquet was ever read for them.

Two distinct causes, both from `results/2026-09-09-metadata-census-400.md`:

1. Three datasets, all from the publisher THULab, declare
   `codebase_version: v3.0` with `data_path` values such as
   `data/jaco_play.tsfile`. A `.tsfile` is not the parquet layout that
   declared version implies. This is not a case of a broken parquet file;
   it is a declared version string that does not match the actual storage
   format, so the layout is simply unreadable by a tool that trusts the
   version string.

2. One dataset, `saaduddinM/OXE_berkeley_autolab_ur5_embeddings`, declares
   `codebase_version: v2.1-embeddings-sharded` with `data_path:
   data/shard-{shard_id:05d}-of-{num_shards:05d}.parquet`. This is ordinary
   parquet under a sharded naming scheme that `ledger.paths` did not know at
   the time of the tier 1 run. This case is recoverable in principle; the
   commit history in this repository (`665f0e0`, "Teach ledger.paths the
   sharded layout") shows work on this landed after the tier 1 run that
   produced the 4-of-400 figure, but this draft does not claim the sharded
   case is resolved in the results reported here, since that would require
   re-running the census.

Neither cause is described as a defective dataset. Per `CLAUDE.md`, that
word is reserved for something a human has confirmed, and no human has
opened either of these.

## 5. The census ran anonymously against a rate-limited API, and 400 is a small sample of a large population

`docs/superpowers/specs/2026-09-05-ledger-census-design.md` records the
measurement that forced this programme's design: an anonymous client hits
HTTP 429 within tens of requests against both the Hub's `api/` and
`resolve/` endpoints, and the LeRobot population on the Hub is at least
75,750 datasets, large enough that an exhaustive crawl is not finite on one
machine. The completed tier 1 census
(`results/2026-09-09-metadata-census-400.md`) confirms it ran anonymously,
with no token, relying on `hubclient.py`'s backoff and minimum request
interval to carry the run.

400 of at least 75,750 datasets, about 0.53 percent is a sample fraction of at most about 3.3
percent of the 75,750 datasets measured on 2026-09-11;
this programme has not measured the exact current population size, only
that it is at least 75,750. The Wilson intervals reported against this
sample in `DRAFT.md` describe that sample. They are not a claim about every
LeRobot dataset on the Hub, and they do not become one by being the only
number available. Throttling shaped how long the run took, not which
datasets it saw, because the sample was drawn from the sorted frame before
the run began (`ledger/census.py`, `draw_sample`); that ordering-independence
is a real property of the design, but it does not change the sample
fraction.

## 6. PROTOCOL.md was committed after tier 1 had already run

`CLAUDE.md` requires each programme repository to carry a `PROTOCOL.md`,
pre-registered "before the first trial". For M-02 it was not. Tier 1 ran
and was committed on 2026-09-09 as `001d1fd` with no protocol on record,
and `PROTOCOL.md` followed later as `8b955d8`, while the deep tier was
already part way through.

The substance of pre-registration survived: the sample was fixed before
anything was looked at, seed 20260905 and size 400, drawn by a function
that sorts the frame first so the draw cannot depend on Hub ordering, and
the deep tier audits that same pre-drawn sample rather than a fresh one.

What the lateness costs a reader is concrete and cannot be repaired after
the fact. There is no commit predating the tier 1 run that fixes the seed
and the sample size in writing, so both numbers must be taken on trust
from the results file and from the code as it stands now. A reviewer
entitled to be sceptical about post hoc sample selection has no
cryptographic recourse here, only this disclosure. `PROTOCOL.md` Section
10 records the same deviation from the other direction.

## 7. Tier 1 measures declarations, not data

Every tier 1 number in `DRAFT.md` comes from reading `meta/info.json`. No
parquet file was opened to produce any of them
(`results/2026-09-09-metadata-census-400.md` states this explicitly: "This
reads each dataset's `meta/info.json` and records what it declares. It does
not open any parquet, so nothing here is a temporal integrity finding.").

Concretely: 32 of 390 reachable datasets (0.082, [0.059, 0.114]) declare
`total_episodes: 0`. This has not been shown to mean the dataset is empty.
It could equally mean the field was never set by the publisher, or that the
upload was abandoned before data was added, or that the parquet exists and
the metadata is simply wrong. Distinguishing these requires opening the
actual data, which is what the deep tier and, beyond the automated checks,
a human confirmation are for. None of that has happened for these 32
datasets. This document does not describe any of them as empty, defective,
or abandoned; it describes only what the metadata field says.

## 8. Self-review against the RSS kill list

`CLAUDE.md` requires reviewing a draft as the harshest RSS reviewer against
a named kill list before submission. Applying that list here, honestly,
rather than only at the end of the process:

- **Loose task definition.** "Temporal integrity" is operationalized here
  as exactly eight automated flags, each a threshold comparison calibrated
  against synthetic clean data (`docs/calibration.md`), not yet against any
  confirmed real defect. Whether these eight flags are the right or
  complete operationalization of temporal integrity is not established by
  anything in this repository.
- **Simulation-only.** The thresholds in `ledger/thresholds.toml` were
  calibrated entirely against `ledger/synth.py`'s synthetic generator. No
  threshold has yet been checked against a real, human-confirmed defect
  from the Hub, because zero real confirmations exist (section 2).
- **Weak or untuned baselines.** Not applicable in the sense of a model
  baseline; this programme audits datasets, not policies. Whether a
  temporal-integrity census needs a baseline of comparison (for instance,
  a hand-labeled reference sample) is an open question this draft does not
  answer and the Discussion section marks Pending.
- **Hardware opacity.** Not applicable; no hardware trial is part of this
  programme. Flagged here only so the omission is a stated decision rather
  than a silent one.
- **Missing ablations and failure analysis.** Real gap: 7 of the 61
  snapshotted deep tier records carry a non-empty `error` field, and this
  document does not yet analyze what those errors are or whether they bias
  the flagged subset.
- **No variance reported.** Tier 1's descriptive numbers all carry a Wilson
  95 percent interval. The deep tier, being incomplete, reports no rate and
  therefore no interval; that is treated here as the correct choice, not as
  an exception to this kill-list item.
- **Video-evidence gap.** Not applicable; this programme audits logged
  parquet and metadata, not video.
- **"Another lab could not rebuild it."** This is the item this programme
  currently satisfies best: the tier 1 run's frame, sample size, and seed
  are recorded (`draw_sample(frame, 400, seed=20260905)`,
  `results/2026-09-09-metadata-census-400.md`), and the exact command is
  written down. The deep tier does not yet meet this bar, because, per
  section 1, it is unclear from the ledger alone what frame and sample size
  produced the 61 records read for this draft.

This section will need to be redone once the deep tier finishes and once
any human confirmation exists; it is a snapshot of gaps, not a final
verdict.
