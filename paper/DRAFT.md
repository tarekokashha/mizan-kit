# Temporal Integrity of Robot Learning Datasets at Population Scale

**DRAFT. This is a skeleton, not a finished paper.** Sections marked
Pending contain no content because the underlying work has not happened
yet. No number in this document is estimated or invented; every number is
sourced from a file in this repository, named where it is used. See
`paper/LIMITATIONS.md` for the full, harder accounting of what this
programme does not yet support.

## Abstract (draft)

This is a draft abstract. What is established: a metadata-only census
(tier 1) over a seeded, reproducible sample of 400 LeRobot Hub datasets,
drawn from a population of at least 75,750, describing what each dataset's
`meta/info.json` declares about its codebase version, storage layout, frame
rate, and size. Every proportion below carries a Wilson 95 percent
confidence interval. What is not established: any finding about temporal
integrity itself. Temporal integrity requires reading the actual recorded
frames, which is the deep tier of this census, and the deep tier is
incomplete at the time of this draft. Separately, and independent of
completeness, this program's own operating rules hold that no automated
flag becomes a finding until a human opens the corresponding dataset and
confirms it; zero such confirmations exist for either tier as of this
draft. There are therefore zero findings in this programme so far. There
are population-level measurements of what publishers declare, and there are
provisional, unconfirmed flags on a partial sample. This draft does not
claim more than that.

## 1. Introduction

A policy trained by imitation learning is trained directly on logged
trajectories: timestamps, actions, and observed states, read frame by
frame from whatever a dataset's parquet files contain. Nothing in a
standard imitation learning pipeline validates that those frames are
temporally coherent before they are used to compute a loss. A dataset
whose timestamps are non-monotonic, whose frames are missing or repeated,
whose observed state is bit-identical across long stretches of playback
(`stuck_state` in `ledger/checks.py`), or whose recorded action channel is
actually a copy of the state channel rather than a genuine control signal
(`identity_frac`), passes that defect straight into whatever policy is
trained on it. A policy cannot tell the difference between a real control
signal and a logging artifact; it will reproduce whichever one it was
shown. This is the motivation for measuring temporal integrity in training
data directly, rather than only downstream in policy behavior.

Measuring this at population scale, across the LeRobot Hub rather than one
named dataset at a time, has not been done inside this codebase before
this programme. `docs/superpowers/specs/2026-09-05-ledger-census-design.md`
records that the tool this programme extends, `ledger/audit.py` v0,
"audits a handful of named LeRobot datasets by downloading whole parquet
files," and that a Hub-wide prevalence table "neither exists," despite
being the README's own stated next step. The same design document measures
why: an anonymous client is rate-limited within tens of requests against
the Hub API, the population is at least 75,750 datasets, and some
individual datasets alone (`kuka_lerobot`, under the older per-episode
layout) span over 200,000 files, so an exhaustive crawl "is not finite on
one machine." A crawl that stops partway has no defined sampling frame and
supports no prevalence claim at all. Those constraints are why this
programme is a seeded, sampled, two-tier census rather than an exhaustive
one, and why, inside this codebase, no population-scale measurement
existed before it.

Whether a population-scale temporal integrity census has been attempted
elsewhere, outside this codebase, is a claim this draft does not make.
That claim would require citations, and the Related Work section below is
intentionally empty for exactly that reason.

## 2. Related Work

**Pending.** `CLAUDE.md` requires that every citation be fetched and
verified before it enters a draft, and states plainly that this project
never cites from memory. No citation has been fetched or verified for
this draft. This section is therefore empty rather than populated with
unverified or remembered references. It will be filled in once sources
have actually been fetched and checked.

## 3. Method

### 3.1 Two-tier census

The census runs in two tiers, per
`docs/superpowers/specs/2026-09-05-ledger-census-design.md`.

**Tier 1, metadata census**, reads one `meta/info.json` per dataset over
the full LeRobot population reachable through the Hub's paginated listing
(Link header cursor, 1000 repositories per page). It records the declared
codebase version, frame rate, episode and frame counts, chunk size,
feature schema, and storage layout family. It never opens a parquet file.

**Tier 2, deep temporal audit**, runs the per-frame checks in
`ledger/checks.py` over parquet data sampled from a seeded draw over the
tier 1 frame. This is the tier that can, once complete and once a human
has confirmed a flag by hand, support an actual temporal integrity claim.

Both tiers write append-only JSONL, one record per dataset, so a run can
resume after a crash or a rate-limit stall without losing prior work
(`ledger/census.py`).

### 3.2 Seeded sample

Sampling sorts the tier 1 frame before drawing, so the draw depends only
on the sorted frame and the seed, not on whatever order the Hub happened
to return repositories in (`ledger/census.py`, `draw_sample`). The
completed tier 1 census used `draw_sample(frame, 400, seed=20260905)`, run
with `python -m ledger.audit --census metadata --sample-size 400 --seed
20260905`, anonymously, with no token
(`results/2026-09-09-metadata-census-400.md`). `ledger/census.py`'s
`CensusConfig` default sample size is 800, not 400; the completed tier 1
run used 400. See `paper/LIMITATIONS.md` for what this means for the
relationship between the two tiers as currently run.

### 3.3 Checks

`ledger/checks.py` and `ledger/report.py` together define eight named
flags, each a threshold comparison against a per-episode or per-dataset
measurement, pooled across sampled episodes as sums of numerators over
sums of denominators rather than a mean of per-episode ratios
(`ledger/report.py`, `summarise`):

- `ts_nonmonotonic`: an episode where any consecutive timestamp difference
  is zero or negative.
- `bad_dt`: the pooled fraction of frame-to-frame time gaps whose
  deviation from the expected `1/fps` interval exceeds 25 percent
  (threshold `frac_bad_dt = 0.05`).
- `frame_gaps`: an episode where the recorded `frame_index` skips or
  repeats a value.
- `stuck_state`: the pooled fraction of consecutive frames whose
  `observation.state` is bit-identical to the previous frame (threshold
  `stuck_state_frac = 0.20`).
- `action_equals_state`: the pooled fraction of frames where the action
  and state vectors are equal within a strict tolerance (threshold
  `identity_frac = 0.50`).
- `large_lag`: the median best-fit lag between the action and state
  channels, from a cross-correlation search (`xcorr_lag`), is at or above
  a threshold of 3 frames (`lag_large`).
- `negative_lag`: that same median best-fit lag is negative.
- `duplicate_episodes`: the pooled fraction of episodes whose action
  values (a rounded hash of the first 50 frames) repeat elsewhere in the
  dataset (threshold `dup_episode_frac = 0.0`).

All five threshold values above are copied from `docs/calibration.md`'s
"Threshold changes" section, which states they are unchanged from the
original values in `ledger/thresholds.toml`. `docs/calibration.md`
separately establishes that `large_lag` and `negative_lag` should be
treated as uninformative for any dataset with fewer than 6 action
dimensions; see `paper/LIMITATIONS.md` section 3.

A flag is an automated, provisional measurement. Per `ledger/report.py`'s
`PROVISIONAL_HEADER` and per `CLAUDE.md`, no dataset is named defective on
the strength of a flag alone; that requires a human to open the dataset
and record a confirmation in the separate `confirmed` field.

### 3.4 Confidence intervals

Every proportion reported in this programme carries a Wilson 95 percent
confidence interval from `cairo_protocol.stats.wilson_interval`. None is
reported bare.

## 4. Results

### 4.1 Tier 1: metadata census, n = 400 (completed 2026-09-09)

Source: `results/2026-09-09-metadata-census-400.md` and
`results/2026-09-09-metadata-census-400.jsonl`. 400 records, 400 unique
repositories, 0 duplicated.

**Reachability.** 390 of 400 returned an `info.json`. 10 did not, rate
0.025 [0.014, 0.045], recorded as `no info.json`, a category that covers
gated, private, deleted, and not-actually-a-LeRobot-dataset repositories
without distinguishing between them.

**Declared codebase version.**

| value | n | rate | 95 percent interval |
|---|---|---|---|
| v3.0 | 280 | 0.718 | [0.671, 0.760] |
| v2.1 | 100 | 0.256 | [0.216, 0.302] |
| v2.0 | 9 | 0.023 | [0.012, 0.043] |
| v2.1-embeddings-sharded | 1 | 0.003 | [0.000, 0.014] |

Four distinct declared strings, not the two families the original design
assumed.

**Declared layout.**

| value | n | rate | 95 percent interval |
|---|---|---|---|
| packed | 277 | 0.710 | [0.663, 0.753] |
| per_episode | 109 | 0.279 | [0.237, 0.326] |
| unknown | 4 | 0.010 | [0.004, 0.026] |

**Declared fps.** 30 fps dominates at rate 0.756 [0.711, 0.796]. The tail
runs 20, 10, 15, 50, 25, 5, 60, 14, and 2 fps. A 2 fps entry exists.

**Size.** Median 10 episodes, p90 130, max 2000. Median 6,175 frames, p90
77,268, max 660,229. The sampled population is dominated by small
datasets.

**About one dataset in twelve declares zero episodes.** 32 of 390
reachable datasets report `total_episodes: 0`, rate 0.082 [0.059, 0.114].
This is a declaration, not a measurement of an empty dataset. Whether the
underlying parquet is also empty, whether the field was simply never set,
or whether these are abandoned uploads is not established by tier 1; see
`paper/LIMITATIONS.md` section 7.

**About one percent of datasets declare a layout no reader here can
follow.** 4 of 400, rate 0.010 [0.004, 0.026]. Two causes: three datasets
from one publisher, THULab, declare `codebase_version: v3.0` with a
`.tsfile` data path that is not parquet at all; one dataset,
`saaduddinM/OXE_berkeley_autolab_ur5_embeddings`, declares
`codebase_version: v2.1-embeddings-sharded` with a sharded parquet naming
template `ledger.paths` did not resolve at the time of this run. See
`paper/LIMITATIONS.md` section 4 for both causes in full and for why
neither is described as a defective dataset.

**The declared version does not determine the layout.** `v3.0` appears
with both the packed template and the unreadable `.tsfile` layout. A tool
that branches on `codebase_version` alone, rather than reading `data_path`
directly, will be wrong for some real datasets on the Hub.

### 4.2 Tier 2: deep temporal audit

The deep tier ran to completion on 2026-09-11 over the same recorded
sample as the metadata tier above. Provenance, integrity checks and the
full tables are in `results/2026-09-11-census-400-both-tiers.md`; the raw
ledgers and the sample record are committed beside it.

353 of 400 datasets were audited. 47 could not be, rate 0.117
[0.090, 0.153], dominated by 31 cases where no parquet exists at the path
the declared layout predicts.

Among the 353 audited, the most common flags were `large_lag` at 0.561
[0.509, 0.612] and `stuck_state` at 0.351 [0.303, 0.402]. The flags that
indicate an outright recording error rather than a timing characteristic
were rare: `duplicate_episodes` 0.020 [0.010, 0.040], `negative_lag` 0.017
[0.008, 0.037], `action_equals_state` 0.014 [0.006, 0.033].

The headline rate is not reportable as a single number. Any flag was
raised for 0.674 [0.624, 0.721] of audited datasets, but 106 of those
carry `large_lag` alone, and `large_lag` is the flag the calibration in
`docs/calibration.md` shows to be least reliable below 6 action
dimensions. Excluding datasets flagged only by `large_lag` gives 0.374
[0.325, 0.426]. The true rate lies somewhere across that span, and this
run does not settle where. Reporting 0.674 without that caveat would
misrepresent the evidence.

One result is validated across both tiers independently. Of the 34
datasets that tier 1 recorded as declaring zero episodes, 29 had no
parquet at the derived path, giving P(no parquet | declares zero
episodes) = 0.853 [0.699, 0.936]. Two datasets fail the other way,
declaring episodes while no parquet is present where the declared layout
says it should be.

**Still pending.** No human has confirmed any flag, so the programme has
measurements and no findings; see `paper/LIMITATIONS.md`.

## 5. Discussion

**Pending.** A discussion requires results to discuss beyond descriptive
statistics of publisher declarations, and beyond that, requires the deep
tier's confirmed findings, of which there are currently none. This section
will be written once the deep tier is complete and at least some flags
have been opened and confirmed by hand.

## 6. Conclusion

**Pending.** For the same reason as the Discussion section above.

## 7. Limitations

See `paper/LIMITATIONS.md` in full. It is the more complete and more
critical document and should be read before any number in this draft is
used for anything. It covers, among other points: the deep tier's
incompleteness and its current disconnect from the tier 1 sample; the zero
human confirmations behind every flag in this programme; the measured
unreliability of the lag estimator below 6 action dimensions, and that the
deep tier ledger does not currently record enough per-dataset information
to apply that limitation record by record; the roughly one percent of
datasets excluded from any parquet-level claim and why; the anonymous,
rate-limited, small-sample nature of this census against a population of
at least 75,750; the absence of a committed `PROTOCOL.md` for this
programme; and the fact that tier 1 measures what publishers declared, not
what their data contains.

## Disclosure

Per `CLAUDE.md`'s disclosure convention, adapted for this draft: the first
drafts of the Introduction, Method, and Results sections of this document,
and of the accompanying `paper/LIMITATIONS.md`, were produced with Claude
(Fable 5.1), drafting directly from the source files named throughout both
documents. Unlike the steady-state form of this disclosure, this is a
first draft and has not yet been verified by the human author; that
verification is a precondition for submission, not something this draft
claims has already happened. All hardware experiments, statistics, and
claims, once reviewed, remain the authors' own.
