# PROTOCOL.md (pre-registration; commit before the first trial)

Programme: M-02 LEDGER census
Owner: Tarek
Date committed: 2026-09-11
Git hash of the harness at commit: <fill after commit>

This programme audits LeRobot datasets on the Hugging Face Hub, not a
robot. The unit under test is a dataset repository, not a trial run on
hardware. `protocols/PROTOCOL_TEMPLATE.md` was written for a physical
experiment, so several of its sections have no literal referent here.
Each of those is marked below, in one line, with what plays its role
instead, rather than deleted or padded with robot language that does
not apply. Section 10 records, in full, that this file is being
committed after the tier 1 metadata census already ran, which is
itself a deviation from "commit before the first trial."

## 1. Claim under test

For each of the eight named temporal integrity flags (`ts_nonmonotonic`,
`bad_dt`, `frame_gaps`, `stuck_state`, `action_equals_state`,
`negative_lag`, `large_lag`, `duplicate_episodes`), the fraction of
LeRobot dataset repositories on the Hugging Face Hub carrying that flag
is non-zero and is estimated by a Wilson 95 percent interval, computed
over the seeded 400-repository sample (seed 20260905) drawn from the
full Hub frame. This is falsifiable: an independent census over a
different sample of the same size that finds a flag's true population
rate outside the interval reported for it here falsifies that
estimate for that flag.

This is a descriptive prevalence claim, not a comparison between
methods, and it never asserts that any specific dataset is defective.
Per `CLAUDE.md`, naming a dataset defective requires a human to open it
and confirm the finding; see Section 8.

## 2. Primary metric and decision rule

- Primary metric: per dataset, the presence or absence of each of the
  eight named flags, exactly as spelled in `ledger/report.py`:
  `ts_nonmonotonic`, `bad_dt`, `frame_gaps`, `stuck_state`,
  `action_equals_state`, `negative_lag`, `large_lag`,
  `duplicate_episodes`. A dataset's `flags` field is the `|`-joined
  subset that fired; `ledger.census.prevalence(reports, flag)` checks
  `flag in r.flags.split("|")` across a list of reports.
- Statistical procedure: every proportion is a Wilson score interval
  from `cairo_protocol.stats.wilson_interval(successes, n, alpha=0.05)`,
  called by `prevalence()`. `prevalence()` returns `(rate, lo, hi)` and
  never a bare rate; when `n == 0` it returns `nan` for all three rather
  than a false zero.
- Decision rule: the template's keep-or-discard rule governs an
  experimental arm. M-02 has no arms, so there is no keep or discard
  decision here; see Section 5. The applicable rule is completeness:
  a dataset's report is either computed, or its `error` field records
  why it was not, and every record is flushed to the JSONL ledger
  immediately, never batched to the end.
- Stopping rule: tier 1 (metadata) runs over the whole reachable frame
  returned by `client.iter_datasets()`, paginated, optionally capped by
  `CensusConfig.max_datasets`. Tier 2 (deep) runs over the fixed
  400-item sample drawn once by `draw_sample`. Neither tier has an
  adaptive width-based stop like the template's arms; the sample size
  is fixed in advance, not derived from a minimum effect of interest,
  because this census makes no comparison and pre-registers no such
  effect size. If a run is interrupted, `census.load_done()` reads the
  ledger already written and `run_census(..., resume=True)` skips
  those repo@revision keys, so a crash costs at most the one dataset
  being processed at the time. `draw_sample` returns its sample in rng
  draw order, not sorted, so a prefix of an interrupted deep-tier run
  is a random subsample of the pre-drawn 400, not a biased one. Any
  report of results from an incomplete run must state the actual n of
  reports obtained, not the nominal 400.
- Trial budget: N = 400 datasets, fixed by `--sample-size 400 --seed
  20260905` on the actual run recorded in
  `results/2026-09-09-metadata-census-400.md`. The design doc
  (`docs/superpowers/specs/2026-09-05-ledger-census-design.md`, Section
  3) states a default sample size of 800, which it computes pins
  prevalence to plus or minus 2.8 points at 95 percent. The run that
  actually executed used 400, an explicit, recorded override of that
  default, not a moving target: see Section 10.
- Minimum effect of interest: does not apply. There is no baseline arm
  to detect an effect against; see Section 5.

## 3. Units and checks (adapted from Tasks)

The template's per-task table (initial-state distribution, success
criterion, judge, failure taxonomy) has no dataset-census equivalent:
there is one implicit task, audit a dataset for temporal integrity, not
several. The analogue of its failure taxonomy is the check registry in
`ledger/checks.py`, pooled into the eight flags by `ledger/report.py`.
"Who judges" is, at this stage, always the automated check; no flag by
itself names a dataset defective, per Section 8.

| flag | what it measures | trigger | threshold source |
|---|---|---|---|
| `ts_nonmonotonic` | any sampled episode has a timestamp diff at or below zero | episode count > 0 | none, any occurrence flags |
| `bad_dt` | fraction of frame intervals more than 25 percent off the expected period 1/fps, pooled as sum(bad_dt)/sum(n_dt) across sampled files | pooled `frac_bad_dt` > threshold | `thresholds.toml` `frac_bad_dt = 0.05` |
| `frame_gaps` | any sampled episode's `frame_index` skips or repeats a value | episode count > 0 | none, any occurrence flags |
| `stuck_state` | fraction of consecutive frame pairs with a bit-identical `observation.state`, pooled as sum(stuck)/sum(n_stuck) | pooled `stuck_state_frac` > threshold | `thresholds.toml` `stuck_state_frac = 0.20` |
| `action_equals_state` | fraction of frames where `action` equals `observation.state` within `atol=0.0`, pooled as sum(ident)/sum(n_ident) | pooled `identity_frac` > threshold | `thresholds.toml` `identity_frac = 0.50` |
| `negative_lag` | the median of the per-episode best lag (`xcorr_lag`, action against state) over sampled episodes is negative | `lag_frames < 0` | none, sign alone flags |
| `large_lag` | the median best lag is at or above the threshold | `lag_frames >= threshold` | `thresholds.toml` `lag_large = 3` |
| `duplicate_episodes` | fraction of sampled episodes whose first 50 action frames hash identically to another sampled episode's | `dup_episode_frac` > threshold | `thresholds.toml` `dup_episode_frac = 0.0` |

All eight thresholds pool sums of numerators over sums of denominators
across every sampled file for a dataset, never a mean of per-episode or
per-file ratios; `ledger/report.py` states this is deliberate, since a
mean of ratios gives a different answer whenever episodes have unequal
length.

By default the deep tier samples one parquet file per dataset
(`CensusConfig.files_per_dataset = 1`); `--files all` samples every
file `source.parquet_paths` can discover. For a packed-layout dataset
whose source cannot enumerate files (no `exists` check),
`packed_sample_is_partial` marks the report's `note` field
`"partial: packed layout, source has no existence check, sampled 1
file only"`, so an absent flag on such a report means no defect was
found in the one file actually read, not in the whole dataset.

### Sampling frame and draw

The frame is every dataset id the Hub returns for the LeRobot filter,
paginated by `client.iter_datasets()` and collected by
`census.build_frame`. `draw_sample(frame, size, seed)` sorts the frame
before drawing, so the draw depends only on the sorted frame and the
seed, never on the order the Hub happened to return datasets in.
Sorting matters because Hub listing order is not guaranteed stable
across requests or over time; without sorting first, the same seed
could draw a different sample on a different day. Seed 20260905, size
400, as run and recorded in
`results/2026-09-09-metadata-census-400.md`. When the frame is smaller
than the requested size, `draw_sample` returns the whole sorted frame
instead of raising.

## 4. Perturbation strata

Does not apply. There is no lighting, distractor, pose, or instruction
condition to vary for a dataset already sitting on the Hub, so there
is no strata table and no `results/order.csv`. What plays a role near
it is the population heterogeneity tier 1 already recorded: declared
codebase version (v3.0, v2.1, v2.0, v2.1-embeddings-sharded) and
declared layout (packed, per_episode, unknown), reported in
`results/2026-09-09-metadata-census-400.md`. The census does not
stratify the deep-tier sample by either; `draw_sample` draws one
simple random sample from the whole sorted frame. The seed plus the
deterministic sort is what the template's stored `order.csv` would
otherwise be needed for: the same seed against the same frame always
reproduces the same draw, so nothing separate needs to be committed.

## 5. Arms and baselines

Does not apply. There is no policy being compared against a baseline;
M-02 measures a property of an existing population, it does not train
or evaluate a controller. The two-tier design (metadata, deep) is not
an arms comparison either: both tiers run over the same population and
answer different questions at different cost, not competing
hypotheses.

## 6. Hardware disclosure (adapted)

There is no robot, firmware, gripper, or camera. What plays the
analogous role is the data acquisition path and the software
environment that produced the measurements.

- Access path: `ledger/hubclient.py` is the sole chokepoint for Hub
  requests. It reads a token from `HF_TOKEN` or the standard CLI
  cache, never writes or logs it, honours `Retry-After`, applies
  exponential backoff with jitter on 429 and 5xx, and holds a global
  minimum interval between requests.
- The actual tier 1 run was anonymous, no token, per
  `results/2026-09-09-metadata-census-400.md`: "the client's backoff
  and minimum request interval carried the run." The design doc
  measured that an anonymous client hits HTTP 429 within tens of
  requests on both `api/` and `resolve/` endpoints, and called a token
  mandatory for a real run; the actual run proceeded anonymously
  anyway, throttled rather than blocked. See Section 10 for what this
  costs.
- Source implementation: `SampleSource` (`ledger/sources.py`) is the
  abstraction the checks run against. `StreamingSource` reads via HTTP
  range reads projected to the audit columns; `DownloadSource` is the
  fallback on a failed range read, deleting the file after use;
  `LocalSource` reads fixtures from disk; `SyntheticSource` is the
  generator. `census.py` tries streaming and degrades to download on
  failure, recording which path each dataset used.
- Software versions actually used for calibration and testing,
  recorded in `docs/calibration.md`: python 3.11.15, numpy 2.4.6,
  pandas 3.0.5, pyarrow 25.0.1, hypothesis 6.167.1, pytest 9.1.1.

## 7. Compute disclosure (adapted)

No GPU, no training step, so no VRAM assertion and no peak-VRAM log
apply. The resource actually spent is Hub bandwidth and wall-clock
time under the rate limiter above. The one measured efficiency number
available is from the design doc's measurement table: the five audit
columns the checks read are 2.7 percent of a file's compressed bytes,
so streaming transfers 37x fewer bytes and uses 40x less RAM than the
v0 download-whole-file path. No GPU hours, VRAM, or dollar cost figure
for the actual runs is available in the source material, so none is
claimed here.

## 8. What counts as a finding (adapted from "intervention")

`CLAUDE.md` forbids naming a dataset defective until a human has
opened it and confirmed the finding. `ledger/report.py` enforces this
structurally: `DatasetReport.flags` is automated and provisional,
`DatasetReport.confirmed` is a separate field that stays empty until a
human fills it in. Every CSV written by `write_csv` carries
`PROVISIONAL_HEADER` as its first line: "LEDGER M-02 automated audit.
Flags are provisional measurements, not findings. No dataset should be
treated as having a confirmed problem until a human has opened it and
filled in the confirmed column." A flag firing is a measurement. A
finding exists only once `confirmed` is filled in by a human who
opened the data.

`large_lag` and `negative_lag` need an extra step before that human
step is even meaningful: both derive from `lag_frames`
(`ledger/checks.py xcorr_lag`), which `docs/calibration.md` measured
does not reliably resolve below 6 action dimensions. For any dataset
with fewer than 6 action dimensions, treat those two flags as
uninformative rather than as a candidate finding; see Section 10.

## 9. Records and logging (adapted from Video and logging)

There is no video. Its role, a durable, content-addressed record of
what was measured, is played by the append-only JSONL ledger. Every
`DatasetReport` is written by `append_jsonl` to `<out_dir>/metadata.jsonl`
or `<out_dir>/deep.jsonl` immediately after it is computed, never
batched to the end, so an interrupted run loses at most the one record
in flight. Each record's key is `repo@revision`; for `StreamingSource`
and `DownloadSource`, `revision` is the Hub's `X-Repo-Commit` header
read via `HubClient.get_revision`, the closest analogue to a video's
SHA-256: it ties a record to the exact content that was read, so a
dataset that changes on the Hub is re-audited rather than silently
treated as already done. `census.load_done()` rebuilds the set of keys
already written, tolerating a truncated final line exactly as a crash
mid-write leaves it, by skipping rather than raising on it.

A single-writer lock, `census_lock` (a `.census.lock` file created with
`O_EXCL`), guards each output directory for the whole run. It exists
because of a real, measured failure: the first live run of this tool
produced 438 records for 223 datasets when two runs were started
against the same output directory, recorded in `ledger/census.py`'s
own docstring. A crashed run's lock is left in place deliberately;
clearing it is an operator decision (`force_unlock`), never inferred
from a timestamp.

`write_csv` writes `PROVISIONAL_HEADER` as its first line on every CSV
export, described in Section 8.

## 10. Deviations

This file was written after the tier 1 metadata census already ran,
and while the tier 2 deep census is part way through. That is a real
deviation from this template's own subtitle, "commit before the first
trial," and it is recorded here plainly rather than hidden.

- The metadata tier ran on 2026-09-09, recorded in
  `results/2026-09-09-metadata-census-400.md`, before this file
  existed.
- The sample was nonetheless fixed in advance: seed 20260905, size
  400, drawn by `draw_sample`, which sorts the frame first so the draw
  does not depend on Hub ordering (Section 3). The deep tier audits
  that same pre-drawn sample, not a fresh one.
- So the substance of pre-registration, a sample fixed before it was
  looked at, was honoured even though the artifact recording it, this
  file, was not committed first.
- What that costs: a reader of `results/2026-09-09-metadata-census-400.md`
  has no commit predating the run that fixes the seed and size in
  writing. They must take 20260905 and 400 on trust, from the results
  file's own Method section and from the code as it exists now, rather
  than from a pre-registration commit whose hash predates the run.
  This file's own commit hash necessarily postdates 2026-09-09.
- The design doc's stated default sample size is 800
  (`docs/superpowers/specs/2026-09-05-ledger-census-design.md`,
  Section 3); the run actually executed used 400. This is recorded as
  an explicit CLI override (`--sample-size 400`), not an undocumented
  change, but it is a deviation from the design document worth noting
  here since the design doc's plus-or-minus 2.8 point precision figure
  was computed for 800, not for the 400 actually drawn.
- The design doc treats an authenticated token as mandatory for a real
  run (Section 2 of that doc); the tier 1 run that actually executed
  was anonymous. It completed under the rate limiter's backoff rather
  than being blocked, but the design's own stated assumption was not
  what happened in practice.

### Known limitations

- Lag flags below 6 action dimensions. `docs/calibration.md` measured
  that `xcorr_lag` exact-matched the injected lag in only 387 of 400
  trials (96.75 percent) at `n_joints=2`, with mismatches up to 6
  frames, while `n_joints=6` and above had 0 mismatches in 400 trials
  per joint count and 2 in a dedicated 6000-trial sweep (0.033
  percent). `large_lag` and `negative_lag` are uninformative below 6
  action dimensions; many real LeRobot datasets, a single gripper task
  or a 2 or 3 DoF planar arm among them, fall in that range.
- Declared layout that no reader can follow. 4 of 400 datasets, rate
  0.010 [0.004, 0.026], declare a layout `ledger.paths` cannot resolve
  to a parquet path: three `.tsfile` datasets from one publisher and
  one sharded-template dataset the code does not yet parse. Neither
  the metadata tier nor the deep tier can measure anything about them.
- Reachability. 390 of the 400 sampled repositories returned an
  `info.json`; 10 did not, rate 0.025 [0.014, 0.045]. Tier 1 cannot
  tell apart gated, private, deleted, and not-actually-a-LeRobot-dataset
  among those 10, and does not claim to. The deep tier draws from the
  same 400-item sample and inherits the same unreachable subset.
- Anonymous rate limiting. The tier 1 run that produced
  `results/2026-09-09-metadata-census-400.md` ran with no Hub token.
  The design doc measured that an anonymous client hits HTTP 429
  within tens of requests; the client's backoff and minimum request
  interval carried the run, at the cost of pace, not of the sample
  itself, since the draw was fixed before the run began.
- Population size. 400 is a sample of at least 12,000 LeRobot datasets
  on the Hub, per the design doc's own measurement. The intervals in
  this protocol and in the results describe that sample; they are not
  a claim about having audited the population.

Any change to this file after this commit is a new commit with the
reason in the message.
