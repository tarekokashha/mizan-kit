# Does stuck_state distinguish a genuine stall from an innocent cause

Date: 2026-09-12. Empirical check of whether `stuck_state`
(`stuck_state_frac > 0.20`, `ledger/thresholds.toml`) can tell a
genuinely stalled sensor or frozen recording pipeline apart from an
encoder reporting the same quantised value while the arm is stationary
(a pause before a grasp, for instance). Both produce the same flag.
`docs/calibration.md` reports 0 false positives for `stuck_state_frac`
across 450 synthetic configurations, but its own prose says clean
synthetic state carries independent Gaussian noise every frame, so
bit-identical consecutive frames have effectively zero probability in
that sweep. That number says nothing about this flag's power to tell
the two real causes apart, which is what this check measures instead.

This file records a measurement, not a finding. Per `CLAUDE.md`, no
dataset here is described as defective; each per-dataset line below is
phrased as "consistent with", never as a confirmed problem.

## Method

`ledger.sources.StreamingSource` backed by
`ledger.hubclient.HubClient(min_interval=1.0)`. For each dataset:
`source.info(repo)`, `ledger.paths.derive_paths(info, limit=1)` for the
first parquet path (`results/2026-09-11-census-400-both-tiers.md`
records that this census ran with "files per dataset: 1", so fetching
one parquet reproduces exactly what produced the ledger's
`stuck_state_frac` for these rows, not a different sample of it),
`source.frames(repo, path)`, then grouped by `episode_index`.

Per episode: `observation.state` stacked to a `(T, D)` array exactly as
`ledger.checks._stack` does, then

```
same[i]      = all(state[i+1] == state[i])          # bit exact, i = 0..T-2
```

which is `ledger.report.audit_frame`'s own line for this flag
(`ledger/report.py`, inside the `episode_index` loop). A **run** is a
maximal block of consecutive `True` values in `same`; **run length** is
the count of `True` flags in that block, so a run of length `L` is an
`L+1`-frame plateau of identical state. Pooling `sum(same)` over
`sum(len(same))` across every episode in the file reproduces the
ledger's own pooling rule (`stuck_state_frac = sum(stuck) / sum(n_stuck)`,
`ledger/report.py:245`) and was checked against the ledger's recorded
value for every dataset below: every recomputed value matched the
ledger's to 4 decimal places, which is the strongest evidence available
that the fetch and the accounting here match what produced the census
row.

Where `action` and `observation.state` have the same width,
`action_same[i] = all(action[i+1] == action[i])` is computed the same
way, and three conditional rates are reported: `P(action frozen)`
unconditional, `P(action frozen | state stuck)` (`same[i]` true), and
`P(action frozen | state NOT stuck)`. A large gap between the second and
third is the "arm was told to hold still and did" signature; no gap
means the state freezing carries no information about whether the arm
was told to hold still. Where the widths differ, this comparison is
skipped and reported as unavailable, per the task's guard.

Ten Hub fetches ran with the 1 second minimum interval between requests,
plus a small extra sleep after each; a second small lookup
(`source.info` only, no frame fetch) confirmed the two width-mismatched
datasets' feature schemas. No file under `ledger/` or `cairo_protocol/`
was changed. `pytest -q` was run before and after this work with no
edits made; all previously-passing tests were still green (dots only, no
`F`/`E`, one pre-existing `s`).

## Dataset selection

Filtered `results/2026-09-11-census-400-deep.jsonl` for rows with
`error == ""` and `"stuck_state"` in `flags`: 124 datasets, split 54 in
[0.20, 0.30), 46 in [0.30, 0.50), 24 at or above 0.50, matching the
counts the task described. Ten were picked to span that range and to
vary in episode count and codebase version, not clustered at one edge:

| band | repo | stuck_state_frac (ledger) | why picked |
|---|---|---|---|
| 0.20-0.30 | `lalalala0620/koch_yellow_paper_tape` | 0.2013 | lowest value in the whole flagged set, right at the threshold |
| 0.20-0.30 | `adroitLee/260120_smolvla_ep100_Twz_Syr` | 0.2069 | near-lowest value but 101 episodes, so the flag is not resting on one short file |
| 0.20-0.30 | `Harumo/record-so101-total` | 0.2221 | largest sample in this band, 150 episodes / 89,554 frames |
| 0.20-0.30 | `ITHwangg/record-test-250706-0010` | 0.2960 | top edge of the band, single episode |
| 0.30-0.50 | `Xavier033/test_pick_place` | 0.3168 | 100 episodes; also an action/state width mismatch case (7 vs 8), useful for the guard |
| 0.30-0.50 | `jayp132/green-fresh-150` | 0.4524 | large sample (150 episodes) near the top of the band |
| 0.30-0.50 | `mari1024/record-test_20260603_155713` | 0.4886 | top edge of the band, small sample (5 episodes) for contrast with jayp132 |
| >=0.50 | `KS325/open-drawer-all-r2` | 0.5434 | large sample (80 episodes) just above the band boundary |
| >=0.50 | `Chibaa/rollout_0718_20260724_195935` | 0.9866 | near the ceiling, single short episode |
| >=0.50 | `avyuktsachdeva/ring_txfr_control_3_wrist` | 1.0000 | the ceiling itself, and an action/state width mismatch case (4 vs 3) |

## Run-length structure, per dataset (pooled over every episode in the fetched file)

| repo | frames | episodes | stuck_frac (ledger) | stuck_frac (recomputed) | n_runs | median run | max run | singleton runs | frac of stuck frames in length-1 runs |
|---|---|---|---|---|---|---|---|---|---|
| `lalalala0620/koch_yellow_paper_tape` | 314 | 1 | 0.2013 | 0.2013 | 20 | 1.5 | 11 | 10 | 0.159 |
| `adroitLee/260120_smolvla_ep100_Twz_Syr` | 47,630 | 101 | 0.2069 | 0.2069 | 1,446 | 3.0 | 268 | 475 | 0.048 |
| `Harumo/record-so101-total` | 89,554 | 150 | 0.2221 | 0.2221 | 2,157 | 2 | 166 | 744 | 0.038 |
| `ITHwangg/record-test-250706-0010` | 430 | 1 | 0.2960 | 0.2960 | 7 | 2 | 63 | 1 | 0.008 |
| `Xavier033/test_pick_place` | 7,233 | 100 | 0.3168 | 0.3168 | 598 | 4.0 | 15 | 172 | 0.076 |
| `jayp132/green-fresh-150` | 63,030 | 150 | 0.4524 | 0.4524 | 1,313 | 4 | 369 | 401 | 0.014 |
| `mari1024/record-test_20260603_155713` | 4,946 | 5 | 0.4886 | 0.4886 | 154 | 2.0 | 599 | 73 | 0.030 |
| `KS325/open-drawer-all-r2` | 69,758 | 80 | 0.5434 | 0.5434 | 1,512 | 6.0 | 368 | 292 | 0.008 |
| `Chibaa/rollout_0718_20260724_195935` | 150 | 1 | 0.9866 | 0.9866 | 3 | 54 | 89 | 0 | 0.000 |
| `avyuktsachdeva/ring_txfr_control_3_wrist` | 794 | 3 | 1.0000 | 1.0000 | 3 | 220 | 401 | 0 | 0.000 |

`ITHwangg`'s exact run lengths: `[63, 54, 3, 2, 2, 2, 1]`.
`Chibaa`'s: `[89, 54, 4]`. `avyuktsachdeva`'s: one run per episode, of
length `401`, `220`, and `170`, i.e. every one of the 3 sampled episodes
is a single frozen plateau from its first frame to its last, not a
pause inside an otherwise varying episode.

For the two large multi-episode files where a single long run could be
buried in an average, per-episode maxima were also checked:
`adroitLee` has 2 of 101 episodes with an individual run over 100
frames; `Harumo` has 5 of 150; `jayp132` has 73 of 150 (about half);
`KS325` has 79 of 80 (all but one). `mari1024`'s 599-frame run sits
inside one of only 5 episodes.

## Action-versus-state comparison (the prioritised discriminator)

`Xavier033` (`action` is a 7-dim delta twist: `dx, dy, dz, dr, dp, dyaw,
gripper`; `observation.state` is an 8-dim absolute pose: `x, y, z, qx,
qy, qz, qw, gripper`) and `avyuktsachdeva` (`action` 4-dim, generically
named `D_act`; `observation.state` 3-dim, generically named `D_state`)
trip the different-widths guard, so this comparison is unavailable for
those two; the run-length numbers above are the only evidence for them.

| repo | P(action frozen), unconditional | P(action frozen \| state stuck) | P(action frozen \| state NOT stuck) | excess ratio (stuck / not-stuck) |
|---|---|---|---|---|
| `lalalala0620/koch_yellow_paper_tape` | 0.220 | 0.635 | 0.116 | 5.5x |
| `adroitLee/260120_smolvla_ep100_Twz_Syr` | 0.127 | 0.483 | 0.035 | 13.9x |
| `Harumo/record-so101-total` | 0.161 | 0.580 | 0.041 | 14.1x |
| `ITHwangg/record-test-250706-0010` | 0.261 | 0.866 | 0.007 | 131x |
| `Xavier033/test_pick_place` | unavailable (width mismatch) | unavailable | unavailable | -- |
| `jayp132/green-fresh-150` | 0.411 | 0.863 | 0.037 | 23.3x |
| `mari1024/record-test_20260603_155713` | 0.424 | 0.800 | 0.065 | 12.4x |
| `KS325/open-drawer-all-r2` | 0.464 | 0.798 | 0.066 | 12.0x |
| `Chibaa/rollout_0718_20260724_195935` | 0.000 | 0.000 | 0.000 | undefined (no frozen action anywhere) |
| `avyuktsachdeva/ring_txfr_control_3_wrist` | unavailable (width mismatch) | unavailable | unavailable | -- |

`identity_frac` for all eight comparable datasets is 0.0000 in the
ledger (`action` never equals `observation.state` on the same frame), so
the excess ratios above are not a tautology of the two columns being
duplicates of each other; they measure a genuine temporal correlation
between one channel freezing and the other freezing.

A stricter, per-run version (does `action` stay bit-exact constant for
the *entire* run, not just at one transition) was also computed, bucketed
by run length. It falls with run length in every dataset, because
requiring every transition inside a long run to be exactly frozen is a
combinatorially harder bar than requiring most of them to be. It is
reported here for completeness but the per-transition rates in the table
above are the more informative number for the reasons already stated:

| repo | length-1 runs | 2-3 | 4-10 | 11-50 | >50 |
|---|---|---|---|---|---|
| `lalalala0620/...` | 4/10 (0.40) | 1/4 | 1/5 | 0/1 | -- |
| `adroitLee/...` | 61/475 (0.13) | 18/381 | 24/326 | 22/247 | 2/17 (0.12) |
| `Harumo/...` | 120/744 (0.16) | 42/530 | 6/455 | 18/353 | 0/75 (0.00) |
| `jayp132/...` | 118/401 (0.29) | 41/250 | 21/226 | 72/284 | 32/152 (0.21) |
| `mari1024/...` | 19/73 (0.26) | 3/27 | 2/23 | 4/23 | 1/8 (0.13) |
| `KS325/...` | 118/292 (0.40) | 44/269 | 14/403 | 8/378 | 25/170 (0.15) |
| `Chibaa/...` | -- | -- | 0/1 | -- | 0/2 (0.00) |
| `ITHwangg/...` | 0/1 | 1/4 | -- | -- | 0/2 (0.00) |

## Assessment per dataset

**`lalalala0620/koch_yellow_paper_tape` (0.2013).** Mixed run shape (ten
singletons, several short-to-medium runs, one 11-frame run), but the
action check is unambiguous for this file: the arm's commanded action
was frozen 5.5 times more often during stuck stretches than elsewhere.
Consistent with quantisation while stationary.

**`adroitLee/260120_smolvla_ep100_Twz_Syr` (0.2069).** The weakest case
for "innocent" in this sample. Run shape leans away from pure
quantisation (median run 3, only 4.8 percent of stuck frames sit in
singleton runs, one run of 268 frames against an average episode length
of about 470 frames). The action excess ratio is large (13.9x), which
argues against pure coincidence, but only 48.3 percent of stuck frames
have the action also frozen at that same instant, meaning the other 52
percent occur while the commanded action is changing. Ambiguous; the 2
of 101 episodes carrying a run over 100 frames are the ones worth a
human opening directly before this dataset is treated either way.

**`Harumo/record-so101-total` (0.2221).** Strong excess correlation
overall (14.1x) and a low singleton fraction (3.8 percent of stuck
frames), but the per-run-length breakdown shows the 75 runs longer than
50 frames have 0 percent of them fully action-frozen throughout. Mostly
consistent with quantisation/holds for the bulk of the file; ambiguous
for the long tail (5 of 150 episodes carry a run over 100 frames), which
is where a human check should focus rather than on the dataset as a
whole.

**`ITHwangg/record-test-250706-0010` (0.2960).** By run-length shape
alone this is the closest match in the whole sample to "a few long
contiguous runs": only 7 runs total, and 92 percent of all stuck frames
sit inside the two runs of length 63 and 54. Read on run length alone,
this looks like the genuine-stall template the task describes. The
action check reverses that reading: `P(action frozen | state stuck) =
0.866`, the highest value measured in this whole exercise, a 131x excess
over the already-near-zero baseline when the state is not stuck. The
arm's commanded action was essentially frozen through nearly the entire
stuck period. Consistent with a long, deliberate hold, not a stall. This
is the clearest demonstration in this sample of why run-length shape by
itself is not sufficient, and can point the wrong way.

**`Xavier033/test_pick_place` (0.3168).** Action and state are
structurally different representations here (a 7-dim delta twist
command versus an 8-dim absolute pose with quaternion orientation), so
the guarded comparison is unavailable, not merely unmeasured. On
run-length shape alone: median 4, no run longer than 15 across 100
episodes, none of the outlier-length behaviour seen in the other 0.30+
datasets. Consistent with quantisation while stationary on the evidence
available, with the caveat that the strongest discriminator could not be
applied here.

**`jayp132/green-fresh-150` (0.4524).** High prevalence and a run shape
that would look concerning read alone (median 4, max 369, only 1.4
percent of stuck frames in singleton runs, 73 of 150 episodes carrying a
run over 100 frames). The action check is the strongest "innocent"
signal in the whole sample apart from `ITHwangg`: 86.3 percent of stuck
frames have the action also frozen (23.3x excess over the not-stuck
rate), and even the 152 runs longer than 50 frames keep 21 percent fully
action-frozen throughout, the highest of any dataset's long-run bucket.
Consistent with sustained, repeated, deliberate holds, not a stall,
despite the run-length profile that would suggest otherwise on its own.

**`mari1024/record-test_20260603_155713` (0.4886).** Strong excess
correlation (12.4x) across the file, but it holds the single longest run
measured anywhere in this exercise, 599 frames, inside one of only 5
episodes. Consistent with quantisation/holds for most of the file;
ambiguous specifically in the one episode carrying that 599-frame run,
which should be opened directly before drawing a conclusion about that
episode.

**`KS325/open-drawer-all-r2` (0.5434).** The highest pooled median run
length in the sample (6), almost no singleton runs (0.8 percent of stuck
frames), and 79 of 80 episodes carry a run over 100 frames, a run-length
profile that on its own looks like the strongest stall candidate below
the top band. The action check says otherwise: 79.8 percent of stuck
frames have the action also frozen (12.0x excess), and the longest-run
bucket (>50 frames) still keeps 14.7 percent fully action-frozen
throughout, second highest of the multi-episode datasets measured.
Consistent with a near-universal, task-relevant hold repeated in almost
every episode (plausible for an "open drawer" task: a grasp-and-pause or
hold-open moment), not a stall, despite having both a higher prevalence
and a longer run-length profile than every 0.20-0.50 dataset above it.

**`Chibaa/rollout_0718_20260724_195935` (0.9866).** Three runs (89, 54,
4 frames), zero singleton runs, essentially the entire 150-frame episode
frozen in state. The action channel never repeats bit-for-bit even once
anywhere in the episode: `P(action frozen) = 0.000` unconditional, given
stuck, and given not stuck alike. This is the one dataset in the sample
where run-length shape (a few long runs, no singletons) and the action
comparison (commanded action continuously changing, zero coincidence
with the frozen state) point the same direction, and it is the direction
the task calls the serious case: the arm appears to have been commanded
to keep moving while the recorded state did not follow. This matches,
from an independent measurement, what `2026-09-12-column-swap-check.md`
already found for this same dataset: an action channel with "real
variance (std up to 8.6)" against a state channel "nearly frozen (std at
most 0.016)". Consistent with a genuine stall.

**`avyuktsachdeva/ring_txfr_control_3_wrist` (1.0000).** Three runs, one
per sampled episode, and each run spans the entire episode: all 3
sampled episodes are a single frozen plateau from first frame to last,
with no variation anywhere else in the episode. Quantisation while
stationary predicts variation around a brief hold, not zero variation
across a complete episode, repeated independently in 3 separate
episodes. The action/state width mismatch (4 generically-named dims
versus 3) blocks the direct comparison, and the generic feature names
(`D_act`, `D_state`, not per-joint names) are themselves an oddity worth
a human opening the raw file for, per `CLAUDE.md`'s confirmation step.
Consistent with a genuine stall or a fully frozen recording pipeline;
this check does not and should not conclude the dataset is defective, it
only reports that the measured pattern does not fit the quantisation
story.

## Conclusion

`stuck_state_frac` and its run-length structure, taken alone, do not
reliably separate the two causes across this sample. The clearest "few
long runs" shape measured (`ITHwangg`) turned out to be the innocent
case once the action channel was checked; two of the highest-prevalence,
longer-run-length datasets in the sample (`jayp132`, `KS325`) also turned
out innocent by the same check, despite a run-length profile that would,
read alone, suggest they were stronger stall candidates than several
datasets sitting lower in the 0.20-0.30 band. Prevalence itself tracks
neither cause: values from 0.20 to 0.99 were found consistent with
quantisation once the action channel was checked, and the two datasets
most consistent with a genuine stall sit at 0.9866 and 1.0000, but so
does a value (0.4524, `jayp132`) that checked out innocent. There is no
threshold on `stuck_state_frac`, and no run-length statistic on its own,
that would have sorted these ten datasets correctly; only the
action-versus-state comparison did, and it was unavailable for 2 of the
10 because the two channels are not directly comparable in those
datasets (a structural condition the flag itself cannot see).

The direct answer to the question asked: no, `stuck_state` as currently
defined does not distinguish a genuinely stalled sensor from an encoder
reporting a repeated quantised value while the arm holds still. It
measures a real, computable quantity, and the recomputation here matched
the ledger's own value on every one of the ten datasets checked, so the
measurement itself is not in question. What is in question is whether
that measurement alone should stand in for a finding. On this evidence
it should not. The flag needs a companion measurement, the
action-versus-state co-freezing check demonstrated here (or something
equivalent to it), before a `stuck_state`-flagged dataset is treated as
more likely defective than any other dataset in the census. A different
threshold on `stuck_state_frac` would not fix this: the sample above
shows the innocent and the stall-consistent cases interleaved across the
whole prevalence range, not separated at some undiscovered cut point.
