# Column-swap check on the six negative_lag datasets

Date: 2026-09-12. This is the decisive check that
`2026-09-12-evidence-16-flagged.md` says had not yet been run: for each
of the six `negative_lag` datasets, does swapping `action` and
`observation.state` produce a positive, physically sensible lag with a
comparable or better correlation, which is the signature a genuine
column transposition would leave.

## Method

For each dataset: `ledger.sources.StreamingSource` backed by
`ledger.hubclient.HubClient(min_interval=1.0)`, `source.info(repo)`,
`ledger.paths.derive_paths(info, limit=1)` for the first parquet path,
`source.frames(repo, path)` to read it, then `action` and
`observation.state` stacked with `np.stack` and passed to
`ledger.checks.xcorr_lag` both ways: `xcorr_lag(action, state)` as
recorded, `xcorr_lag(state, action)` swapped. `LAGS` is the module
default, `range(-5, 11)`, so a positive lag up to 10 frames or a
negative lag down to -5 frames is what the estimator can report either
way. All six fetches ran back to back with the polite 1 second interval
between Hub requests, no live-network test was skipped, and all 183
previously passing tests plus the 1 pre-existing network-marked skip
were still green before and after this work.

Two of the six datasets are packed (codebase v3.0) datasets whose first
parquet file holds more than one episode: `Hailey-5-2026/...` (20
episodes in the file) and `schronn/...` (2 episodes). For those two, the
single whole-file `xcorr_lag` call concatenates episodes end to end,
which is a different computation from the census's own `lag_frames`
column, which is a per-episode `xcorr_lag` followed by a median across
episodes (`ledger/report.py`, `groupby("episode_index")` then
`np.median(lags)`). Both numbers are reported below: the whole-file
number, which is what fetching "the first parquet" and running
`xcorr_lag` on it directly produces, and a per-episode-then-median
number, computed the same way the census computes it, so the two can be
told apart rather than one silently standing in for the other.

## Primary results: whole first parquet, both directions

| dataset | action dims | state dims | frames read | as recorded: lag, r@lag0, r@best | swapped: lag, r@lag0, r@best |
|---|---|---|---|---|---|
| `omnaathg/trial_so101_4` | 6 | 6 | 1200 | -5, 0.000, 0.000 | -5, 0.000, 0.000 |
| `Hailey-5-2026/grab_block_20ep_20260908_180440` | 6 | 6 | 8894 | +4, 0.987, 0.995 | -4, 0.987, 0.995 |
| `Chibaa/rollout_0718_20260724_195935` | 6 | 6 | 150 | -3, 0.044, 0.049 | +3, 0.044, 0.049 |
| `zouhan/record-test` | 6 | 6 | 299 | -5, 0.000, 0.000 | -5, 0.000, 0.000 |
| `schronn/record-test_20260701_162939` | 6 | 6 | 972 | +3, 0.994, 0.998 | -3, 0.994, 0.998 |
| `Kovavavvavava/pick_toys_human_1_ss_filter` | 8 | 8 | 271 | -5, 0.000, 0.000 | -5, 0.000, 0.000 |

All six report `action_dims == state_dims`, so none of the six trip the
different-widths guard in `ledger/checks.py` (that guard is exercised by
`FedorX8/dobbe_lerobot`, a Class C dataset outside this check's scope).

Every dataset here has exactly 6 action dimensions except
`Kovavavvavava/pick_toys_human_1_ss_filter`, which has 8. None has fewer
than 6. `docs/calibration.md` established that the lag estimator's
recovery of the true lag was validated down to 6 action dimensions and
no lower, with 0 mismatches in 400 trials at exactly 6 joints and 2 in a
dedicated 6000-trial sweep, all near ties rather than large misses. Five
of these six sit at that lowest validated boundary, not comfortably
above it, so a lag number from these five carries the estimator's
weakest tested reliability, and the identity-of-r-values pattern below
matters more than the raw lag digit.

### A structural property of the swap, not extra evidence

In every row above, the correlation at lag 0 and at the best lag is
numerically identical between "as recorded" and "swapped", and the best
lag itself is the negation of the other direction's best lag (`-5` and
`-5` tie aside, `+4`/`-4`, `-3`/`+3`, `+3`/`-3`). This is not a
coincidence and it is not independent confirmation: `xcorr_lag(state,
action)` evaluated at lag `k` computes exactly the same product as
`xcorr_lag(action, state)` evaluated at lag `-k`, by construction of the
shifting in `ledger/checks.py`. So "comparable or better correlation"
after swapping is close to guaranteed whenever the true best lag sits
inside the shared `-5..10` window on both sides, as it does in all six
cases measured here. The correlation strength therefore cannot be the
discriminating fact in this check. The only thing that can discriminate
is whether flipping the sign turns a negative lag into a positive,
small, physically plausible one (consistent with an actuator taking a
few frames to move), and only on data where the correlation is actually
above noise.

## Per-dataset variance context

`omnaathg/trial_so101_4`, `zouhan/record-test`, and
`Kovavavvavava/pick_toys_human_1_ss_filter` show `r@lag0 = r@best =
0.000` in both directions. Checking the per-dimension standard
deviation of the fetched frames explains why:

| dataset | action std (max over dims) | state std (max over dims) |
|---|---|---|
| `omnaathg/trial_so101_4` | 0.0 | 0.0 |
| `zouhan/record-test` | 0.0 | 0.010 |
| `Kovavavvavava/pick_toys_human_1_ss_filter` | 0.0 | 0.0 |

`xcorr_lag` divides by `std + 1e-9`, so a column that never changes
value within the sampled file drives every lag's score to exactly 0.0,
and `max()` over an all-zero dict of scores returns whichever lag was
inserted first (`-5`, the first value in `LAGS`). The `-5` reported for
these three is that tie-break, not a measured anti-correlation. There is
no signal in the fetched frames for the estimator to find a lag in,
positive or negative.

`Kovavavvavava/pick_toys_human_1_ss_filter` was re-fetched and checked
directly: `action` equals `observation.state` exactly on all 271 frames
(`identity_frac = 1.0`), and both are the constant zero vector,
`[0, 0, 0, 0, 0, 0, 0, 0]`, on the first and last frame alike. This is
the same dataset the evidence pack already lists under Class A
(`action_equals_state`, identity 1.000, 271 frames, 8 dims), confirmed
here again against the same file. Swapping two columns that are bit for
bit identical is a no-op: there is no transposition to detect, because
swapping them changes nothing about the data.

`Chibaa/rollout_0718_20260724_195935` is a third, softer version of the
same problem. Its action channel has real variance (std up to 8.6 across
dimensions), but its state channel is nearly frozen, std at most 0.016
across all 6 dimensions, three orders of magnitude smaller than the
action channel. The resulting correlation, 0.044 at lag 0 and 0.049 at
the best lag in both directions, is far below the 0.98 to 0.998 seen in
the datasets with genuine two-sided signal below. A near-flat state
series cannot support a meaningful lag estimate either way.

## Per-episode breakdown for the two packed, multi-episode files

`Hailey-5-2026/grab_block_20ep_20260908_180440`: the fetched file holds
20 episodes. Episodes 4 through 19 (16 of 20) are fully degenerate,
zero variance in both columns, the same `r = 0.000` tie-break as above.
Episodes 0 through 3 (4 of 20) carry real signal:

| episode | frames | as recorded lag | as recorded r@best | swapped lag |
|---|---|---|---|---|
| 0 | 446 | +3 | 0.985 | -3 |
| 1 | 449 | +4 | 0.987 | -4 |
| 2 | 450 | 0 | 0.980 | 0 |
| 3 | 450 | +4 | 0.998 | -4 |

Every episode that has any correlation to measure shows a small
positive lag as recorded (action leads state by 0 to 4 frames, the
direction a real actuator with real latency should produce) at
correlations of 0.98 and above. The per-episode-then-median value,
computed the same way `ledger/report.py` computes `lag_frames`, is
`-5.0`, matching the confirmation worksheet's recorded value for this
dataset. That `-5` median is arithmetic dominance by the 16 degenerate
episodes, each contributing a tie-broken `-5`, not a sign that the
correlated episodes point negative. On the episodes with signal,
swapping flips a physically ordinary positive lag into a negative one.

`schronn/record-test_20260701_162939`: the fetched file holds 2
episodes. Episode 0 (869 frames) has strong signal: as recorded lag
+3, r@best 0.996. Episode 1 (103 frames) is fully degenerate, `r =
0.000`, tie-broken to -5. The per-episode median is `-1.0`, again
matching the confirmation worksheet exactly. With only two episodes,
that median is a straight average of one genuinely positive-lag episode
and one uninformative one, not evidence of two-sided negative
correlation.

## Assessment per dataset

**`omnaathg/trial_so101_4`.** Both `action` and `observation.state` are
exactly constant across all 1200 sampled frames and all 6 dimensions.
There is no correlation for `xcorr_lag` to measure in either direction,
and the reported `-5` is a tie-break artifact, not a finding. This
check neither confirms nor refutes transposition here. The dataset's
`negative_lag` flag rests on frozen columns, a separate phenomenon
(`stuck_state`, already flagged separately in the census) that a
lag-and-swap check cannot adjudicate.

**`Hailey-5-2026/grab_block_20ep_20260908_180440`.** This is the
clearest result of the six. On the 4 of 20 episodes that carry real,
strong correlation (0.98 to 0.998), the as-recorded order already shows
a small positive lag, the physically expected direction for a real
robot. Swapping the columns on those same episodes turns that positive
lag negative, the physically implausible direction. The dataset's
`negative_lag` flag comes from a median dominated by 16 degenerate
episodes, not from genuine anti-causal correlation. Swapping does not
produce a more sensible result here; it produces a less sensible one on
the only episodes worth trusting. This is evidence against transposition
explaining the flag on this dataset.

**`Chibaa/rollout_0718_20260724_195935`.** The state channel is nearly
frozen (std at most 0.016 against an action channel with std up to 8.6),
and the correlation is negligible, 0.044 to 0.049, in both directions.
There is not enough signal in the state column for a lag estimate,
recorded or swapped, to mean anything. Inconclusive, not a confirmation
and not a clean refutation.

**`zouhan/record-test`.** `action` is exactly constant across all 299
sampled frames; `observation.state` is nearly constant too (one
dimension moves by 0.010, the rest do not move at all). No usable
signal in either direction. This check cannot speak to transposition
here; the flag again rests on frozen data.

**`schronn/record-test_20260701_162939`.** The one informative episode
of the two sampled (episode 0, 869 frames, r@best 0.996) shows the same
pattern as Hailey-5-2026 on a smaller scale: a small positive lag as
recorded, the physically ordinary direction, which swapping would turn
negative. The dataset's `negative_lag` flag is a median of that one
informative, positive-lag episode against one fully degenerate episode.
On the strength of the one episode that means anything, this weighs
against transposition explaining the flag, though with much less data
than Hailey-5-2026 (one informative episode instead of four).

**`Kovavavvavava/pick_toys_human_1_ss_filter`.** `action` and
`observation.state` are bit for bit identical on every one of 271
frames, and both are the constant zero vector throughout, reconfirmed
directly in this check. Swapping two identical columns changes nothing;
there is no transposition for a swap to reveal or rule out. This
dataset's `negative_lag` flag is better understood as downstream of the
same zero-variance degeneracy seen in `omnaathg` and `zouhan`, layered
on top of the already-confirmed Class A finding (`action_equals_state`)
for this same dataset, not as an independent negative-lag result this
check can speak to.

## What this check establishes

Two of the six datasets, `Hailey-5-2026/grab_block_20ep_20260908_180440`
clearly and `schronn/record-test_20260701_162939` more weakly, show
active evidence against column transposition as the explanation for
their `negative_lag` flag: on the episodes within each dataset that
carry real correlated signal, the recorded column order already shows
the physically ordinary positive lag, and swapping makes it negative
rather than fixing it. That is a refutation of the transposition
hypothesis for those two datasets specifically, not a claim that
nothing is wrong with them. The census flag itself is a separate,
already-owner-confirmed measurement; this check tested one specific
causal story for that flag, transposition, and found it does not hold
up for the episodes where the estimator has anything to work with.

For three datasets, `omnaathg/trial_so101_4`, `zouhan/record-test`, and
`Kovavavvavava/pick_toys_human_1_ss_filter`, the sampled data is
degenerate, constant or near constant in one or both columns, so there
is no correlation signal for a lag-and-swap check to measure in either
direction. This check is silent on transposition for these three; it
neither confirms nor refutes it, because the input the check needs is
not present in the frames that were fetched.

For one dataset, `Chibaa/rollout_0718_20260724_195935`, the correlation
is too weak in both directions (0.04 to 0.05, against a near-frozen
state channel) to support a conclusion either way.

This check also demonstrates, from the actual numbers rather than from
reading the code alone, that `xcorr_lag`'s "swapped" correlation is
mathematically tied to the "as recorded" correlation rather than an
independent measurement: the two are numerically equal in every row
above, and the best lag under swap is the negation of the best lag as
recorded. A future use of this check should weigh the sign and physical
plausibility of the lag, not the correlation strength, since the
correlation strength does not change under swapping.

## What this check does not establish

This check does not establish that any of the six datasets is free of a
recording problem. The owner's confirmation of all sixteen flagged
datasets, including these six, stands on its own and is not something
this check was designed to overturn; it tested one narrow hypothesis
(column transposition) against one specific piece of evidence
(`xcorr_lag`'s sign and magnitude), not the full set of reasons a
dataset might have been flagged. `stuck_state` and other flags on these
same six datasets are untouched by this analysis.

This check also does not establish reliable lag estimates at all for
five of the six datasets, since five sit at exactly 6 action dimensions,
the lowest dimensionality `docs/calibration.md` validated rather than a
safe margin above it. A degenerate or near-zero correlation on a
6-dimension dataset is not distinguishable, from this check alone, from
a genuine absence of signal versus an estimator operating at the edge of
its validated range. The one dataset with 8 dimensions
(`Kovavavvavava/pick_toys_human_1_ss_filter`) sits safely inside the
validated range, but its degeneracy is total (identical, constant
columns), so dimensionality was never the limiting factor there.

## Verdict

Six datasets checked. Two, `Hailey-5-2026/grab_block_20ep_20260908_180440`
and `schronn/record-test_20260701_162939`, show real evidence against
transposition as the explanation for their `negative_lag` flag: on their
informative episodes, the recorded order is already physically ordinary
and swapping makes it worse. Three, `omnaathg/trial_so101_4`,
`zouhan/record-test`, and `Kovavavvavava/pick_toys_human_1_ss_filter`,
are too degenerate (constant or near-constant columns) for this check to
speak to transposition at all. One, `Chibaa/rollout_0718_20260724_195935`,
is inconclusive on a near-frozen state channel. None of the six shows
the signature a genuine transposition should leave: a negative lag as
recorded turning into a positive, physically plausible, well correlated
lag once swapped.
