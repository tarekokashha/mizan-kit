# Flag audit: which flags name a fault and which name a condition

Date: 2026-09-12. Every flag checked against the first rule added to
`CLAUDE.md` that day: a check must be able to distinguish its innocent cause
from its fault cause, or it reports a condition and not a defect.

Four of the eight were found to fail that test, three of them in one
session. This table is the summary; the reasoning follows.

| flag | innocent cause it must exclude | can it? | reads as |
|---|---|---|---|
| `ts_nonmonotonic` | none identified | not applicable | **fault** |
| `bad_dt` | fps declared wrong, or deliberate variable rate | no | **condition** |
| `frame_gaps` | deliberate subsampling or filtering | no | **condition** |
| `stuck_state` | arm deliberately holding still | no, fixed by companion | **condition** |
| `action_equals_state` | leader-follower recording follower position | no | **condition** |
| `large_lag` | estimator unreliable below 6 action dims | partly, gated | **weak** |
| `negative_lag` | no signal at all | no, now gated to zero | **retracted** |
| `duplicate_episodes` | episodes sharing a home pose | no, fixed by whole-episode hash | **fault, after fix** |

## The four that failed

### `negative_lag`, retracted entirely

`xcorr_lag` selects the best lag with `max(scores, key=scores.get)`. On ties,
`max` returns the first key, and `LAGS` starts at -5. A dataset with zero
correlation was therefore reported as lag -5 and flagged, every time. All six
census instances had `r_best` below 0.5, four of them below 0.1. Prevalence
went from 0.0170 to 0.0000. Fixed by a correlation gate. See
`results/2026-09-12-lag-gate-correction.md`.

### `stuck_state`, a condition

Bit identical consecutive observations occur both when an arm deliberately
holds still, which is innocent and common, and when the state fails to follow
a command, which is a fault. Of 10 flagged datasets sampled, 8 were
consistent with the innocent cause, and neither the threshold nor the run
length structure separated them. Fixed by adding `stuck_while_commanded`,
which compares the action channel. The census predates that check, so its
0.351 stands as a rate of a condition. See
`results/2026-09-12-stuck-state-check.md`.

### `duplicate_episodes`, fixed

`action_head_hash` hashed the first 50 action frames only, so episodes that
begin from a shared home pose hashed identically however differently they
ended. Robot episodes routinely start from a home pose. Measured: four fully
divergent episodes sharing a 60 frame home pose gave `dup_episode_frac`
0.750 and raised the flag. Fixed by hashing the whole action array, with
length folded in so different-length episodes cannot collide on a prefix.

The census predates the fix, so its 7 instances are **unverified under the
corrected check**. They may all survive, since a dataset flagged on a
50 frame prefix may well be duplicated throughout, but that has not been
measured and should not be assumed. Re-running the deep tier would settle it.

### `action_equals_state`, a condition with an unchanged consequence

Some leader-follower rigs record the follower's measured position as the
action. That is a known and sometimes deliberate pattern, so the flag cannot
establish a mistake. It is kept as a reported condition rather than
downgraded, because the consequence for training does not depend on intent: a
policy learns to reproduce the present observation rather than command the
next one. The five census instances are all at identity exactly 1.000 across
126 to 29,870 frames, so the measurement is not in doubt.

## The two not yet examined

`bad_dt` and `frame_gaps` both fail the rule on inspection and neither has
been checked empirically.

`bad_dt` counts frame intervals more than 25 percent from the declared
period. A dataset whose real rate differs from its declared `fps` produces
this for every frame, which is a metadata error rather than dropped frames,
and the flag cannot tell which. A deliberately variable rate recording
produces it too.

`frame_gaps` counts holes in `frame_index`. Several dataset names in the
census sample contain the word `filter`, which suggests deliberate
subsampling, and that produces holes indistinguishable from lost frames.

Neither is part of any confirmed finding, so nothing published depends on
them. Both should be checked the way `stuck_state` was, by sampling flagged
datasets and looking for the discriminating signal, before either is
described as finding a fault.

## What the pattern says

Three of the four failures share one shape: a check measured a proxy that
correlates with the thing of interest, and the proxy had a common benign
cause nobody had enumerated. The fourth, `negative_lag`, was an estimator
returning a default instead of admitting it had no answer.

Neither shape is visible from reading the code, and neither was caught by a
test suite that grew to 190 tests. Both were caught by fetching real data and
asking what else could produce this number. That is the argument for the
first two rules now in `CLAUDE.md`.
