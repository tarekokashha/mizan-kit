# Correction: the lag flags were firing without a correlation

Date: 2026-09-12. This corrects published numbers downward. It was found by
running the column swap check that `2026-09-12-evidence-16-flagged.md` said
had not been run, and it is the reason that check was worth running.

## The defect

`ledger.checks.xcorr_lag` selects the best lag with
`max(scores, key=scores.get)`. When every lag scores identically, which is
what happens when `action` or `observation.state` is constant and the
correlation is 0.000 at every lag, `max` returns the first key. `LAGS` is
`range(-5, 11)`, so the first key is **-5**.

A dataset with no usable signal was therefore reported as `lag_frames = -5`
and flagged `negative_lag`, every time. The estimator was defaulting, not
detecting.

Reproduced directly:

    constant action, constant state -> best_lag=-5, r0=0.000, r_best=0.000

## How far it reached

Across the 353 audited datasets of the 2026-09-11 census:

- **All six** `negative_lag` flags were on datasets where the lag estimate
  is not supportable. Four had `r_best` below 0.1, meaning effectively no
  correlation at any lag. The remaining two were 0.197 and 0.498.
- **Eleven** `large_lag` flags had the same problem.

The column swap check that exposed this found the same thing from the other
direction: of the six `negative_lag` datasets, none showed the signature of
a genuine column transposition. Two showed a physically normal **positive**
lag as recorded, with correlation above 0.98, which actively refutes
transposition. The rest had no signal to interpret.

## The fix

A lag flag now requires a correlation strong enough for the lag estimate to
mean anything. `min_lag_correlation = 0.5` in `ledger/thresholds.toml`,
applied in `ledger/report.py`.

0.5 was chosen against the measurements rather than picked: it excludes the
two weakest observed correlations, 0.197 and 0.498, while leaving the clean
synthetic generator untouched, which correlates above 0.9.

The gate is a deliberate divergence from v0, so `summarise` takes
`min_lag_correlation`, and passing 0.0 reproduces v0 exactly. The
equivalence tests now pass 0.0 explicitly, which is what keeps them
equivalence tests rather than tests that the two happen to agree.

Two tests pin the behaviour: a fully degenerate dataset must not be flagged
`negative_lag`, and a genuinely swapped dataset still must be.

## What changed in the numbers

Measured flags across the 353 audited datasets:

| flag | before | after |
|---|---|---|
| `large_lag` | 198 | 187 |
| `stuck_state` | 124 | 124 |
| `duplicate_episodes` | 7 | 7 |
| `action_equals_state` | 5 | 5 |
| **`negative_lag`** | **6** | **0** |
| any flag | 238 | 229 |

`negative_lag` prevalence was published as 0.0170 [0.0078, 0.0366]. With
the gate it is **0.0000 [0.0000, 0.0108]**. Every instance was an artifact.

## The confirmed findings shrink

The owner confirmed 16 datasets on 2026-09-12. Four of those carried
`negative_lag` as their only confirmable flag, and now carry none:

| dataset | was | now | r_best |
|---|---|---|---|
| `omnaathg/trial_so101_4` | `stuck_state\|negative_lag` | `stuck_state` | 0.000 |
| `Chibaa/rollout_0718_20260724_195935` | `stuck_state\|negative_lag` | `stuck_state` | 0.049 |
| `zouhan/record-test` | `stuck_state\|negative_lag` | `stuck_state` | 0.000 |
| `schronn/record-test_20260701_162939` | `stuck_state\|negative_lag` | `stuck_state` | 0.498 |

Two other confirmed datasets carried `negative_lag` alongside a flag that
survives, so they remain confirmed findings on that other basis:
`Kovavavvavava` on `action_equals_state`, `Hailey-5-2026` on
`duplicate_episodes`.

**Confirmed prevalence: 12/353 = 0.0340 [0.0196, 0.0585]**, corrected down
from 16/353 = 0.0453 [0.0281, 0.0724].

The four datasets above are not cleared of anything. They still carry
`stuck_state`, which was never part of the confirmation pass. What changed
is that the specific claim made about them, that their action and state
columns look transposed, is not supported and was never supported.

## What this says about the work

The confirmation pass confirmed a flag that the fixed code does not raise.
That is exactly the failure a blanket sign-off is vulnerable to, and it is
recorded here rather than quietly repaired.

It also vindicates the decision to run the check at all. The evidence pack
called Class B the weakest of the three and named the column swap check as
the thing that would settle it. It settled it against the flag.

A refuted flag is a result. The lag estimator now has a documented failure
mode, a gate against it, and two regression tests. The census numbers are
smaller and better founded than they were this morning.
