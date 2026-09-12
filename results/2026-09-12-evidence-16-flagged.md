# Evidence pack: the 16 confirmable datasets

Every one of the 16 was fetched and its `action` and `observation.state`
columns compared frame by frame. Raw output in
`2026-09-12-evidence-16-flagged.txt`.

**Status: confirmed by the owner on 2026-09-12.** All 16 are recorded
`confirmed = yes` in `2026-09-12-confirmation-worksheet.csv` and in the
`confirmed` column of `2026-09-11-census-400-deep.jsonl`.

The confirmation was a blanket sign-off across all 16 rather than a set of
per-dataset inspection notes, and the record says so rather than implying
otherwise. `CLAUDE.md` places that authority with the owner, who owns every
claim. A reader weighing these findings should know which kind of
confirmation stands behind them, which is why the provenance is recorded
alongside the verdict.

The evidence below is what was gathered to support that judgement. It was
produced by fetching each dataset and comparing its columns, and it stands
on its own regardless of the verdict.

The evidence separates the 16 into three classes that a single verdict
would have flattened.

## Class A: identity is exact, on every frame and every dimension

Five datasets. `action` equals `observation.state` bit for bit, with no
exceptions anywhere in the file read.

| dataset | identity | frames | dims |
|---|---|---|---|
| `qingshu123/Airbot_MMK2_storage_egg_white_box` | 1.000 | 126 | 36 |
| `RoboCOIN/Airbot_MMK2_storage_onion_sweet_potato` | 1.000 | 152 | 36 |
| `ConnorJiang/qg_act_double_rm75_cup_scan_2` | 1.000 | 597 | 36 |
| `Kovavavvavava/pick_toys_human_1_ss_filter` | 1.000 | 271 | 8 |
| `vis22/act_rightbiscuit_20260728_175044` | 1.000 | 29,870 | 7 |

29,870 frames matching across 7 dimensions is not a coincidence, and 36
dimensions matching on every frame is not a rounding artefact. The
measurement is beyond doubt.

What remains genuinely open is its meaning. Some leader-follower rigs
record the follower's measured position as the action, and that is a known
and sometimes deliberate pattern rather than a mistake. The consequence for
training is the same either way: a policy learns to predict the present
rather than command the future. Whether to call that a defect in the
dataset or a property of the rig is the human call.

## Class B: a different phenomenon, on weaker evidence

Six datasets flagged `negative_lag`. Identity is 0.000 for all of them, so
whatever is happening is not the Class A problem.

| dataset | identity | action dims |
|---|---|---|
| `omnaathg/trial_so101_4` | 0.000 | 6 |
| `Hailey-5-2026/grab_block_20ep_20260908_180440` | 0.000 | 6 |
| `Chibaa/rollout_0718_20260724_195935` | 0.000 | 6 |
| `zouhan/record-test` | 0.000 | 6 |
| `schronn/record-test_20260701_162939` | 0.000 | 6 |
| `Kovavavvavava/pick_toys_human_1_ss_filter` | 1.000 | 8 |

Five of the six sit at **exactly 6 action dimensions**. `docs/calibration.md`
established that the lag estimator is unreliable below 6, which makes 6 the
lowest dimensionality the calibration actually validated, not a comfortable
margin above it. These datasets sit on the boundary of the regime where the
estimator was shown to work.

That does not make the flags wrong. It means the lag number alone is not
sufficient evidence, and confirming these on the strength of
`lag_frames = -5` would be confirming exactly what the calibration warned
against. The useful check is whether swapping the two columns produces a
positive, physically sensible lag.

## Class C: mismatched column widths

One dataset, and it is not something the flag list anticipated.

`FedorX8/dobbe_lerobot` has `action` with **7** dimensions and
`observation.state` with **8**, over 1,002,592 frames. The two cannot be
compared elementwise at all, so `identity_frac` is nan rather than a
number, and the audit's identity check is silently inapplicable rather than
passing.

This surfaced as a crash in `tools/inspect_flagged.py`, which lacked the
shape guard `ledger/checks.py` already had. The tool now reports the
mismatch instead. A dataset whose action and state have different widths is
worth understanding on its own terms: it may be a gripper channel present
in one and not the other, or a genuine schema error.

## The three classes should not be weighed equally

A single verdict covers all 16, but the evidence behind them is not
uniform, and a reader should know that rather than infer it.

Class A stands on its own. Identity of exactly 1.000 across up to 29,870
frames and 36 dimensions is not something further inspection would overturn.
The only open question was ever interpretation, not measurement.

Class B is weaker, and deliberately so. Five of its six datasets sit at
exactly 6 action dimensions, the lowest dimensionality `docs/calibration.md`
validated. The lag number alone did not establish it, and the check that would settle
them has since been run: it refuted all six. See
`2026-09-12-column-swap-check.md` and `2026-09-12-lag-gate-correction.md`.

Class C is a different kind of object altogether. `FedorX8/dobbe_lerobot`
was flagged `duplicate_episodes`, which the episode hash evaluates
independently of column widths, so its confirmation is meaningful on its own
terms. Its mismatched action and state widths are a separate observation
that no flag in the current list covers, and it deserves its own
investigation rather than being folded into this count.

If any Class B dataset is later refuted, that is a result rather than an
embarrassment: a flag a human rejected is direct evidence the lag threshold
needs work, and belongs in `docs/calibration.md`.

## To record a verdict

`2026-09-12-confirmation-worksheet.csv`, `confirmed` and `notes` columns.
Suggested values `yes`, `no`, `unclear`. To look again at any single one:

    python tools/inspect_flagged.py <repo-id>

## Confirmed prevalence, corrected

The table this section originally carried is superseded. All six
`negative_lag` flags turned out to be an artifact of `xcorr_lag` returning
the leftmost lag for a dataset with no signal, so four of the sixteen
confirmations no longer carry any confirmable flag. Full account in
`2026-09-12-lag-gate-correction.md`.

| finding | n | rate | 95 percent Wilson |
|---|---|---|---|
| `duplicate_episodes` | 7/353 | 0.0198 | [0.0096, 0.0404] |
| `action_equals_state` | 5/353 | 0.0142 | [0.0061, 0.0327] |
| `negative_lag` | 0/353 | 0.0000 | [0.0000, 0.0108] |
| **any confirmed finding** | **12/353** | **0.0340** | **[0.0196, 0.0585]** |

About one dataset in thirty carries a confirmed recording defect, interval
2.0 to 5.9 percent.

It remains a **lower bound**: `large_lag` at 0.530 after the gate and
`stuck_state` at 0.351 were held out of the confirmation pass.

It also rests on a **blanket confirmation**, so its strength is the strength
of the evidence in this document rather than of 16 independent inspections.
Class A stands on its own. Class B did not survive the check.
