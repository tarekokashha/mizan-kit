# Evidence pack: the 16 confirmable datasets

Every one of the 16 was fetched and its `action` and `observation.state`
columns compared frame by frame. Raw output in
`2026-09-12-evidence-16-flagged.txt`.

This document presents evidence. It confirms nothing. Per `CLAUDE.md` the
`confirmed` column is a human judgement, and the point of gathering this is
to make that judgement fast and informed rather than to pre-empt it.

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

## Why one verdict for all 16 would have been wrong

Class A has overwhelming evidence. Class B has evidence the project's own
calibration says to distrust in precisely this regime. Class C is a
different kind of thing entirely and was not even measurable by the check
that flagged its neighbours.

A uniform confirmation would have published Class B claims at the same
confidence as Class A ones, and would have asserted an identity result for
Class C that the data cannot support. Refutations in Class B would be a
real result too: a flag a human looked at and rejected is evidence the
threshold needs work, and belongs in `docs/calibration.md`.

## To record a verdict

`2026-09-12-confirmation-worksheet.csv`, `confirmed` and `notes` columns.
Suggested values `yes`, `no`, `unclear`. To look again at any single one:

    python tools/inspect_flagged.py <repo-id>
