# Confirmation worksheet, 16 datasets

The step that turns measurements into findings. `CLAUDE.md` requires a human
to open a dataset before anything in a LEDGER report becomes a finding, and
the `confirmed` column is where that judgement is recorded. Right now it is
empty for all 400 records, so the programme has zero findings.

These 16 are the ones worth doing first.

## Why these 16

Three flags are worth confirming ahead of the others, for the same reason:
each indicates an outright recording error rather than a timing
characteristic, and **none depends on the lag estimator**, which
`docs/calibration.md` shows is unreliable below 6 action dimensions.

| flag | n | what it claims |
|---|---|---|
| `action_equals_state` | 5 | `action` and `observation.state` are bit identical, so the recorded action carries no command information |
| `negative_lag` | 6 | state predicts action better than action predicts state, consistent with the two columns being swapped |
| `duplicate_episodes` | 7 | distinct episodes whose opening frames hash identically |
| **union** | **16** | some datasets carry more than one |

`large_lag` and `stuck_state` are deliberately excluded from this first
pass. `large_lag` is the unreliable one. `stuck_state` has innocent
explanations, such as servo read quantisation while an arm is stationary,
and needs a different kind of inspection.

## The worksheet

`2026-09-12-confirmation-worksheet.csv`, one row per dataset, with the
measured evidence pre-filled and two empty columns:

- `confirmed`: your verdict. Suggested values `yes`, `no`, `unclear`.
- `notes`: what you saw. This is the part a reviewer will read.

## How to check one

```bash
python tools/inspect_flagged.py <repo-id>
```

It fetches the dataset's first parquet, prints the columns the flag was
raised on, and asserts nothing about whether the dataset is defective. That
judgement is yours.

Worked example, the first `action_equals_state` case:

```
qingshu123/Airbot_MMK2_storage_egg_white_box
  codebase v2.1  fps 30  episodes 43
  read data/chunk-000/episode_000000.parquet: 126 frames
  action dims 36, state dims 36
  frames where action == observation.state EXACTLY: 126 of 126 (1.000)

    [0] action [0.97982, -0.8803, 0.99184, 0.0268, -1.09951, 0.83429]
        state  [0.97982, -0.8803, 0.99184, 0.0268, -1.09951, 0.83429]
        equal: True
```

126 of 126 frames identical across all 36 dimensions. Whether that is a
recording bug, a deliberate choice for a teleoperation setup where the
commanded and measured positions genuinely coincide, or something else, is
exactly the question only a human can settle. The measurement is not in
doubt; its meaning is.

## What to look for, per flag

**`action_equals_state`.** Confirm the identity holds across episodes, not
just the first. Then ask whether it is plausible for this robot: some
leader-follower rigs record the follower's measured position as the action,
which is a known and sometimes intentional pattern. A policy trained on it
learns to predict the present rather than command the future, which is the
harm regardless of intent.

**`negative_lag`.** Check whether swapping `action` and `observation.state`
makes the lag positive and physically sensible. If it does, the columns are
probably transposed. Note the action dimensionality first: below 6, the
estimator is uninformative and the flag should not be confirmed on lag
evidence alone.

**`duplicate_episodes`.** Open two episodes the audit hashed identically and
compare beyond the first 50 frames. Genuinely duplicated episodes inflate a
dataset's apparent size and bias training. Episodes that merely start
identically, for instance from a fixed home pose, are normal and should be
refuted.

## When the column is filled

Feed the CSV back and the prevalence tables can be restated as confirmed
findings with their own intervals, `paper/DRAFT.md` gains a Discussion and
a Conclusion, and the programme has its first results that survive the
distinction `CLAUDE.md` draws.

Refutations matter as much as confirmations. A flag a human looked at and
rejected is evidence the thresholds need work, and belongs in
`docs/calibration.md`.
