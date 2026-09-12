# Census, 400 datasets, both tiers, 2026-09-11

The first census run that is fully auditable. Both tiers ran over one
recorded sample, the frame that produced it is fingerprinted, and the
integrity of the output was checked rather than assumed.

## Provenance

| | |
|---|---|
| Population (frame) | 75,750 unique LeRobot dataset ids |
| Frame sha256 | `31733e715dd49ac767c3640544e81f1a564328e55642ff6ccacbfcb1b44e7eaf` |
| Seed | 20260905 |
| Sample | 400, about 0.53 percent of the population |
| Drawn at | 2026-09-11T12:32:19Z |
| Sample record | `census_v2/sample.json`, lists all 400 ids |
| Auth | anonymous, no token |
| Files per dataset | 1 |

Integrity, checked not assumed: 400 metadata records and 400 deep records,
400 unique in each, zero duplicates, and both tiers cover exactly the 400
ids `sample.json` declares. Neither of the two failure modes that spoiled
earlier runs is present.

Every proportion below carries a Wilson 95 percent interval from
`cairo_protocol.stats.wilson_interval`.


## Correction, 2026-09-12

The flag counts below were computed before the lag correlation gate existed,
so they include lag flags that the current code does not raise. All six
`negative_lag` flags were an artifact of `xcorr_lag` returning the leftmost
lag, -5, for any dataset whose columns are constant, and eleven `large_lag`
flags had the same defect.

Corrected counts over the same 353 audited datasets:

| flag | as published below | corrected |
|---|---|---|
| `large_lag` | 198, rate 0.561 | 187, rate 0.530 |
| `negative_lag` | 6, rate 0.0170 | 0, rate 0.0000 |
| any flag | 238, rate 0.674 | 229, rate 0.6487 |
| `stuck_state`, `duplicate_episodes`, `action_equals_state` | unchanged | unchanged |

Two other statements below have also been overtaken by later work. The
"What this does not establish" section says the `confirmed` column is empty
for all 400 records: 16 were confirmed by the owner on 2026-09-12, and 12
still carry a confirmable flag after this correction. The "Next" section
asks for a human to open the `negative_lag` and `action_equals_state`
cases: that was done, and it is what exposed the defect above.

The numbers below are left as published because this is the dated record of
what that run measured. Rewriting them would falsify it. The full account is
in `2026-09-12-lag-gate-correction.md`, the confirmations in
`2026-09-12-evidence-16-flagged.md`.

## The headline number depends on a flag I do not fully trust

Read this before the table.

| measure | n | rate | 95 percent interval |
|---|---|---|---|
| any flag raised, as measured | 238/353 | 0.674 | [0.624, 0.721] |
| any flag, excluding large_lag-only | 132/353 | 0.374 | [0.325, 0.426] |

106 of the 353 audited datasets carry `large_lag` and nothing else.
`large_lag` is simultaneously the most common flag and the one
`docs/calibration.md` shows to be least reliable: below 6 action dimensions
the lag estimator does not resolve the argmax dependably, and the median
Hub dataset is small. So the true rate of temporal problems in this
population is somewhere in a range roughly twice as wide as either interval
suggests, and which end it sits at is an open question this run does not
settle.

Quoting 0.674 alone would be the single most misleading thing this
document could do.

## Deep tier

353 of 400 were audited. 47 could not be, rate 0.117 [0.090, 0.153].

| error | n |
|---|---|
| FileNotFoundError | 31 |
| ValueError | 10 |
| ArrowInvalid | 6 |

Prevalence among the 353 audited:

| flag | n | rate | 95 percent interval |
|---|---|---|---|
| `large_lag` | 198 | 0.561 | [0.509, 0.612] |
| `stuck_state` | 124 | 0.351 | [0.303, 0.402] |
| `duplicate_episodes` | 7 | 0.020 | [0.010, 0.040] |
| `negative_lag` | 6 | 0.017 | [0.008, 0.037] |
| `action_equals_state` | 5 | 0.014 | [0.006, 0.033] |
| no flag raised | 115 | 0.326 | [0.279, 0.376] |

90 datasets carry both `stuck_state` and `large_lag`.

## Finding: a dataset declaring zero episodes usually has no parquet

This is the strongest result here, because it is the only one validated
across both tiers independently.

Tier 1 recorded 34 reachable datasets declaring `total_episodes: 0`. Tier 2
then tried to read parquet from each.

P(no parquet at the derived path | declares zero episodes) = **0.853**
[0.699, 0.936], 29 of 34.

The earlier tier 1 write-up listed three candidate explanations for a
zero-episode declaration: genuinely empty data, an unset field, or an
abandoned upload. This narrows it. In the large majority of cases the
declaration is accurate about there being nothing to read. It does not yet
distinguish an abandoned upload from a deliberate metadata-only publication,
which needs a human to look.

Two datasets run the other way: they declare episodes but no parquet exists
at the derived path.

- `ArshiaE/conditional-manipulation-task-3_50_percent`, declares 33
- `fecasado/burger-to-plate-filtered`, declares 55

Those two are the more interesting failure, because the metadata promises
data that is not where the declared layout says it should be. Whether the
file is absent or merely elsewhere is not established here.

## What this does not establish

- **No dataset here is defective.** Every number above is a measurement.
  `CLAUDE.md` requires a human to open the data before anything becomes a
  finding, and the `confirmed` column is empty for all 400 records.
- The headline rate is unresolved within roughly a factor of two, for the
  reason given at the top.
- One parquet file per dataset was read. A defect confined to later files
  is invisible here.
- 400 of 75,750 is about 0.53 percent.
- Anonymous, so throttling shaped the pace. It did not shape the sample,
  which was fixed and recorded before the run.
- `FileNotFoundError` means no parquet at the path the declared layout
  predicts. It does not prove the repository is empty.

## Next

1. A human opens a sample of the flagged datasets, starting with the 6
   `negative_lag` and 5 `action_equals_state` cases, which are the two
   flags that indicate an outright recording bug rather than a timing
   characteristic, and are rare enough to check exhaustively.
2. Resolve the `large_lag` question, either by recording action
   dimensionality per dataset and stratifying, or by improving the
   estimator. Until then the headline rate stays a range.
3. Repeat at a larger sample once a token is available. The frame hash
   makes the two runs comparable.
