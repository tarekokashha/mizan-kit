<div align="center">

# MIZAN kit

**Measure the data before you train on it. Measure the trial before you believe it.**

A dataset audit for robot learning, a statistics library that makes a single
trial analysable, and a hardware harness with simulator based CI.

[![tests](https://github.com/tarekokashha/mizan-kit/actions/workflows/tests.yml/badge.svg)](https://github.com/tarekokashha/mizan-kit/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![tests count](https://img.shields.io/badge/tests-183%20%2B%2017%20doctests-brightgreen)](tests/)

</div>

---

## The result

A policy inherits whatever is wrong with the data it was trained on. Nobody
had measured how much is wrong, at population scale, on the Hub everyone
trains from.

On 2026-09-11 this kit audited a seeded random sample of **400 LeRobot
datasets** drawn from a population of **75,750**, and opened the parquet of
each. Full write up, provenance and raw ledgers:
[`results/2026-09-11-census-400-both-tiers.md`](results/2026-09-11-census-400-both-tiers.md).

**Somewhere between roughly a third and roughly two thirds of LeRobot
datasets show a temporal irregularity.** That span is the honest answer, and
this README will not collapse it into one number:

| measure | n | rate | 95 percent interval |
|---|---|---|---|
| any flag raised, as measured | 238/353 | 0.674 | [0.624, 0.721] |
| any flag, excluding `large_lag` only | 132/353 | 0.374 | [0.325, 0.426] |

106 of the 238 flagged datasets carry `large_lag` and nothing else, and
`large_lag` is the one flag this kit's own calibration shows to be
unreliable below 6 action dimensions, which is where most of the Hub lives.
Quoting 0.674 alone would be the most misleading thing this document could
do.

One result survives that caveat completely, because both tiers found it
independently:

> A dataset that declares zero episodes usually has no parquet at all.
> **P = 0.853, interval [0.699, 0.936]**, 29 of 34.

### Confirmed findings

Per [`CLAUDE.md`](CLAUDE.md), a flag is a measurement until a human confirms
it. On 2026-09-12 the owner confirmed the 16 datasets carrying a flag that
indicates an outright recording error and does not depend on the lag
estimator:

| finding | n | rate | 95 percent Wilson |
|---|---|---|---|
| `duplicate_episodes` | 7/353 | 0.0198 | [0.0096, 0.0404] |
| `action_equals_state` | 5/353 | 0.0142 | [0.0061, 0.0327] |
| `negative_lag` | 0/353 | 0.0000 | [0.0000, 0.0108] |
| **any confirmed finding** | **12/353** | **0.0340** | **[0.0196, 0.0585]** |

`negative_lag` is zero because all six of its flags turned out to be an
artifact. `xcorr_lag` returned the leftmost lag, -5, for any dataset whose
columns are constant, so datasets with no signal were flagged automatically.
A correlation gate now prevents that, and the correction is written up in
[`results/2026-09-12-lag-gate-correction.md`](results/2026-09-12-lag-gate-correction.md).
It took the confirmed count from 16 to 12.

**About one dataset in thirty carries a confirmed recording defect.** That
is a lower bound: `large_lag` at 0.530 and `stuck_state` at 0.351 were held
out of the confirmation pass and remain unconfirmed measurements. The
evidence, and the fact that this was a blanket owner sign-off rather than 16
separate inspections, is recorded in
[`results/2026-09-12-evidence-16-flagged.md`](results/2026-09-12-evidence-16-flagged.md).

The measurement-versus-finding distinction is enforced by a CI gate, not
just by intention.

## What this is

| | what it does | state |
|---|---|---|
| **[`cairo_protocol/`](cairo_protocol/)** | Anytime valid confidence sequences, so you may look after every trial and stop when you like without inflating error. Plus exact tests and power for fixed n work. | Complete |
| **[`ledger/`](ledger/)** | Temporal integrity audit for LeRobot datasets, run as a resumable two tier census over the Hugging Face Hub. | Complete, and run |
| **[`lerobot_ur/`](lerobot_ur/)** | Universal Robots e-Series follower over RTDE with force and velocity clamps, and a simulator based CI gate. | Skeleton |

One idea connects them: **a number without an interval is not a result, and
a dataset nobody audited is not evidence.**

## Quickstart

```bash
pip install -e ".[test]"
python -m pytest                     # 183 tests, 1 live Hub test deselected
python -m ledger.audit --demo        # inject known defects, watch the checks catch them
```

The demo injects exactly one defect per row. The `flags` column names it:

```
repo                 fps  frac_bad_dt  stuck_state_frac  identity_frac  lag_frames  flags
synthetic/clean      30   0.000        0.000             0.000           2.0
synthetic/identity   30   0.000        0.000             1.000           0.0        action_equals_state
synthetic/drops      30   0.055        0.000             0.000           2.0        bad_dt
synthetic/stuck      30   0.000        0.249             0.000           2.0        stuck_state
synthetic/swapped    30   0.000        0.000             0.000          -2.0        negative_lag
synthetic/duplicate  30   0.000        0.000             0.000           6.0        large_lag|duplicate_episodes
```

That output is pinned as a golden file, so any future change to a threshold
shows up as a reviewable diff rather than a silent shift in published
numbers.

## 1. `cairo_protocol`: the statistics

```python
from cairo_protocol.stats import anytime_cs, sequential_compare, wilson_interval

lo, hi = anytime_cs(outcomes)  # valid at EVERY stopping time
sequential_compare(arm_a, arm_b)  # declares a winner only when the sequences separate
wilson_interval(successes, n)  # the fixed n workhorse
```

`anytime_cs` is Robbins' beta binomial mixture. The interval it returns
after any number of trials contains the true rate with probability at least
1 minus alpha, simultaneously over all times. Optional stopping is therefore
free: look after every trial, stop whenever you like, coverage still holds.
Verified by Monte Carlo in the suite, where 3.3 percent of runs ever miss at
a nominal 5 percent.

`sequential_compare` is deliberately conservative, valid by a union bound.
On equal policies it made zero false decisions in 400 runs of 60 interleaved
pairs, and it separates a 0.9 policy from a 0.6 policy in a median of 84
trials per arm.

This matters for robotics specifically. Real trials are expensive, so you
run few of them and you are tempted to peek. These tools make peeking legal.

[`protocols/PROTOCOL_TEMPLATE.md`](protocols/PROTOCOL_TEMPLATE.md) is the
pre-registration a programme commits before its first trial.

## 2. `ledger`: the dataset census

### What it checks

Per episode, then pooled per dataset:

| flag | what it means |
|---|---|
| `ts_nonmonotonic` | timestamps go backwards |
| `bad_dt` | frame intervals outside the declared fps |
| `frame_gaps` | holes in `frame_index` |
| `stuck_state` | consecutive observations bit identical, a stalled sensor |
| `action_equals_state` | `action == observation.state` exactly, the "action is just the state" recording bug |
| `large_lag` / `negative_lag` | the lag at which action best predicts state is large, or inverted |
| `duplicate_episodes` | episodes whose opening frames hash identically |

Aggregation is **pooled**, `sum(numerator) / sum(denominator)` across parts,
never a mean of per episode fractions. Those two differ whenever episodes
have unequal length, which is exactly what dropped frames produce.

### Reproducibility, the part most tools get wrong

Every run writes a `sample.json` recording the seed, the size, the sampling
frame's size and **sha256**, the timestamp, and the full list of datasets
drawn. That is what makes a prevalence claim checkable by someone who was
not there.

It exists because of a bug this kit shipped and then caught. The sampler
originally drew *indices* into a sorted frame, so when the Hub population
grew between two runs, the same seed produced an almost entirely different
sample: two runs declaring `seed 20260905, size 400` overlapped in **3 of
400**. A seed that does not identify a sample cannot support a
pre-registered claim, which is the only reason to seed a draw at all.

Selection is now by `blake2b(seed:repo_id)`, taking the lowest ranks. Three
properties follow, each covered by a test:

- **Growth safe.** New datasets cannot evict existing members except at the
  selection boundary. The same growth that used to turn the sample over
  completely now leaves 380 of 400 in place.
- **Nested.** A draw of 50 is exactly the first 50 of a draw of 100, so an
  interrupted run is a genuine prefix of the planned sample.
- **Order free.** Only ids and the seed matter, never the Hub's ordering.

### Why it samples instead of crawling everything

These were measured, not assumed:

| finding | value | consequence |
|---|---|---|
| LeRobot datasets on the Hub | 75,750 | an exhaustive crawl is not finite on one machine |
| anonymous rate limit | HTTP 429 within tens of requests, on both the api and resolve hosts | a token is effectively mandatory |
| v2.x layout | one parquet per episode; one dataset alone is 209,880 files | "every file" is tens of millions of requests |
| audit columns | 2.7 percent of compressed parquet bytes | streaming transfers roughly 37x fewer bytes than downloading |
| `meta/info.json` | carries a `data_path` template | paths are derived, so the census makes zero tree API calls |

A partial crawl also has no sampling frame, so it supports no prevalence
claim at all. A seeded random sample is cheaper *and* more defensible.

### Running it

```bash
huggingface-cli login    # see below, this matters
python -m ledger.audit --census both --sample-size 400 --seed 20260905 --out-dir census_out --resume
```

Tier 1 reads each dataset's `meta/info.json` and records what it declares.
Tier 2 opens parquet and runs the temporal checks. Both operate on the same
recorded sample. Prevalence is reported through
`cairo_protocol.wilson_interval`, so no rate is ever printed bare.

**A token is effectively required.** Anonymous requests are throttled within
about a minute. The token is read from `HF_TOKEN` or the CLI cache, is never
written to the repository, never logged, and never appears in a repr or an
error message. There are tests asserting both that it reaches the request
and that a tokenless client sends no `Authorization` header at all.

**Resume** appends one JSONL record per dataset, keyed by `repo@revision`,
flushed per dataset rather than at the end. The revision is the real Hub sha
from the `X-Repo-Commit` header of a response the run already makes, so it
costs nothing extra. A crash at dataset 3,000 costs one dataset, and a
dataset that changed on the Hub is re-audited rather than skipped. An output
directory is locked while a run owns it, because two runs appending to one
ledger silently double it, which is worse than a crash because it still
looks like data.

### Flags are provisional, always

Every CSV carries a header saying so. `flags` are measurements. The
`confirmed` column stays empty until a human opens the data. Nothing in this
tooling calls a dataset defective, and **CI fails the build if any code
tries.**

## 3. `lerobot_ur`: the hardware harness

A LeRobot style follower for Universal Robots e-Series over RTDE, with the
built in force torque sensor in the observation, velocity and force clamps
on every command, and a monotonic timestamp so the audit can check sync.
`ursim_smoke.py` measures RTDE stream rate and jitter against the official
simulator: hardware in the loop CI with no arm in the room.

This is a skeleton. The LeRobot base class wiring is the next task, and its
CI gate has never been green. **No real robot number has been produced by
this repository, and none should be until that wiring lands.**

## Install

```bash
pip install -e .              # library
pip install -e ".[test]"      # plus pytest and hypothesis
pip install -e ".[dev]"       # plus ruff
pip install -e ".[hardware]"  # plus ur_rtde, for M-03
```

Python 3.11 or newer.

## Development

```bash
python -m pytest                            # offline suite, no network
python -m pytest -m network                 # live Hub tests, needs a token
python -m doctest cairo_protocol/stats.py   # silence means all 17 pass
ruff check . && ruff format --check .
```

CI runs the suite on Linux and Windows across Python 3.11 and 3.13, plus
ruff and three repository convention gates: no em dashes, no code that
labels a dataset defective, and no committed token.

### On Windows

Windows has no `make`, so the `Makefile` targets do not run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ledger.audit --demo
```

The M-03 URSim jitter gate is a Linux target. Docker Desktop on Windows adds
a WSL2 network hop and Windows has a coarse default timer, so a p99.9 jitter
figure measured there describes the host, not the robot stack.

## Repository layout

```
cairo_protocol/   confidence sequences, exact tests, power
ledger/           the census
  checks.py       temporal checks in an open registry
  report.py       pooled aggregation, flagging, CSV and JSONL writers
  census.py       two tier run loop, stable sampling, frame recording, resume, locking
  sources.py      SampleSource: streaming, download, local, synthetic
  hubclient.py    token aware HTTP with backoff and pagination
  paths.py        parquet paths derived from info.json, three layout families
  synth.py        seeded generator, one injector per defect
  audit.py        the CLI
lerobot_ur/       UR5e follower and its URSim smoke test
protocols/        the pre-registration template
PROTOCOL.md       the M-02 census pre-registration
results/          append only census logs and write ups
paper/            DRAFT.md and LIMITATIONS.md
docs/             calibration measurements, design spec, decision record
tools/            summarise_census.py
tests/            183 tests, plus a frozen v0 reference
```

## Design decisions worth knowing

**The refactor provably changed nothing.** `tests/v0_reference.py` is a
frozen snapshot of the audit as v0 shipped it, and equivalence tests assert
the current pipeline reproduces it exactly, for single and multi part
inputs. The guarantee was verified as non tautological: a deliberate change
to the pooled denominator makes both equivalence tests fail.

**Thresholds are calibrated, not guessed.**
[`docs/calibration.md`](docs/calibration.md) records the measured false
positive sweep behind every value in `ledger/thresholds.toml`, including the
property based sweep that found the low degree of freedom lag limitation.

**Adding a check is adding a function.** Checks live in a registry, so the
flag list, the CSV columns and the report schema cannot drift apart.

**Three layout families, because the Hub has three.** `per_episode`,
`packed`, and `sharded`. The declared `codebase_version` does not determine
the layout: `v3.0` appears with both the packed template and with a
`.tsfile` that is not parquet at all. Any tool branching on the version
string rather than reading `data_path` is wrong for real datasets.

## Limitations

Stated plainly, because a research kit that hides these is worse than none.

- **The headline rate is unresolved within a factor of two**, for the
  `large_lag` reason at the top of this README.
- **Zero human confirmations, so zero findings.** The programme has
  measurements only.
- **Lag on low degree of freedom arms.** Below 6 action dimensions the
  correlation differs so little between adjacent lags that the argmax is not
  reliably the true lag: measured exact match was 96.5 to 97 percent at 2
  joints, against 0 to 2 mismatches per 6000 to 8000 trials at 6 joints and
  above. Treat `large_lag` and `negative_lag` as uninformative there.
- **One parquet file per dataset** was read in the published run. A defect
  confined to later files is invisible, so these rates are lower bounds with
  respect to per-file coverage.
- **400 of 75,750** is about 0.53 percent, run anonymously.
- **About 1 percent of datasets declare a layout no reader can follow**, and
  are excluded from any parquet level claim.
- **`lerobot_ur` is a skeleton** that has produced no robot number.

Full detail in [`paper/LIMITATIONS.md`](paper/LIMITATIONS.md).

## Citing

See [`CITATION.cff`](CITATION.cff), or GitHub's "Cite this repository"
button.

## Disclosure

Experiment code, simulation harnesses, data audit tooling and first drafts
were produced with Claude and verified by the authors; all hardware
experiments, statistics and claims are the authors' own.

## License

Apache-2.0. See [LICENSE](LICENSE).
