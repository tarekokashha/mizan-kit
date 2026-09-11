<div align="center">

# MIZAN kit

**Statistics, dataset auditing and hardware harnesses for robot learning.**

Analyse the first trial you ever run correctly. Find out what is wrong with a
robot dataset before you train on it. Run hardware in the loop CI with no arm
in the room.

[![tests](https://github.com/tarekokashha/mizan-kit/actions/workflows/tests.yml/badge.svg)](https://github.com/tarekokashha/mizan-kit/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![tests count](https://img.shields.io/badge/tests-148%20%2B%2017%20doctests-brightgreen)](tests/)

</div>

---

## What this is

Three components of the MIZAN research programme. All three run today, and
the suite is green offline with no hardware and no network.

| | what it does | status |
|---|---|---|
| **`cairo_protocol/`** | The statistics under every experiment. Anytime valid confidence sequences, so you may look after every trial and stop when you like without inflating error. | Complete, 17 doctests |
| **`ledger/`** | Temporal integrity audit for LeRobot datasets, run as a two tier census over the Hugging Face Hub with prevalence reported as confidence intervals. | Complete, v1 |
| **`lerobot_ur/`** | Universal Robots e-Series follower over RTDE, with force and velocity clamps and a simulator based CI gate. | Skeleton |

The unifying idea is that a number without an interval is not a result, and a
dataset nobody audited is not evidence.

## Quickstart

```bash
pip install -e ".[test]"
python -m pytest                     # 148 tests, 1 live Hub test deselected
python -m ledger.audit --demo        # injects known defects, proves the checks catch them
```

The demo injects one defect per row. The `flags` column should name it:

```
repo                 fps  frac_bad_dt  stuck_state_frac  identity_frac  lag_frames  flags
synthetic/clean      30   0.000        0.000             0.000           2.0
synthetic/identity   30   0.000        0.000             1.000           0.0        action_equals_state
synthetic/drops      30   0.055        0.000             0.000           2.0        bad_dt
synthetic/stuck      30   0.000        0.249             0.000           2.0        stuck_state
synthetic/swapped    30   0.000        0.000             0.000          -2.0        negative_lag
synthetic/duplicate  30   0.000        0.000             0.000           6.0        large_lag|duplicate_episodes
```

## 1. `cairo_protocol`, the statistics

```python
from cairo_protocol.stats import anytime_cs, sequential_compare, wilson_interval

lo, hi = anytime_cs(outcomes)  # valid at EVERY stopping time
sequential_compare(arm_a, arm_b)  # declares a winner only when the sequences separate
wilson_interval(successes, n)  # the fixed n workhorse
```

`anytime_cs` is Robbins' beta binomial mixture. The interval it returns after
any number of trials contains the true rate with probability at least
1 - alpha, simultaneously over all times, so optional stopping is free. Checked
by Monte Carlo in the suite: 3.3 percent of runs ever miss at a nominal 5
percent.

`sequential_compare` is deliberately conservative, valid by a union bound. On
equal policies it made zero false decisions in 400 runs of 60 interleaved
pairs, and it separates a 0.9 policy from a 0.6 policy in a median of 84
trials per arm.

`protocols/PROTOCOL_TEMPLATE.md` is the pre-registration a programme commits
before its first trial.

## 2. `ledger`, the dataset census

Policies inherit whatever is wrong with the data they were trained on. This
audits the data first.

### What it checks

Per episode, then pooled per dataset: non monotonic timestamps, frame
intervals outside the declared fps, frame index holes, bit identical
consecutive states (a stalled sensor), `action == observation.state` exactly
(the "action is just the state" recording bug), the lag at which action best
predicts state, and duplicate episodes.

### Running it

```bash
huggingface-cli login                 # required, see below
python -m ledger.audit --census metadata --out-dir census_out
python -m ledger.audit --census deep --sample-size 800 --out-dir census_out --resume
```

Tier 1 records metadata for every LeRobot dataset on the Hub. Tier 2 runs the
temporal checks over a seeded random sample and reports prevalence through
`cairo_protocol.wilson_interval`. A sample of 800 datasets pins any prevalence
to about plus or minus 2.8 points at 95 percent.

### Why it samples instead of crawling everything

These were measured on 2026-09-05, not estimated:

| finding | value | consequence |
|---|---|---|
| LeRobot datasets on the Hub | at least 75,750 | an exhaustive crawl is not finite on one machine |
| anonymous rate limit | HTTP 429 within tens of requests, on both the api and resolve hosts | a token is mandatory |
| v2.0 layout | one parquet per episode; one dataset alone is 209,880 files | "every file" is tens of millions of requests |
| audit columns | 2.7 percent of compressed parquet bytes | streaming transfers roughly 37x fewer bytes than downloading |
| `meta/info.json` | carries a `data_path` template | paths are derived, so the census makes zero tree API calls |

A partial crawl also has no sampling frame, so it supports no prevalence claim
at all. A seeded random sample is cheaper and more defensible, and the draw is
recorded before the run in the same spirit as the protocol template.

### A token is required

```bash
huggingface-cli login
# or, PowerShell, current session only:
$env:HF_TOKEN = "hf_..."
```

The token is read from `HF_TOKEN` or the CLI cache. It is never written to the
repository, never logged, and never appears in a repr or an error message.
There is a test asserting both that it reaches the request and that a
tokenless client sends no `Authorization` header at all.

### Resume

Each dataset appends one JSONL record keyed by `repo@revision`, flushed per
dataset rather than at the end. The revision is the real Hub sha, taken from
the `X-Repo-Commit` header of a response the run already makes, so it costs no
extra requests. A crash at dataset 3,000 costs one dataset, not the run, and a
dataset that changed on the Hub is audited again rather than skipped.

### Flags are provisional, always

Every CSV carries a header saying so. `flags` are automated measurements. The
`confirmed` column stays empty until a human opens the data. Nothing in this
tooling calls a dataset defective, and CI fails the build if any code tries.

## 3. `lerobot_ur`, the hardware harness

A LeRobot style follower for Universal Robots e-Series over RTDE, with the
built in force torque sensor in the observation, velocity and force clamps on
every command, and a monotonic timestamp so the audit can check sync.
`ursim_smoke.py` measures the RTDE stream rate and jitter against the official
simulator, which is the CI gate: hardware in the loop with no arm.

This is a skeleton. The LeRobot base class wiring is the next task.

## Install

```bash
pip install -e .            # library
pip install -e ".[test]"    # plus pytest and hypothesis
pip install -e ".[dev]"     # plus ruff
pip install -e ".[hardware]" # plus ur_rtde, for M-03
```

Python 3.11 or newer.

## Development

```bash
python -m pytest                            # offline suite
python -m pytest -m network                 # live Hub tests, needs a token
python -m doctest cairo_protocol/stats.py   # silence means all 17 pass
ruff check . && ruff format --check .
```

### On Windows

Windows has no `make`, so the `Makefile` targets do not run. Use:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest                            # make test
.\.venv\Scripts\python.exe -m ledger.audit --demo               # make demo
docker run --rm -d --name ursim -p 29999:29999 -p 30001-30004:30001-30004 universalrobots/ursim_e-series   # make ursim
```

The M-03 URSim jitter gate is a Linux target. Docker Desktop on Windows adds a
WSL2 network hop and Windows has a coarse default timer, so a p99.9 jitter
figure measured there describes the host, not the robot stack.

## Repository layout

```
cairo_protocol/   statistics: confidence sequences, exact tests, power
ledger/           the census
  checks.py       temporal checks in an open registry
  report.py       pooled aggregation, flagging, CSV and JSONL writers
  census.py       two tier run loop, seeded sampling, resume
  sources.py      SampleSource: streaming, download, local, synthetic
  hubclient.py    token aware HTTP with backoff and pagination
  paths.py        parquet paths derived from info.json
  synth.py        seeded generator, one injector per defect
  audit.py        the CLI
lerobot_ur/       UR5e follower and its URSim smoke test
protocols/        the pre-registration template
docs/             calibration measurements, design spec, decision record
tests/            148 tests, plus a frozen v0 reference
```

## Design decisions worth knowing

**The refactor provably changed nothing.** `tests/v0_reference.py` is a frozen
snapshot of the audit as v0 shipped it, and equivalence tests assert the
current pipeline reproduces it exactly, for single and multi part inputs. The
guarantee was verified as non tautological: a deliberate change to the pooled
denominator makes both equivalence tests fail.

**Aggregation is pooled, not averaged.** Rates are
`sum(numerator) / sum(denominator)` across parts, never a mean of per episode
fractions. Those two differ whenever episodes have unequal length, which is
exactly what dropped frames and real Hub data produce.

**Thresholds are calibrated, not guessed.** `docs/calibration.md` records the
measured false positive sweep behind every value in `ledger/thresholds.toml`.

**Adding a check is adding a function.** Checks live in a registry, so the
flag list, the CSV columns and the report schema cannot drift apart.

## Limitations

Stated plainly, because a kit that hides these is worse than no kit.

- **Lag on low degree of freedom arms.** `lag_frames` cross correlates action
  against state. Below 6 action dimensions the correlation differs so little
  between adjacent lags that the argmax is not reliably the true lag: measured
  exact match was 96.5 to 97 percent at 2 joints, against 0 to 2 mismatches
  per 6000 to 8000 trials at 6 joints and above. Treat `large_lag` and
  `negative_lag` as uninformative below 6 action dimensions and confirm by
  hand. Full measurement in `docs/calibration.md`.
- **The census has not been run at full scale yet.** Everything is verified
  offline against fixtures and a small number of live probes.
- **`--files all` on packed v3.0 layouts** discovers files by probing
  sequential indices. A source that cannot answer an existence check falls
  back to sampling one file, and the report records that the audit was
  partial.
- **`--files all` on the quick audit is capped.** The quick workflow
  (`--top`, `--repos`) downloads whole parquet files, and a v2.x dataset
  stores one per episode, so an uncapped run fills the disk. It audits the
  first 25 files and says so on stderr. Lift it with `--no-file-cap`, or use
  `--census` for a whole population survey, which streams instead of
  downloading.
- **The legacy `--top` and `--repos` Hub glue** predates this work and has no
  test coverage.
- **`lerobot_ur` is a skeleton.** No real robot number has been produced by
  this repository, and none should be until the LeRobot wiring lands.

## Citing

See `CITATION.cff`, or use the GitHub "Cite this repository" button.

## Disclosure

Experiment code, simulation harnesses, data audit tooling and first drafts
were produced with Claude and verified by the authors; all hardware
experiments, statistics and claims are the authors' own.

## License

Apache-2.0. See [LICENSE](LICENSE).
