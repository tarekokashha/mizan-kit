# Contributing

This is a research kit. Code here produces numbers that go into papers, so
the bar is a little different from a normal library: a change that makes a
result look better without being more correct is a defect, not an
improvement.

## Setup

```bash
git clone https://github.com/tarekokasha22/mizan-kit
cd mizan-kit
python -m venv .venv
.venv/bin/pip install -e ".[dev]"        # Windows: .venv\Scripts\pip
```

Python 3.11 or newer. On Windows use `.venv\Scripts\python.exe` throughout;
there is no `make`, and the PowerShell equivalents are in the README.

## The checks that must pass

```bash
python -m pytest                          # 148 tests, 1 live Hub test deselected
python -m doctest cairo_protocol/stats.py # 17 doctests, silence means pass
ruff check . && ruff format --check .
```

Live Hub tests carry the `network` marker and are deselected by default. Run
them with `python -m pytest -m network`, and expect rate limiting without a
token.

## House rules

These are enforced by CI, not just by convention.

**No em dashes.** Plain sentences, everywhere, including commit messages.

**No dataset is ever called defective.** A LEDGER flag is a measurement, not
a finding. The `confirmed` column stays empty until a human has opened the
data and filled it in. Any output that labels a dataset defective is a bug.

**Never weaken a check to make a result pass.** If a threshold is wrong,
change it deliberately, record the measurement that justified the change in
`docs/calibration.md`, and say so in the commit message. If a result is not
significant, the paper says so.

**Every rate carries a confidence interval.** `cairo_protocol` exists so that
no proportion is ever reported bare.

**Do not touch another programme's code.** `cairo_protocol/` and
`lerobot_ur/` belong to other MIZAN fronts. The linter is scoped to exclude
them for exactly this reason.

## Behaviour preservation

`tests/v0_reference.py` is a frozen snapshot of the audit as v0 shipped it.
The equivalence tests assert that the current pipeline reproduces it exactly.
Never edit that file to make a test pass. If your change alters a published
number, that is the finding: say so, and justify it.

You can prove the guarantee still has teeth by breaking it on purpose. Change
the pooled denominator in `ledger/report.py`, run
`python -m pytest tests/test_report.py`, and confirm the equivalence tests
fail. Then revert.

## Tests

Test driven, and the tests have to be able to fail. A test that asserts only
that a function returned something is not a test. When you add a check, add
a case that would fail if the check were stubbed out.

The suite runs fully offline. Read frames through `LocalSource` or
`SyntheticSource` and build fixtures with `ledger.synth.write_v20_fixture` or
`write_v30_fixture`. Nothing except the single `network` marked test may
touch the Hub.

Seed everything. Determinism is a requirement, not an aspiration.

## Adding a check

Add a function to the registry in `ledger/checks.py`:

```python
@register("my_check", threshold_key="my_check_frac")
def my_check(stats: EpisodeStats) -> float: ...
```

Then add its threshold to `ledger/thresholds.toml`, and add a property based
case to `tests/test_calibration.py` proving clean data does not trip it
across randomised fps, joint count, episode length and lag. Thresholds are
calibrated against measured false positive rates, never guessed.

Cross episode aggregates such as lag and duplicate detection are not registry
entries. They live in `ledger/report.py` and are declared in
`checks.AGGREGATE_FLAGS`.

## Commits

Present tense, and say what changed and why. If a commit changes a reported
metric, put the number in the message.
