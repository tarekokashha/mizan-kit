# MIZAN starter kit

Companion to the MIZAN research programme (the published page is the
specification; this kit is the first commit). Three things are in here and
all three run today.

## 1. `cairo_protocol/` : the statistics under every experiment

`cairo_protocol/stats.py` gives you, with doctests and a test suite:

- `anytime_cs(outcomes)` : an anytime-valid confidence sequence for a success
  rate (Robbins' beta-binomial mixture). You may look after every trial and
  stop whenever you like; coverage still holds. Checked by Monte Carlo in
  `tests/test_stats.py` (3.3 percent of runs ever miss at nominal 5 percent).
- `sequential_compare(a, b)` : declares "A>B" only when the two sequences
  separate. With equal policies it made zero false decisions in 400 runs of
  60 interleaved pairs; it separates a 0.9 from a 0.6 policy in a median of
  84 trials per arm.
- `barnard_test`, `wilson_interval`, `bootstrap_diff_ci`, `trials_needed`
  for the fixed-n parts of a protocol.

`protocols/PROTOCOL_TEMPLATE.md` is the pre-registration file every
programme commits before its first trial.

## 2. `ledger/` : the M-02 audit, v0, running against the Hub

```
python -m ledger.audit --demo                       # offline, injects defects, proves the checks
python -m ledger.audit --repos lerobot/svla_so101_pickplace,lerobot/aloha_static_cups_open
python -m ledger.audit --top 50 --files 1 --out ledger_report.csv
```

First live run on 2 September 2026 (one parquet file per dataset):

```
repo                               fps   eps  frac_bad_dt  stuck_state  identity  lag_frames  r_lag0  r_best  flags
lerobot/svla_so101_pickplace       30    50   0.000        0.202        0.000     4.0         0.960   0.998   stuck_state|large_lag
lerobot/aloha_sim_insertion_human  50    15   0.000        0.000        0.000     1.0         0.937   0.938
lerobot/aloha_static_cups_open     50    50   0.000        0.001        0.000     2.0         0.985   0.987
IPEC-COMMUNITY/kuka_lerobot        10    1    0.000        0.000        nan       nan         nan     nan
```

Read that first row the way the paper will have to: a 4-frame lag at 30 fps
is 133 ms between the commanded action and the state that reaches it, and
one frame in five repeats the previous joint reading exactly. Both may have
innocent explanations (servo read quantisation while the arm is still, or a
leader-follower pipeline that is simply slow) and both are exactly the kind
of thing a policy trained on the data inherits. Nothing is a finding until
you open the dataset and confirm it; that rule is in `CLAUDE.md`.

## 3. `lerobot_ur/` : the M-03 skeleton and its URSim CI

- `robot_ur5e.py` : a follower class over RTDE with the built-in force-torque
  sensor in the observation, velocity and force clamps on every command,
  and a monotonic timestamp so the audit can check sync. The LeRobot base
  class wiring is the first overnight task (see the file header).
- `ursim_smoke.py` : starts the stream against the official UR simulator and
  reports rate and p50/p99/p99.9 jitter. This is the CI gate.
- `.github/workflows/ursim-ci.yml` : runs URSim as a service container and
  the smoke test on every push. Hardware-in-the-loop CI with no arm.


## 4. `ledger/` census : the Hub wide survey (M-02, v1)

The v0 audit above samples named datasets. The census surveys the whole
LeRobot population and reports prevalence with confidence intervals.

```
python -m ledger.audit --census metadata --out-dir census_out
python -m ledger.audit --census deep --sample-size 800 --seed 20260905 --out-dir census_out --resume
```

### Two tiers, and why it is not an exhaustive crawl

Tier 1, metadata, visits every LeRobot dataset and records codebase version,
fps, episode and frame counts, chunk size, layout family and the feature
schema. Tier 2, deep, runs the temporal checks over a seeded random sample
drawn from that frame.

Tier 2's `--files` picks how many parquet files are sampled per dataset, an
integer or `all`. For a per-episode (v2.0) layout `all` is exact, since the
episode count is already in `info.json`. For a packed (v3.0) layout the file
count is not in `info.json` at all (see the table below), so `all` instead
probes file indices one at a time and stops at the first one that is not
there; that needs the source to be able to check whether a path exists,
which `--source stream`, `download` and `local` all can. A source that
cannot falls back to sampling one file and marks that report's `note` column
partial, rather than silently claiming full coverage it did not have.

Prevalence is a binomial proportion, so it is reported through
`cairo_protocol.wilson_interval` and never as a bare rate. A sample of 800
datasets pins any prevalence to about plus or minus 2.8 points at 95 percent,
and 1600 to about 2.0 points.

An exhaustive every file crawl is not attempted. These are measurements taken
from this machine on 5 September 2026, not estimates:

| finding | value |
|---------|-------|
| LeRobot datasets on the Hub | at least 12,000, cursor paginated at 1000 per page |
| anonymous rate limit | HTTP 429 within tens of requests, on both the api and resolve hosts |
| v2.0 layout | one parquet per episode; `IPEC-COMMUNITY/kuka_lerobot` alone is 209,880 files |
| v3.0 layout | episodes packed into large files |
| audit columns | 2.7 percent of compressed parquet bytes, so streaming transfers roughly 37x fewer bytes than downloading |
| `meta/info.json` | carries a `data_path` template, so paths are derived and no tree API call is needed |

Across at least 12,000 datasets that is tens of millions of requests against a
free service. A partial crawl also has no sampling frame, so it supports no
prevalence claim at all. A seeded random sample is both cheaper and more
defensible, and the draw is recorded before the run in the same spirit as
`protocols/PROTOCOL_TEMPLATE.md`.

### A token is required

Anonymous requests are rate limited within about a minute. Set one of:

```
huggingface-cli login
$env:HF_TOKEN = "hf_..."        # PowerShell, current session only
```

The token is read from `HF_TOKEN` or the CLI cache. It is never written to the
repository and never logged.

### Resume

Each dataset appends one JSONL record keyed by `repo@revision`, flushed per
dataset rather than at the end. The revision is the real Hub sha, taken from
the `X-Repo-Commit` header of a response the run already makes. Re running
with `--resume` skips what is already recorded, so a crash at dataset 3,000
costs one dataset and not the run, and a dataset that changed on the Hub is
audited again rather than skipped.

### Flags are provisional

Every CSV carries a header saying so. `flags` are automated measurements; the
`confirmed` column stays empty until a human opens the data. Nothing in this
tooling calls a dataset defective, per `CLAUDE.md`.

### Known limitation: lag on low degree of freedom arms

`lag_frames` is estimated by cross correlating action against state. Below 6
action dimensions the correlation differs so little between adjacent lags that
the argmax is not reliably the true lag: measured exact match was 96.5 to 97
percent at 2 joints, against 0 to 2 mismatches per 6000 to 8000 trials at 6
joints and above. Treat `large_lag` and `negative_lag` as uninformative for
datasets with fewer than 6 action dimensions and confirm by hand. The full
measurement is in `docs/calibration.md`.

## Run everything

```
pip install -r requirements.txt
pytest -q
python -m ledger.audit --demo
make ursim && sleep 60 && python -m lerobot_ur.ursim_smoke
```

### On Windows

Windows does not ship GNU make, so the `Makefile` targets above do not run.
The PowerShell equivalents are:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q                       # make test
.\.venv\Scripts\python.exe -m ledger.audit --demo             # make demo
.\.venv\Scripts\python.exe -m ledger.audit --top 50 --files 1 --out ledger_report.csv   # make audit
docker run --rm -d --name ursim -p 5900:5900 -p 6080:6080 -p 29999:29999 -p 30001-30004:30001-30004 universalrobots/ursim_e-series   # make ursim
```

Live Hub tests are deselected by default. Run them with `-m network`.

The M-03 URSim jitter gate is a Linux target. Docker Desktop on Windows adds a
WSL2 network hop and Windows has a coarse default timer, so a p99.9 jitter
figure measured there describes the host, not the robot stack. Run that gate
on Linux.

## The rules

`CLAUDE.md` is the operating contract for the engine: what it may edit, the
reviewer-two pass, citation verification, the disclosure line, and the four
things it never does (write a real-robot number, run an attack outside URSim,
name a dataset without a human opening it, open a second front).
