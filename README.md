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

## Run everything

```
pip install -r requirements.txt
pytest -q                     # 8 tests
python -m ledger.audit --demo
make ursim && sleep 60 && python -m lerobot_ur.ursim_smoke
```

## The rules

`CLAUDE.md` is the operating contract for the engine: what it may edit, the
reviewer-two pass, citation verification, the disclosure line, and the four
things it never does (write a real-robot number, run an attack outside URSim,
name a dataset without a human opening it, open a second front).
