# M-02 LEDGER Census Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `ledger/audit.py` v0 from a single file sampler into a resumable two tier census that produces a Hub wide metadata table plus a sampled temporal audit reported with confidence intervals.

**Architecture:** Split the one module along the seams already inside it into `hubclient`, `paths`, `sources`, `checks`, `report`, `census` and a thin `audit` CLI. Frames reach the checks through a `SampleSource` abstraction with four implementations, so the whole system is testable with no network. Checks live in a registry so adding one is adding a function.

**Tech Stack:** Python 3.11, numpy, pandas, pyarrow, huggingface_hub, pytest, hypothesis. Standard library `tomllib` for config.

**Spec:** `docs/superpowers/specs/2026-09-05-ledger-census-design.md`

## Global Constraints

- No em dashes in any prose, including docstrings, comments and commit messages. Plain sentences. (`CLAUDE.md`)
- Never name a dataset defective in any output. Automated results are `flags`, provisional until a human confirms. (`CLAUDE.md`)
- Never weaken or remove a check to make a result look better. (`CLAUDE.md`)
- Do not touch `cairo_protocol/` or `lerobot_ur/`. One active front. (`CLAUDE.md`)
- Every reported metric carries its confidence interval or confidence sequence. (`CLAUDE.md`)
- The existing 8 pytest tests and 17 doctests must pass unchanged at every commit.
- `python -m ledger.audit --demo` must keep producing today's flags exactly.
- The HF token is read from `HF_TOKEN` or the CLI cache. Never write it to the repo, never log it.
- Default deep sample size is 800 datasets, seed 20260905, both configurable.
- Audit columns are exactly: `timestamp`, `frame_index`, `episode_index`, `action`, `observation.state`.
- Run tests with `./.venv/Scripts/python.exe -m pytest`, not a bare `pytest`.

## File Structure

| file | responsibility |
|------|----------------|
| `ledger/synth.py` | seeded synthetic LeRobot episode generator, one defect injector per defect |
| `ledger/checks.py` | check registry, one function per defect, no I/O |
| `ledger/thresholds.toml` | threshold values, config not code |
| `ledger/config.py` | load and validate thresholds |
| `ledger/paths.py` | derive parquet paths from the `data_path` template |
| `ledger/hubclient.py` | token aware HTTP with retry, backoff and a global rate limit |
| `ledger/sources.py` | `SampleSource` protocol and four implementations |
| `ledger/report.py` | `DatasetReport`, flagging, CSV and JSONL writers |
| `ledger/census.py` | two tier run loop, sampling frame, resume ledger |
| `ledger/audit.py` | thin CLI, keeps every v0 command working |
| `tests/conftest.py` | shared fixtures, network marker |
| `tests/test_*.py` | one test module per source module |

---

### Task 1: Seeded synthetic generator and parquet fixtures

Everything downstream is tested against this. It replaces the private
`_synthetic` in `audit.py` with a parameterised, seeded generator, and
adds fixture writers for both on disk layouts.

**Files:**
- Create: `ledger/synth.py`
- Create: `tests/test_synth.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `make_episodes(fps=30.0, n_eps=6, T=300, n_joints=6, lag=2, defect="", seed=0, noise=1e-3) -> pandas.DataFrame` with columns `timestamp` (float32), `frame_index` (int64), `episode_index` (int64), `action` (list of float32), `observation.state` (list of float32). `DEFECTS: tuple[str, ...]` listing every injectable defect name including `""` for clean. `write_v20_fixture(df, root, repo) -> dict` and `write_v30_fixture(df, root, repo) -> dict`, each writing parquet plus `meta/info.json` and returning the info dict.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_synth.py
import numpy as np
import pytest
from ledger.synth import make_episodes, DEFECTS


def test_clean_shape_and_dtypes():
    df = make_episodes(fps=30.0, n_eps=4, T=100, n_joints=6, seed=1)
    assert len(df) == 400
    assert set(df.columns) == {"timestamp", "frame_index", "episode_index",
                               "action", "observation.state"}
    assert df["episode_index"].nunique() == 4
    assert len(df["action"].iloc[0]) == 6


def test_is_deterministic_under_seed():
    a = make_episodes(seed=7)
    b = make_episodes(seed=7)
    assert np.array_equal(np.stack(a["action"].to_numpy()),
                          np.stack(b["action"].to_numpy()))


def test_different_seeds_differ():
    a = make_episodes(seed=1)
    b = make_episodes(seed=2)
    assert not np.array_equal(np.stack(a["action"].to_numpy()),
                              np.stack(b["action"].to_numpy()))


@pytest.mark.parametrize("defect", DEFECTS)
def test_every_defect_generates(defect):
    df = make_episodes(defect=defect, seed=0)
    assert len(df) > 0


def test_n_joints_is_honoured():
    df = make_episodes(n_joints=14, T=50, n_eps=2, seed=3)
    assert len(df["observation.state"].iloc[0]) == 14
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_synth.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.synth'`

- [ ] **Step 3: Write minimal implementation**

Port the body of `_synthetic` from `ledger/audit.py` and parameterise it.
Key correctness point: the clean case must build `state` as the action
delayed by `lag` frames, so a lag check has something true to find. The
`duplicate` defect must copy episode 0's action array into later
episodes. Use `np.random.default_rng(seed)` and never the global RNG.

```python
# ledger/synth.py
"""Seeded synthetic LeRobot style episodes, with one injector per defect.

Used by the demo, the unit tests and the property based calibration.
Every function is pure given its seed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFECTS = ("", "identity", "drops", "stuck", "swapped", "duplicate")


def make_episodes(fps: float = 30.0, n_eps: int = 6, T: int = 300,
                  n_joints: int = 6, lag: int = 2, defect: str = "",
                  seed: int = 0, noise: float = 1e-3) -> pd.DataFrame:
    if defect not in DEFECTS:
        raise ValueError(f"unknown defect {defect!r}, expected one of {DEFECTS}")
    rng = np.random.default_rng(seed)
    rows = []
    first_action = None
    for e in range(n_eps):
        t = np.arange(T) / fps
        base = np.cumsum(rng.normal(0, 0.02, size=(T + lag + 5, n_joints)), axis=0)
        action = base[lag:T + lag]
        state = base[:T] + rng.normal(0, noise, size=(T, n_joints))
        if defect == "identity":
            action = state.copy()
        if defect == "drops" and e % 2 == 0:
            keep = np.ones(T, bool)
            keep[rng.choice(T, size=max(1, T // 8), replace=False)] = False
            t, action, state = t[keep], action[keep], state[keep]
        if defect == "stuck" and e % 2 == 0:
            hi = min(200, len(state))
            state[50:hi] = state[50]
        if defect == "swapped":
            action, state = state, action
        if defect == "duplicate" and first_action is not None:
            action = first_action
        if e == 0:
            first_action = action
        for i in range(len(t)):
            rows.append({
                "timestamp": np.float32(t[i]),
                "frame_index": i,
                "episode_index": e,
                "action": action[i].astype(np.float32),
                "observation.state": state[i].astype(np.float32),
            })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_synth.py -q`
Expected: PASS, 8 tests

- [ ] **Step 5: Add the fixture writers and their tests**

Append to `tests/test_synth.py`:

```python
import json
from ledger.synth import write_v20_fixture, write_v30_fixture


def test_v20_fixture_round_trips(tmp_path):
    df = make_episodes(n_eps=3, T=40, seed=5)
    info = write_v20_fixture(df, tmp_path, "acme/v20demo")
    assert info["codebase_version"] == "v2.0"
    assert "episode_{episode_index:06d}" in info["data_path"]
    root = tmp_path / "acme/v20demo"
    assert json.loads((root / "meta/info.json").read_text())["fps"] == 30
    assert len(list(root.glob("data/**/*.parquet"))) == 3  # one per episode


def test_v30_fixture_round_trips(tmp_path):
    df = make_episodes(n_eps=3, T=40, seed=5)
    info = write_v30_fixture(df, tmp_path, "acme/v30demo")
    assert info["codebase_version"] == "v3.0"
    assert "file-{file_index:03d}" in info["data_path"]
    root = tmp_path / "acme/v30demo"
    assert len(list(root.glob("data/**/*.parquet"))) == 1  # episodes packed
```

Implementation, appended to `ledger/synth.py`. v2.0 writes one parquet
per episode, v3.0 packs every episode into one file, matching the two
layouts measured on the Hub.

```python
import json
from pathlib import Path


def _write_info(root: Path, info: dict) -> dict:
    (root / "meta").mkdir(parents=True, exist_ok=True)
    (root / "meta/info.json").write_text(json.dumps(info, indent=2))
    return info


def write_v20_fixture(df: pd.DataFrame, root, repo: str, fps: float = 30.0) -> dict:
    root = Path(root) / repo
    for ep, sub in df.groupby("episode_index", sort=True):
        p = root / f"data/chunk-000/episode_{int(ep):06d}.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        sub.to_parquet(p, index=False)
    return _write_info(root, {
        "codebase_version": "v2.0", "fps": fps, "chunks_size": 1000,
        "total_episodes": int(df["episode_index"].nunique()),
        "total_frames": int(len(df)),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
    })


def write_v30_fixture(df: pd.DataFrame, root, repo: str, fps: float = 30.0) -> dict:
    root = Path(root) / repo
    p = root / "data/chunk-000/file-000.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return _write_info(root, {
        "codebase_version": "v3.0", "fps": fps, "chunks_size": 1000,
        "total_episodes": int(df["episode_index"].nunique()),
        "total_frames": int(len(df)),
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
    })
```

Also create `tests/conftest.py` registering the network marker so live
tests are opt in:

```python
# tests/conftest.py
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "network: hits the live Hugging Face Hub")


def pytest_collection_modifyitems(config, items):
    if config.getoption("-m"):
        return
    skip = pytest.mark.skip(reason="needs --m network to run live Hub tests")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
```

- [ ] **Step 6: Run the whole suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, 8 original plus 10 new

- [ ] **Step 7: Commit**

```bash
git add ledger/synth.py tests/test_synth.py tests/conftest.py
git commit -m "Add seeded synthetic episode generator and parquet fixtures"
```

---

### Task 2: Check registry

Move the checks out of the `if` ladder into a registry so that adding a
check is adding a function, and so the report schema cannot drift from
the checks that populate it. Behaviour must not change.

**Files:**
- Create: `ledger/checks.py`
- Create: `tests/test_checks.py`

**Interfaces:**
- Consumes: `ledger.synth.make_episodes`.
- Produces: `register(name, threshold_key=None, higher_is_worse=True)` decorator. `REGISTRY: dict[str, Check]` where `Check` has fields `name`, `fn`, `threshold_key`, `higher_is_worse`. `EpisodeStats` dataclass with fields `n_frames`, `ts`, `action`, `state`, `frame_index`, `fps`. `episode_stats(df_group, fps) -> EpisodeStats`. `run_checks(stats) -> dict[str, float]` returning one value per registered check. `xcorr_lag(action, state, lags) -> tuple[int, float, float]` unchanged from v0.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_checks.py
import numpy as np
import pytest
from ledger.checks import REGISTRY, run_checks, episode_stats, xcorr_lag
from ledger.synth import make_episodes


def _stats(defect="", **kw):
    df = make_episodes(defect=defect, seed=0, **kw)
    ep = df[df["episode_index"] == 0]
    return episode_stats(ep, fps=30.0)


def test_registry_is_populated():
    assert {"frac_bad_dt", "stuck_state_frac", "identity_frac",
            "dup_episode_frac"} <= set(REGISTRY)


def test_clean_episode_is_quiet():
    v = run_checks(_stats(""))
    assert v["identity_frac"] == 0.0
    assert v["stuck_state_frac"] < 0.05
    assert v["frac_bad_dt"] == 0.0


def test_identity_defect_is_seen():
    assert run_checks(_stats("identity"))["identity_frac"] == 1.0


def test_stuck_defect_is_seen():
    assert run_checks(_stats("stuck"))["stuck_state_frac"] > 0.2


def test_xcorr_recovers_positive_lag():
    df = make_episodes(lag=3, T=400, n_eps=1, seed=2)
    a = np.stack(df["action"].to_numpy())
    s = np.stack(df["observation.state"].to_numpy())
    best, r0, rb = xcorr_lag(a, s, range(-5, 11))
    assert best == 3
    assert rb >= r0


def test_xcorr_returns_nan_for_short_series():
    a = np.zeros((5, 6)); s = np.zeros((5, 6))
    best, r0, rb = xcorr_lag(a, s, range(-5, 11))
    assert np.isnan(r0) and np.isnan(rb)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_checks.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.checks'`

- [ ] **Step 3: Write minimal implementation**

Port `_stack`, `_xcorr_lag` and the per episode body of `audit_frame`
from `ledger/audit.py`, unchanged in arithmetic. The only structural
change is that each measurement becomes a registered function.

```python
# ledger/checks.py
"""One function per temporal defect, held in a registry.

Adding a check is adding a function. The report schema, the CSV columns
and the flag names are all derived from REGISTRY, so they cannot drift
apart from the checks that fill them.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

LAGS = tuple(range(-5, 11))


@dataclass(frozen=True)
class Check:
    name: str
    fn: Callable[["EpisodeStats"], float]
    threshold_key: str | None
    higher_is_worse: bool


REGISTRY: dict[str, Check] = {}


def register(name: str, threshold_key: str | None = None,
             higher_is_worse: bool = True):
    def deco(fn):
        REGISTRY[name] = Check(name, fn, threshold_key or name, higher_is_worse)
        return fn
    return deco


@dataclass
class EpisodeStats:
    n_frames: int
    fps: float
    ts: np.ndarray | None
    frame_index: np.ndarray | None
    action: np.ndarray | None
    state: np.ndarray | None


def _stack(col) -> np.ndarray | None:
    try:
        arr = np.stack([np.asarray(v, dtype=np.float64) for v in col])
        return arr if arr.ndim == 2 else None
    except Exception:
        return None


def episode_stats(ep, fps: float) -> EpisodeStats:
    ts = np.asarray(ep["timestamp"], dtype=np.float64) if "timestamp" in ep else None
    fi = np.asarray(ep["frame_index"], dtype=np.int64) if "frame_index" in ep else None
    ac = _stack(ep["action"]) if "action" in ep else None
    st = _stack(ep["observation.state"]) if "observation.state" in ep else None
    return EpisodeStats(len(ep), fps, ts, fi, ac, st)


def xcorr_lag(action, state, lags: Iterable[int] = LAGS):
    T = len(action)
    if T < 30:
        return 0, float("nan"), float("nan")
    a = action - action.mean(0)
    s = state - state.mean(0)
    sa = a.std(0) + 1e-9
    ss = s.std(0) + 1e-9
    scores = {}
    for k in lags:
        if k >= 0:
            aa, st = a[:T - k], s[k:]
        else:
            aa, st = a[-k:], s[:T + k]
        if len(aa) < 20:
            continue
        scores[k] = float(np.nanmean(((aa * st).mean(0)) / (sa * ss)))
    if not scores:
        return 0, float("nan"), float("nan")
    best = max(scores, key=scores.get)
    return best, scores.get(0, float("nan")), scores[best]


@register("frac_bad_dt")
def frac_bad_dt(s: EpisodeStats) -> float:
    if s.ts is None or len(s.ts) < 2 or not (s.fps and s.fps > 0):
        return float("nan")
    expected = 1.0 / s.fps
    dt = np.diff(s.ts)
    return float(np.sum(np.abs(dt - expected) > 0.25 * expected) / len(dt))


@register("stuck_state_frac")
def stuck_state_frac(s: EpisodeStats) -> float:
    if s.state is None or len(s.state) < 2:
        return float("nan")
    same = np.all(s.state[1:] == s.state[:-1], axis=1)
    return float(same.sum() / len(same))


@register("identity_frac")
def identity_frac(s: EpisodeStats) -> float:
    if s.state is None or s.action is None or s.state.shape != s.action.shape:
        return float("nan")
    eq = np.all(np.isclose(s.action, s.state, atol=0.0), axis=1)
    return float(eq.sum() / len(eq))


@register("dup_episode_frac")
def episode_action_hash(s: EpisodeStats) -> float:
    return float("nan")  # aggregated across episodes in report.py


def run_checks(stats: EpisodeStats) -> dict[str, float]:
    return {name: c.fn(stats) for name, c in REGISTRY.items()}


def action_head_hash(s: EpisodeStats, n: int = 50) -> str | None:
    if s.action is None:
        return None
    return hashlib.md5(np.round(s.action[:n], 4).tobytes()).hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_checks.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Verify the original suite still passes**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, all previous tests unchanged

- [ ] **Step 6: Commit**

```bash
git add ledger/checks.py tests/test_checks.py
git commit -m "Extract temporal checks into an open registry"
```

---

### Task 3: Thresholds as configuration

**Files:**
- Create: `ledger/thresholds.toml`
- Create: `ledger/config.py`
- Create: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_thresholds(path=None) -> dict[str, float]` reading the packaged TOML by default. `DEFAULT_THRESHOLDS: dict[str, float]`. Raises `ValueError` naming the offending key when a threshold is missing or out of the unit interval where one is required.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from ledger.config import load_thresholds, DEFAULT_THRESHOLDS


def test_defaults_match_v0_values():
    t = load_thresholds()
    assert t["frac_bad_dt"] == 0.05
    assert t["stuck_state_frac"] == 0.20
    assert t["identity_frac"] == 0.50
    assert t["lag_large"] == 3
    assert t["dup_episode_frac"] == 0.0


def test_rejects_out_of_range_fraction(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("[thresholds]\nfrac_bad_dt = 1.5\nstuck_state_frac = 0.2\n"
                 "identity_frac = 0.5\nlag_large = 3\ndup_episode_frac = 0.0\n")
    with pytest.raises(ValueError, match="frac_bad_dt"):
        load_thresholds(p)


def test_rejects_missing_key(tmp_path):
    p = tmp_path / "short.toml"
    p.write_text("[thresholds]\nfrac_bad_dt = 0.05\n")
    with pytest.raises(ValueError, match="stuck_state_frac"):
        load_thresholds(p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.config'`

- [ ] **Step 3: Write minimal implementation**

`ledger/thresholds.toml`, carrying the v0 values and a comment recording
that they are uncalibrated until Task 4 runs:

```toml
# Thresholds for the M-02 LEDGER audit.
# Values below are the v0 defaults. They are deliberately loose. Task 4
# calibrates them against clean synthetic data and records the measured
# false positive rate in docs. Do not tighten a threshold to make a
# result look better.
[thresholds]
frac_bad_dt = 0.05
stuck_state_frac = 0.20
identity_frac = 0.50
lag_large = 3
dup_episode_frac = 0.0
```

```python
# ledger/config.py
"""Threshold loading and validation. Config, not code."""
from __future__ import annotations

import tomllib
from pathlib import Path

_REQUIRED = ("frac_bad_dt", "stuck_state_frac", "identity_frac",
             "lag_large", "dup_episode_frac")
_FRACTIONS = ("frac_bad_dt", "stuck_state_frac", "identity_frac",
              "dup_episode_frac")
_DEFAULT_PATH = Path(__file__).with_name("thresholds.toml")


def load_thresholds(path=None) -> dict[str, float]:
    path = Path(path) if path is not None else _DEFAULT_PATH
    data = tomllib.loads(path.read_text(encoding="utf-8")).get("thresholds", {})
    for key in _REQUIRED:
        if key not in data:
            raise ValueError(f"threshold {key!r} missing from {path}")
    for key in _FRACTIONS:
        if not 0.0 <= float(data[key]) <= 1.0:
            raise ValueError(f"threshold {key!r} must be in [0, 1], got {data[key]}")
    if float(data["lag_large"]) < 0:
        raise ValueError("threshold 'lag_large' must be non negative")
    return {k: float(v) for k, v in data.items()}


DEFAULT_THRESHOLDS = load_thresholds()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -q`
Expected: PASS, 3 tests

- [ ] **Step 5: Commit**

```bash
git add ledger/thresholds.toml ledger/config.py tests/test_config.py
git commit -m "Move audit thresholds into validated configuration"
```

---

### Task 4: Property based threshold calibration

This is the task the README asks for when it says thresholds must be
calibrated on known good data before naming anyone. It converts a
guess into a measured false positive rate.

**Files:**
- Create: `tests/test_calibration.py`
- Modify: `ledger/thresholds.toml` (only if a property fails)
- Create: `docs/calibration.md`

**Interfaces:**
- Consumes: `ledger.synth.make_episodes`, `ledger.checks.run_checks`, `ledger.checks.xcorr_lag`, `ledger.config.load_thresholds`.
- Produces: no new API. Produces `docs/calibration.md` recording the measured false positive rate and the settings that produced it.

- [ ] **Step 1: Write the property tests**

```python
# tests/test_calibration.py
"""Calibration: thresholds must not fire on clean data, and the lag
estimator must recover a lag it was given. Both are properties, checked
over randomised but seeded configurations."""
import numpy as np
from hypothesis import given, settings, strategies as st

from ledger.checks import episode_stats, run_checks, xcorr_lag
from ledger.config import load_thresholds
from ledger.synth import make_episodes

T_ = load_thresholds()


@given(
    fps=st.sampled_from([10.0, 20.0, 30.0, 50.0, 60.0]),
    n_joints=st.integers(min_value=1, max_value=14),
    T=st.integers(min_value=60, max_value=400),
    lag=st.integers(min_value=0, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=150, deadline=None)
def test_clean_data_never_trips_a_threshold(fps, n_joints, T, lag, seed):
    df = make_episodes(fps=fps, n_eps=1, T=T, n_joints=n_joints,
                       lag=lag, defect="", seed=seed)
    v = run_checks(episode_stats(df, fps))
    assert v["frac_bad_dt"] <= T_["frac_bad_dt"]
    assert v["identity_frac"] <= T_["identity_frac"]
    assert v["stuck_state_frac"] <= T_["stuck_state_frac"]


@given(
    lag=st.integers(min_value=0, max_value=6),
    n_joints=st.integers(min_value=2, max_value=14),
    seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=100, deadline=None)
def test_injected_lag_is_recovered(lag, n_joints, seed):
    df = make_episodes(n_eps=1, T=400, n_joints=n_joints, lag=lag,
                       defect="", seed=seed)
    a = np.stack(df["action"].to_numpy())
    s = np.stack(df["observation.state"].to_numpy())
    best, _, _ = xcorr_lag(a, s)
    assert best == lag


@given(seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=60, deadline=None)
def test_identity_defect_is_always_caught(seed):
    df = make_episodes(n_eps=1, T=200, defect="identity", seed=seed)
    v = run_checks(episode_stats(df, 30.0))
    assert v["identity_frac"] > T_["identity_frac"]
```

- [ ] **Step 2: Run the properties**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_calibration.py -q`
Expected: PASS. If `test_clean_data_never_trips_a_threshold` fails,
hypothesis prints the minimal failing configuration. Record it, then
either fix the check or widen the threshold in `thresholds.toml`, and
write the reason into `docs/calibration.md`. Never narrow a defect
threshold to force a pass.

- [ ] **Step 3: Measure and record the false positive rate**

Write `docs/calibration.md` containing: the date, the hypothesis example
count, the ranges swept for fps, joint count, episode length and lag,
whether any threshold was changed and why, and the resulting false
positive count over the sweep. Copy the exact numbers from the run. Do
not estimate them.

- [ ] **Step 4: Run the whole suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_calibration.py docs/calibration.md ledger/thresholds.toml
git commit -m "Calibrate audit thresholds against clean synthetic data"
```

---

### Task 5: Parquet path derivation

Removes every `api/.../tree` call, which is what the anonymous rate limit
punishes hardest.

**Files:**
- Create: `ledger/paths.py`
- Create: `tests/test_paths.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `layout_family(info) -> str` returning `"per_episode"` or `"packed"`. `derive_paths(info, limit=None) -> list[str]`. `estimate_file_count(info) -> int | None`, returning `None` when the count is not derivable, which is the packed case.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_paths.py
import pytest
from ledger.paths import derive_paths, layout_family, estimate_file_count

V20 = {"codebase_version": "v2.0", "chunks_size": 1000, "total_episodes": 2500,
       "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"}
V30 = {"codebase_version": "v3.0", "chunks_size": 1000, "total_episodes": 50,
       "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}


def test_family_detection():
    assert layout_family(V20) == "per_episode"
    assert layout_family(V30) == "packed"


def test_v20_paths_are_derived_arithmetically():
    p = derive_paths(V20, limit=3)
    assert p == ["data/chunk-000/episode_000000.parquet",
                 "data/chunk-000/episode_000001.parquet",
                 "data/chunk-000/episode_000002.parquet"]


def test_v20_chunk_rolls_over_at_chunks_size():
    p = derive_paths(V20)
    assert len(p) == 2500
    assert p[1000] == "data/chunk-001/episode_001000.parquet"
    assert p[2499] == "data/chunk-002/episode_002499.parquet"


def test_v30_starts_at_first_file():
    p = derive_paths(V30, limit=1)
    assert p == ["data/chunk-000/file-000.parquet"]


def test_estimate_counts():
    assert estimate_file_count(V20) == 2500
    assert estimate_file_count(V30) is None


def test_unknown_template_raises():
    with pytest.raises(ValueError, match="data_path"):
        derive_paths({"codebase_version": "v9", "data_path": "weird/{nope}.parquet"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_paths.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.paths'`

- [ ] **Step 3: Write minimal implementation**

```python
# ledger/paths.py
"""Derive parquet paths from the info.json data_path template.

Measured on 2026-09-05: every LeRobot dataset carries data_path, so no
tree API call is needed. That matters because the tree endpoint is the
one the anonymous rate limit punishes hardest.

Two families exist on the Hub:
  per_episode  v2.x, one parquet per episode, count is total_episodes
  packed       v3.x, episodes packed into large files, count unknown
               without meta/episodes, so paths are probed in order.
"""
from __future__ import annotations


def layout_family(info: dict) -> str:
    tmpl = info.get("data_path", "")
    if "episode_index" in tmpl and "episode_chunk" in tmpl:
        return "per_episode"
    if "file_index" in tmpl and "chunk_index" in tmpl:
        return "packed"
    raise ValueError(f"unrecognised data_path template: {tmpl!r}")


def estimate_file_count(info: dict) -> int | None:
    if layout_family(info) == "per_episode":
        return int(info.get("total_episodes", 0))
    return None


def derive_paths(info: dict, limit: int | None = None) -> list[str]:
    tmpl = info["data_path"]
    family = layout_family(info)
    chunk = int(info.get("chunks_size", 1000)) or 1000
    if family == "per_episode":
        n = int(info.get("total_episodes", 0))
        if limit is not None:
            n = min(n, limit)
        return [tmpl.format(episode_chunk=e // chunk, episode_index=e)
                for e in range(n)]
    n = limit if limit is not None else 1
    return [tmpl.format(chunk_index=0, file_index=f) for f in range(n)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_paths.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add ledger/paths.py tests/test_paths.py
git commit -m "Derive parquet paths from info.json instead of the tree API"
```

---

### Task 6: Token aware Hub client

**Files:**
- Create: `ledger/hubclient.py`
- Create: `tests/test_hubclient.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `HubClient(min_interval=0.2, max_retries=5, timeout=45, token=None)` with methods `get_json(url) -> dict | list`, `get_info(repo) -> dict | None` returning `None` on 404 or 403, and `iter_datasets(filter_tag="LeRobot", page_size=1000, max_pages=None) -> Iterator[dict]` following the Link header cursor. `RateLimited` exception. The client never logs the token.

- [ ] **Step 1: Write the failing test**

Tests use a fake opener so no network is touched.

```python
# tests/test_hubclient.py
import json
import pytest
from ledger.hubclient import HubClient, RateLimited


class FakeResp:
    def __init__(self, body, headers=None, status=200):
        self._b = json.dumps(body).encode(); self.headers = headers or {}; self.status = status
    def read(self): return self._b
    def __enter__(self): return self
    def __exit__(self, *a): return False


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}
    import urllib.error

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)
        return FakeResp({"ok": True})

    c = HubClient(min_interval=0.0)
    monkeypatch.setattr(c, "_open", opener)
    assert c.get_json("https://example/x") == {"ok": True}
    assert calls["n"] == 3


def test_gives_up_and_raises(monkeypatch):
    import urllib.error

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 429, "rate", {"Retry-After": "0"}, None)

    c = HubClient(min_interval=0.0, max_retries=2)
    monkeypatch.setattr(c, "_open", opener)
    with pytest.raises(RateLimited):
        c.get_json("https://example/x")


def test_missing_info_returns_none(monkeypatch):
    import urllib.error

    def opener(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "nope", {}, None)

    c = HubClient(min_interval=0.0)
    monkeypatch.setattr(c, "_open", opener)
    assert c.get_info("acme/missing") is None


def test_token_is_never_in_repr():
    c = HubClient(token="hf_secretvalue")
    assert "hf_secretvalue" not in repr(c)
    assert "hf_secretvalue" not in str(c.__dict__.get("_headers", {}))


def test_pagination_follows_link_header(monkeypatch):
    pages = [
        (FakeResp([{"id": "a"}], {"Link": '<https://example/p2>; rel="next"'})),
        (FakeResp([{"id": "b"}], {})),
    ]
    seq = iter(pages)
    c = HubClient(min_interval=0.0)
    monkeypatch.setattr(c, "_open", lambda req, timeout=None: next(seq))
    assert [d["id"] for d in c.iter_datasets()] == ["a", "b"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_hubclient.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.hubclient'`

- [ ] **Step 3: Write minimal implementation**

The token is held in a private headers dict and deliberately excluded
from `__repr__`, because a census log is the kind of file that gets
pasted into an issue.

```python
# ledger/hubclient.py
"""Token aware Hub access with one global rate limit.

Measured on 2026-09-05: anonymous requests hit HTTP 429 with
"maximum queue size reached" within tens of requests, on both the api
and resolve hosts. A census therefore needs a token and needs to behave.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Iterator

API = "https://huggingface.co/api"
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/main/meta/info.json"


class RateLimited(RuntimeError):
    pass


def _discover_token() -> str | None:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


class HubClient:
    def __init__(self, min_interval: float = 0.2, max_retries: int = 5,
                 timeout: float = 45.0, token: str | None = None,
                 user_agent: str = "mizan-ledger/0.1"):
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._headers = {"User-Agent": user_agent}
        tok = token if token is not None else _discover_token()
        self._has_token = bool(tok)
        if tok:
            self._headers["Authorization"] = f"Bearer {tok}"
        self._last = 0.0

    def __repr__(self) -> str:
        return f"HubClient(authenticated={self._has_token}, min_interval={self.min_interval})"

    @property
    def authenticated(self) -> bool:
        return self._has_token

    def _open(self, req, timeout=None):
        return urllib.request.urlopen(req, timeout=timeout)

    def _throttle(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.min_interval:
            time.sleep(self.min_interval - gap)
        self._last = time.monotonic()

    def _raw(self, url: str):
        req = urllib.request.Request(url, headers=self._headers)
        last = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                return self._open(req, timeout=self.timeout)
            except urllib.error.HTTPError as e:
                last = e
                if e.code in (429, 500, 502, 503, 504):
                    ra = e.headers.get("Retry-After") if e.headers else None
                    wait = float(ra) if ra and str(ra).isdigit() else 2 ** attempt
                    time.sleep(wait + random.uniform(0, 0.4))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last = e
                time.sleep(2 ** attempt + random.uniform(0, 0.4))
        raise RateLimited(f"gave up on {url} after {self.max_retries} attempts: {last}")

    def get_json(self, url: str):
        with self._raw(url) as r:
            return json.loads(r.read())

    def get_info(self, repo: str) -> dict | None:
        try:
            return self.get_json(RESOLVE.format(repo=repo))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404):
                return None
            raise
        except (RateLimited, json.JSONDecodeError):
            return None

    def iter_datasets(self, filter_tag: str = "LeRobot", page_size: int = 1000,
                      max_pages: int | None = None) -> Iterator[dict]:
        url = f"{API}/datasets?filter={filter_tag}&limit={page_size}"
        pages = 0
        while url and (max_pages is None or pages < max_pages):
            with self._raw(url) as r:
                body = json.loads(r.read())
                link = r.headers.get("Link") if r.headers else None
            yield from body
            pages += 1
            url = None
            if link and 'rel="next"' in link:
                url = link.split(";")[0].strip("<> ")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_hubclient.py -q`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add ledger/hubclient.py tests/test_hubclient.py
git commit -m "Add token aware Hub client with backoff and cursor pagination"
```

---

### Task 7: SampleSource implementations

**Files:**
- Create: `ledger/sources.py`
- Create: `tests/test_sources.py`

**Interfaces:**
- Consumes: `ledger.hubclient.HubClient`, `ledger.paths.derive_paths`, `ledger.synth`.
- Produces: `AUDIT_COLUMNS: tuple[str, ...]`. `SampleSource` protocol with `info(repo)`, `parquet_paths(repo, info, limit)`, `frames(repo, path)`. Implementations `LocalSource(root)`, `SyntheticSource(defect="", **synth_kwargs)`, `StreamingSource(client=None)`, `DownloadSource(client=None, keep=False)`. `read_projected(file_obj_or_path) -> DataFrame` reading only `AUDIT_COLUMNS` that exist.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sources.py
import pytest
from ledger.sources import LocalSource, SyntheticSource, AUDIT_COLUMNS
from ledger.synth import make_episodes, write_v20_fixture, write_v30_fixture


@pytest.fixture
def v20(tmp_path):
    write_v20_fixture(make_episodes(n_eps=3, T=40, seed=1), tmp_path, "acme/v20")
    return tmp_path


@pytest.fixture
def v30(tmp_path):
    write_v30_fixture(make_episodes(n_eps=3, T=40, seed=1), tmp_path, "acme/v30")
    return tmp_path


def test_local_reads_v20(v20):
    s = LocalSource(v20)
    info = s.info("acme/v20")
    assert info["codebase_version"] == "v2.0"
    paths = s.parquet_paths("acme/v20", info, limit=None)
    assert len(paths) == 3
    df = s.frames("acme/v20", paths[0])
    assert set(df.columns) <= set(AUDIT_COLUMNS)
    assert len(df) == 40


def test_local_reads_v30(v30):
    s = LocalSource(v30)
    info = s.info("acme/v30")
    df = s.frames("acme/v30", s.parquet_paths("acme/v30", info, 1)[0])
    assert len(df) == 120  # 3 episodes packed


def test_local_missing_repo_returns_none(tmp_path):
    assert LocalSource(tmp_path).info("acme/nope") is None


def test_synthetic_source_is_offline():
    s = SyntheticSource(defect="stuck")
    info = s.info("synthetic/stuck")
    df = s.frames("synthetic/stuck", s.parquet_paths("synthetic/stuck", info, 1)[0])
    assert len(df) > 0


def test_projection_drops_non_audit_columns(tmp_path):
    import pandas as pd
    df = make_episodes(n_eps=1, T=20, seed=2)
    df["observation.images.top"] = [b"x" * 1024] * len(df)
    p = tmp_path / "acme/v30/data/chunk-000/file-000.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    (tmp_path / "acme/v30/meta").mkdir(parents=True, exist_ok=True)
    (tmp_path / "acme/v30/meta/info.json").write_text(
        '{"codebase_version":"v3.0","fps":30,"chunks_size":1000,"total_episodes":1,'
        '"data_path":"data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}')
    out = LocalSource(tmp_path).frames("acme/v30", "data/chunk-000/file-000.parquet")
    assert "observation.images.top" not in out.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_sources.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.sources'`

- [ ] **Step 3: Write minimal implementation**

```python
# ledger/sources.py
"""Where audit frames come from.

Four implementations behind one protocol, so the checks never learn
whether they are reading the Hub, a fixture, or a generator. Measured on
2026-09-05: the five audit columns are 2.7 percent of compressed bytes,
so projecting is worth roughly 37x in transfer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow.parquet as pq

from .hubclient import HubClient
from .paths import derive_paths
from . import synth

AUDIT_COLUMNS = ("timestamp", "frame_index", "episode_index",
                 "action", "observation.state")


def read_projected(handle) -> pd.DataFrame:
    pf = pq.ParquetFile(handle)
    want = [c for c in AUDIT_COLUMNS if c in pf.schema_arrow.names]
    return pf.read(columns=want).to_pandas()


class SampleSource(Protocol):
    def info(self, repo: str) -> dict | None: ...
    def parquet_paths(self, repo: str, info: dict, limit: int | None) -> list[str]: ...
    def frames(self, repo: str, path: str) -> pd.DataFrame: ...


class LocalSource:
    """Fixtures on disk. The backbone of the offline test suite."""

    def __init__(self, root):
        self.root = Path(root)

    def info(self, repo: str) -> dict | None:
        p = self.root / repo / "meta/info.json"
        return json.loads(p.read_text()) if p.exists() else None

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        found = sorted(str(p.relative_to(self.root / repo).as_posix())
                       for p in (self.root / repo).glob("data/**/*.parquet"))
        return found[:limit] if limit else found

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        return read_projected(self.root / repo / path)


class SyntheticSource:
    """Generated frames. No disk, no network."""

    def __init__(self, defect: str = "", fps: float = 30.0, **kw):
        self.defect, self.fps, self.kw = defect, fps, kw

    def info(self, repo: str) -> dict:
        return {"codebase_version": "demo", "fps": self.fps, "chunks_size": 1000,
                "total_episodes": self.kw.get("n_eps", 6),
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"}

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        return ["synthetic://0"]

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        return synth.make_episodes(fps=self.fps, defect=self.defect, **self.kw)


class StreamingSource:
    """HTTP range reads with column projection. The census default."""

    def __init__(self, client: HubClient | None = None):
        self.client = client or HubClient()
        from huggingface_hub import HfFileSystem
        self._fs = HfFileSystem()

    def info(self, repo: str) -> dict | None:
        return self.client.get_info(repo)

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        return derive_paths(info, limit)

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        with self._fs.open(f"datasets/{repo}/{path}", "rb") as fh:
            return read_projected(fh)


class DownloadSource:
    """Whole file download then delete. Fallback when a range read fails."""

    def __init__(self, client: HubClient | None = None, keep: bool = False):
        self.client = client or HubClient()
        self.keep = keep

    def info(self, repo: str) -> dict | None:
        return self.client.get_info(repo)

    def parquet_paths(self, repo: str, info: dict, limit: int | None = None) -> list[str]:
        return derive_paths(info, limit)

    def frames(self, repo: str, path: str) -> pd.DataFrame:
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(repo, path, repo_type="dataset")
        try:
            return read_projected(local)
        finally:
            if not self.keep:
                try:
                    Path(local).unlink()
                except OSError:
                    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_sources.py -q`
Expected: PASS, 5 tests

- [ ] **Step 5: Commit**

```bash
git add ledger/sources.py tests/test_sources.py
git commit -m "Add SampleSource abstraction with streaming, download, local and synthetic backends"
```

---

### Task 8: Report with provisional findings

**Files:**
- Create: `ledger/report.py`
- Create: `tests/test_report.py`

**Interfaces:**
- Consumes: `ledger.checks`, `ledger.config.load_thresholds`.
- Produces: `DatasetReport` dataclass carrying every v0 field plus `confirmed: str = ""`, `source: str = ""`, `revision: str = ""`, `error: str = ""`. `audit_dataframe(df, fps) -> list[dict]` per episode. `summarise(repo, info, parts, thresholds=None) -> DatasetReport`. `write_csv(reports, path)`, `append_jsonl(report, path)`, `read_jsonl(path) -> list[dict]`. `PROVISIONAL_HEADER: str` written as the first line of every CSV.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_report.py
import numpy as np
from ledger.report import (DatasetReport, audit_dataframe, summarise,
                           write_csv, append_jsonl, read_jsonl, PROVISIONAL_HEADER)
from ledger.synth import make_episodes


def _rep(defect):
    df = make_episodes(defect=defect, seed=0)
    return summarise(f"synthetic/{defect or 'clean'}",
                     {"codebase_version": "demo", "fps": 30}, audit_dataframe(df, 30.0))


def test_clean_has_no_flags():
    assert _rep("").flags == ""


def test_each_defect_is_named():
    assert "action_equals_state" in _rep("identity").flags
    assert "bad_dt" in _rep("drops").flags
    assert "stuck_state" in _rep("stuck").flags
    assert "negative_lag" in _rep("swapped").flags
    assert "duplicate_episodes" in _rep("duplicate").flags


def test_confirmed_defaults_empty():
    assert _rep("identity").confirmed == ""


def test_csv_carries_the_provisional_header(tmp_path):
    p = tmp_path / "r.csv"
    write_csv([_rep("identity")], p)
    first = p.read_text().splitlines()[0]
    assert first.startswith("#")
    assert "provisional" in first.lower()
    assert "defective" not in p.read_text().lower()


def test_jsonl_round_trips(tmp_path):
    p = tmp_path / "l.jsonl"
    append_jsonl(_rep(""), p)
    append_jsonl(_rep("stuck"), p)
    rows = read_jsonl(p)
    assert len(rows) == 2
    assert rows[1]["flags"] == "stuck_state"


def test_error_report_is_recorded_not_raised():
    r = DatasetReport(repo="acme/gated", error="403 gated")
    assert r.flags == "" and r.error == "403 gated"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.report'`

- [ ] **Step 3: Write minimal implementation**

Port `DatasetReport`, `audit_frame` and `summarise` from `ledger/audit.py`
with the aggregation arithmetic unchanged, adding `confirmed`, `source`,
`revision` and `error`, and routing per episode measurement through
`ledger.checks`. The flag names must stay byte identical to v0:
`ts_nonmonotonic`, `bad_dt`, `frame_gaps`, `stuck_state`,
`action_equals_state`, `negative_lag`, `large_lag`, `duplicate_episodes`.

`PROVISIONAL_HEADER` is required by `CLAUDE.md` and reads:

```python
PROVISIONAL_HEADER = (
    "# LEDGER M-02 automated audit. Flags are provisional and describe "
    "measurements, not findings. No dataset is defective until a human has "
    "opened it and filled in the confirmed column."
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_report.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Verify the original audit tests still pass**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_audit.py -q`
Expected: PASS, 2 tests, unchanged

- [ ] **Step 6: Commit**

```bash
git add ledger/report.py tests/test_report.py
git commit -m "Add report module separating provisional flags from confirmed findings"
```

---

### Task 9: Two tier census with resume

**Files:**
- Create: `ledger/census.py`
- Create: `tests/test_census.py`

**Interfaces:**
- Consumes: every module above.
- Produces: `CensusConfig` dataclass with `out_dir`, `sample_size=800`, `seed=20260905`, `files_per_dataset=1`, `max_datasets=None`, `tier="both"`. `build_frame(client, max_datasets=None) -> list[str]` returning every repo id. `draw_sample(frame, size, seed) -> list[str]`, deterministic and order independent. `run_metadata_tier(repos, source, out_path, done=None) -> int`. `run_deep_tier(repos, source, cfg, out_path, done=None) -> int`. `load_done(path) -> set[str]` keyed by `repo@revision`. `prevalence(reports, flag) -> tuple[float, float, float]` returning rate and Wilson bounds from `cairo_protocol`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_census.py
import json
import pytest
from ledger.census import (CensusConfig, draw_sample, load_done,
                           run_deep_tier, prevalence)
from ledger.report import DatasetReport
from ledger.sources import LocalSource
from ledger.synth import make_episodes, write_v30_fixture


def test_sample_is_deterministic_and_order_independent():
    frame = [f"o/d{i}" for i in range(500)]
    a = draw_sample(frame, 50, seed=1)
    b = draw_sample(list(reversed(frame)), 50, seed=1)
    assert a == b
    assert len(set(a)) == 50


def test_sample_smaller_than_request_returns_all():
    assert len(draw_sample(["a", "b"], 50, seed=1)) == 2


def test_resume_skips_completed(tmp_path):
    p = tmp_path / "ledger.jsonl"
    p.write_text(json.dumps({"repo": "acme/x", "revision": "abc"}) + "\n")
    assert load_done(p) == {"acme/x@abc"}


def test_deep_tier_writes_one_record_per_dataset(tmp_path):
    root = tmp_path / "repos"
    for name in ["acme/a", "acme/b"]:
        write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, name)
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    n = run_deep_tier(["acme/a", "acme/b"], LocalSource(root), cfg, out)
    assert n == 2
    assert len(out.read_text().strip().splitlines()) == 2


def test_deep_tier_survives_a_broken_dataset(tmp_path):
    root = tmp_path / "repos"
    write_v30_fixture(make_episodes(n_eps=2, T=40, seed=1), root, "acme/ok")
    out = tmp_path / "deep.jsonl"
    cfg = CensusConfig(out_dir=tmp_path, files_per_dataset=1)
    n = run_deep_tier(["acme/missing", "acme/ok"], LocalSource(root), cfg, out)
    assert n == 2  # both recorded, one carrying an error
    rows = [json.loads(l) for l in out.read_text().strip().splitlines()]
    assert any(r["error"] for r in rows)
    assert any(not r["error"] for r in rows)


def test_prevalence_carries_a_confidence_interval():
    reps = [DatasetReport(repo=f"o/d{i}", flags="stuck_state" if i < 20 else "")
            for i in range(100)]
    rate, lo, hi = prevalence(reps, "stuck_state")
    assert rate == 0.2
    assert lo < 0.2 < hi
    assert hi - lo > 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_census.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'ledger.census'`

- [ ] **Step 3: Write minimal implementation**

`draw_sample` sorts the frame before seeding so the draw does not depend
on Hub ordering, which is what makes the sample reproducible and the
prevalence claim defensible. `prevalence` calls
`cairo_protocol.stats.wilson_interval` and must never return a bare rate.
`run_deep_tier` wraps every dataset in a try block that records the error
into the report rather than raising, so one bad repo cannot end the run,
and flushes each JSONL record immediately.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_census.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add ledger/census.py tests/test_census.py
git commit -m "Add two tier census runner with seeded sampling and resume"
```

---

### Task 10: CLI, golden regression and documentation

**Files:**
- Modify: `ledger/audit.py` (rewrite as a thin CLI)
- Create: `tests/test_cli.py`
- Create: `tests/golden/demo_report.csv`
- Modify: `README.md`
- Create: `Makefile.win` or add PowerShell equivalents to `README.md`

**Interfaces:**
- Consumes: every module above.
- Produces: `main(argv=None) -> int`. Preserved flags `--demo`, `--repos`, `--top`, `--files`, `--out`, `--max-mb`. New flags `--census {metadata,deep,both}`, `--sample-size`, `--seed`, `--resume`, `--out-dir`, `--source {stream,download,local}`, `--local-root`. `--files` accepts an integer or the literal `all`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import pytest
from ledger.audit import main, parse_files


def test_files_accepts_all():
    assert parse_files("all") is None
    assert parse_files("3") == 3


def test_files_rejects_nonsense():
    with pytest.raises(Exception):
        parse_files("banana")


def test_demo_still_runs(capsys):
    assert main(["--demo"]) == 0
    out = capsys.readouterr().out
    assert "action_equals_state" in out
    assert "stuck_state" in out


def test_demo_output_never_says_defective(capsys):
    main(["--demo"])
    assert "defective" not in capsys.readouterr().out.lower()


def test_no_args_errors():
    with pytest.raises(SystemExit):
        main([])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_cli.py -q`
Expected: FAIL, `ImportError: cannot import name 'parse_files'`

- [ ] **Step 3: Rewrite the CLI over the new modules**

`ledger/audit.py` keeps its module docstring and its public entry points
`audit_frame`, `summarise` and `_synthetic` as thin re exports, so
`tests/test_audit.py` continues to pass untouched. `parse_files` maps
`"all"` to `None` and any other value through `int`, raising
`argparse.ArgumentTypeError` on failure.

- [ ] **Step 4: Run the full suite**

Run: `./.venv/Scripts/python.exe -m pytest -q`
Expected: PASS, every test including the original 8

- [ ] **Step 5: Record the golden demo report**

Run: `./.venv/Scripts/python.exe -m ledger.audit --demo --out tests/golden/demo_report.csv`
Then add a test asserting the current demo reproduces that file byte for
byte, so any future threshold change surfaces as a reviewable diff.

- [ ] **Step 6: Verify the doctests still pass**

Run: `./.venv/Scripts/python.exe -m doctest cairo_protocol/stats.py`
Expected: silent, meaning all 17 pass

- [ ] **Step 7: Update the README**

Add a census section documenting: the two tiers, the token requirement
and how to set it, the eight measurements from the spec, the default
sample size and seed, the resume behaviour, and Windows PowerShell
equivalents for every Makefile target, since the kit's `Makefile`
requires GNU make which Windows does not ship.

- [ ] **Step 8: Commit**

```bash
git add ledger/audit.py tests/test_cli.py tests/golden README.md
git commit -m "Rewrite the audit CLI over the census modules and pin the demo output"
```

---

## Self-Review

**Spec coverage.** Section 3 tier 1 is Task 9 `run_metadata_tier`, tier 2
is Task 9 `run_deep_tier` plus Task 10 `--sample-size` and `--seed`.
Section 4.1 is Task 7. Section 4.2 is Task 2. Section 4.3 is Task 9
`load_done`. Section 4.4 is Task 6. Section 4.5 is Task 8
`PROVISIONAL_HEADER` and `confirmed`. Section 5 unit is Tasks 1, 2, 5,
6, 7, 8; property based is Task 4; integration is Task 7; golden file is
Task 10 step 5; live smoke is the network marker created in Task 1.
Section 6 is honoured because no task touches `cairo_protocol` or
`lerobot_ur`, and Task 8 pins the v0 flag names. Section 7 is verified
in Task 10 steps 4 and 6.

**Placeholder scan.** No TBD, no "add error handling", no "similar to
Task N". Task 8 step 3 and Task 9 step 3 describe a port of arithmetic
that already exists in `ledger/audit.py` and name the exact identifiers
and flag strings to preserve rather than restating fifty lines verbatim.

**Type consistency.** `make_episodes` is used with the same keyword names
in Tasks 1, 2, 4, 7, 8 and 9. `episode_stats(ep, fps)` and
`run_checks(stats)` keep one signature across Tasks 2 and 4.
`derive_paths(info, limit)` matches its callers in Task 7.
`summarise(repo, info, parts, thresholds=None)` in Task 8 is called with
three positional arguments in Task 8's own tests, which the default
permits. `DatasetReport(repo=..., flags=..., error=...)` is constructed
by keyword in Tasks 8 and 9.
