"""Threshold loading and validation. Config, not code."""

from __future__ import annotations

import tomllib
from pathlib import Path

_REQUIRED = (
    "frac_bad_dt",
    "stuck_state_frac",
    "identity_frac",
    "lag_large",
    "dup_episode_frac",
    "min_lag_correlation",
)
_FRACTIONS = (
    "frac_bad_dt",
    "stuck_state_frac",
    "identity_frac",
    "dup_episode_frac",
    "min_lag_correlation",
)
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
