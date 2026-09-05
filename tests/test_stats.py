import numpy as np
from cairo_protocol.stats import anytime_cs, sequential_compare, wilson_interval, trials_needed, bootstrap_diff_ci


def test_wilson_contains_point_estimate():
    lo, hi = wilson_interval(15, 20)
    assert lo < 0.75 < hi


def test_anytime_cs_coverage_under_optional_stopping():
    rng = np.random.default_rng(0)
    misses = 0
    runs, T, p = 150, 40, 0.7
    for _ in range(runs):
        x = (rng.random(T) < p).astype(int)
        if any(not (lo <= p <= hi) for lo, hi in (anytime_cs(x[:t], 0.05, grid=601) for t in range(1, T + 1))):
            misses += 1
    assert misses / runs <= 0.08  # nominal 0.05, slack for Monte Carlo noise


def test_sequential_compare_separates_clear_gap():
    a = [1] * 29 + [0]
    b = [1] * 8 + [0] * 22
    assert sequential_compare(a, b).decision == "A>B"
    assert sequential_compare(b, a).decision == "B>A"


def test_sequential_compare_waits_when_uncertain():
    assert sequential_compare([1, 0, 1], [0, 1, 0]).decision == "continue"


def test_trials_needed_monotone():
    assert trials_needed(0.9, 0.7) < trials_needed(0.9, 0.8) < trials_needed(0.9, 0.85)


def test_bootstrap_diff_sign():
    rng = np.random.default_rng(3)
    a = (rng.random(60) < 0.9).astype(int)
    b = (rng.random(60) < 0.5).astype(int)
    lo, hi = bootstrap_diff_ci(a, b)
    assert lo > 0
