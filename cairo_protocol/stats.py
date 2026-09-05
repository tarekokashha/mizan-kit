"""
cairo_protocol.stats
====================

The statistics under every MIZAN experiment. Small, dependency-light, and
written so that the first real-robot trial you ever run is already analysed
correctly.

What is in here
---------------
- wilson_interval        : binomial confidence interval for one success rate.
- barnard_test           : exact test for "policy A beats policy B" at fixed n
                           (TRI's 2026 recommendation for fixed-trial A/B).
- bootstrap_diff_ci      : bootstrap interval for the difference of two rates.
- anytime_cs             : anytime-valid confidence sequence for one success
                           rate (Robbins' beta-binomial mixture). Valid at
                           EVERY stopping time, so you may look after every
                           trial and stop early without inflating error.
- sequential_compare     : a conservative anytime-valid comparison of two
                           policies built from two confidence sequences at
                           alpha/2 each. Stops when the sequences separate.
- trials_needed          : rough fixed-n power calculation for planning.

Design notes
------------
The mixture confidence sequence is exact for Bernoulli data and needs no
tuning beyond the Beta prior (default Beta(1,1)). It is somewhat wider than
the near-optimal STEP procedure (Snyder et al., RSS 2025); if you need to
squeeze the last 20 percent of trials, implement STEP on top of this file.
The comparison rule here is valid by a union bound and is therefore honest
but conservative. That is the right default for a first paper.

Every function is pure and documented with a doctest so that `pytest` and
`python -m doctest` both exercise it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:  # scipy is optional for everything except barnard_test
    from scipy.stats import barnard_exact as _barnard_exact
    from scipy.special import betaln as _betaln
except Exception:  # pragma: no cover
    _barnard_exact = None
    _betaln = None


# --------------------------------------------------------------------------- #
# Fixed-n tools
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    >>> lo, hi = wilson_interval(18, 20)
    >>> round(lo, 3), round(hi, 3)
    (0.699, 0.972)
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = _z(alpha)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def barnard_test(succ_a: int, n_a: int, succ_b: int, n_b: int, alternative: str = "greater") -> float:
    """Barnard's exact test p-value for H1: rate_A > rate_B (default).

    Requires scipy. Use this for a fixed, pre-registered number of trials.

    >>> p = barnard_test(17, 20, 10, 20)
    >>> p < 0.05
    True
    """
    if _barnard_exact is None:  # pragma: no cover
        raise ImportError("scipy is required for barnard_test")
    table = np.array([[succ_a, n_a - succ_a], [succ_b, n_b - succ_b]])
    res = _barnard_exact(table, alternative=alternative)
    return float(res.pvalue)


def bootstrap_diff_ci(a: Sequence[int], b: Sequence[int], alpha: float = 0.05,
                      n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for mean(a) - mean(b) of two 0/1 outcome vectors.

    >>> rng = np.random.default_rng(1)
    >>> a = (rng.random(40) < 0.85).astype(int); b = (rng.random(40) < 0.55).astype(int)
    >>> lo, hi = bootstrap_diff_ci(a, b)
    >>> lo > 0
    True
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    return float(np.quantile(diffs, alpha / 2)), float(np.quantile(diffs, 1 - alpha / 2))


def trials_needed(p_a: float, p_b: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Approximate per-arm trials to detect p_a vs p_b (two-proportion z-test, one-sided).

    >>> trials_needed(0.9, 0.7)
    49
    >>> trials_needed(0.9, 0.85) > 300
    True
    """
    if not (0 < p_a < 1 and 0 < p_b < 1) or p_a == p_b:
        raise ValueError("rates must be in (0,1) and different")
    z_a = _z(2 * alpha)  # one-sided
    z_b = _z(2 * (1 - power))
    pbar = (p_a + p_b) / 2
    num = (z_a * math.sqrt(2 * pbar * (1 - pbar)) + z_b * math.sqrt(p_a * (1 - p_a) + p_b * (1 - p_b))) ** 2
    return int(math.ceil(num / (p_a - p_b) ** 2))


# --------------------------------------------------------------------------- #
# Anytime-valid tools
# --------------------------------------------------------------------------- #
def _log_mixture_martingale(s: int, t: int, p: np.ndarray, a: float, b: float) -> np.ndarray:
    """log of Robbins' beta-binomial mixture martingale evaluated at rate p."""
    if _betaln is not None:
        lb = _betaln(a + s, b + t - s) - _betaln(a, b)
    else:  # pragma: no cover
        lb = (math.lgamma(a + s) + math.lgamma(b + t - s) - math.lgamma(a + b + t)
              - (math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)))
    with np.errstate(divide="ignore"):
        return lb - (s * np.log(p) + (t - s) * np.log1p(-p))


def anytime_cs(outcomes: Iterable[int], alpha: float = 0.05, prior: tuple[float, float] = (1.0, 1.0),
               grid: int = 2001) -> tuple[float, float]:
    """Anytime-valid confidence sequence for a Bernoulli success rate.

    The interval returned after ANY number of trials contains the true rate
    with probability at least 1 - alpha, simultaneously over all times. You
    may therefore look after every trial and stop whenever you like.

    >>> lo, hi = anytime_cs([1]*17 + [0]*3)
    >>> 0.5 < lo < 0.85 and 0.9 < hi <= 1.0
    True
    >>> anytime_cs([]) == (0.0, 1.0)
    True
    """
    x = np.asarray(list(outcomes), dtype=int)
    t = int(len(x))
    if t == 0:
        return 0.0, 1.0
    s = int(x.sum())
    p = np.linspace(1e-6, 1 - 1e-6, grid)
    logm = _log_mixture_martingale(s, t, p, *prior)
    inside = logm < math.log(1 / alpha)
    if not inside.any():  # cannot happen for a valid CS, guard anyway
        return 0.0, 1.0
    idx = np.where(inside)[0]
    return float(p[idx[0]]), float(p[idx[-1]])


@dataclass
class ComparisonState:
    """Result of a sequential comparison after the trials seen so far."""
    n_a: int
    n_b: int
    rate_a: float
    rate_b: float
    cs_a: tuple[float, float]
    cs_b: tuple[float, float]
    decision: str  # "A>B", "B>A", or "continue"


def sequential_compare(outcomes_a: Iterable[int], outcomes_b: Iterable[int], alpha: float = 0.05) -> ComparisonState:
    """Conservative anytime-valid comparison of two policies.

    Builds a confidence sequence for each policy at alpha/2 and declares a
    winner only when the sequences separate. Valid at every stopping time by
    a union bound, so interleave trials, look after each pair, and stop when
    `decision != "continue"` or when your pre-registered trial budget is spent.

    >>> a = [1]*29 + [0]*1; b = [1]*8 + [0]*22
    >>> sequential_compare(a, b).decision
    'A>B'
    >>> sequential_compare([1,0,1,0], [1,0,1,0]).decision
    'continue'
    """
    a = list(outcomes_a)
    b = list(outcomes_b)
    cs_a = anytime_cs(a, alpha / 2)
    cs_b = anytime_cs(b, alpha / 2)
    if cs_a[0] > cs_b[1]:
        decision = "A>B"
    elif cs_b[0] > cs_a[1]:
        decision = "B>A"
    else:
        decision = "continue"
    return ComparisonState(
        n_a=len(a), n_b=len(b),
        rate_a=float(np.mean(a)) if a else float("nan"),
        rate_b=float(np.mean(b)) if b else float("nan"),
        cs_a=cs_a, cs_b=cs_b, decision=decision,
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _z(alpha: float) -> float:
    """Two-sided normal quantile without scipy (Acklam's approximation).

    >>> round(_z(0.05), 3)
    1.96
    """
    p = 1 - alpha / 2
    # Acklam's rational approximation
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= 1 - plow:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


if __name__ == "__main__":  # pragma: no cover
    import doctest
    print(doctest.testmod())
