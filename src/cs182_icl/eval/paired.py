"""Paired comparisons over shared items (same item_id and same c / condition keys)."""

from __future__ import annotations

import numpy as np

from cs182_icl.eval.metrics import Scores


def paired_sq_err_diff(a: Scores, b: Scores) -> np.ndarray:
    """Per-item squared-error difference a - b over items valid (status ok) for both."""
    if a.keys != b.keys:
        raise AssertionError("paired comparison requires identical item keys")
    ok = a.ok & b.ok
    return (a.sq_err - b.sq_err)[ok]


def summarize_diff(diff: np.ndarray, *, seed: int, n_boot: int = 2000) -> dict:
    """mean, paired SE (std ddof=1 / sqrt n) and a 95% bootstrap CI over items."""
    n = len(diff)
    if n == 0:
        return {"n": 0, "mean_diff": float("nan"), "se": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    rng = np.random.default_rng(seed)
    boots = diff[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return {"n": n, "mean_diff": float(diff.mean()), "se": se, "ci_lo": float(lo), "ci_hi": float(hi)}


def paired_across_keys(
    a: Scores, b: Scores, key_a, key_b, *, seed: int
) -> dict:
    """Paired difference of squared error between score sets whose cases differ in key but
    share item_id (e.g. same-force cell vs distinct-force cell at the same c, or c vs c=0).

    key_a / key_b: boolean masks over a / b selecting the cases to compare. Pairs are matched on
    item_id and must be one-to-one.
    """
    ia, ib = a.item_id[key_a], b.item_id[key_b]
    common = np.intersect1d(ia, ib)
    if len(np.unique(ia)) != len(ia) or len(np.unique(ib)) != len(ib):
        raise AssertionError("duplicate item_id inside a paired selection")
    ea = {i: (e, o) for i, e, o in zip(ia, a.sq_err[key_a], a.ok[key_a])}
    eb = {i: (e, o) for i, e, o in zip(ib, b.sq_err[key_b], b.ok[key_b])}
    diff = np.array([ea[i][0] - eb[i][0] for i in common if ea[i][1] and eb[i][1]])
    out = summarize_diff(diff, seed=seed)
    out["n_pairs_total"] = len(common)
    return out
