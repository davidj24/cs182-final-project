"""Distributions for cubes and actions (PILOT_SPEC C; configs/pilot/initial_e2e_2026_11_16.json).

Every function takes an explicit numpy Generator. Actions never depend on cube parameters.
"""

from __future__ import annotations

import numpy as np

M_RANGE = (0.5, 1.5)
M_HEAVY_RANGE = (1.5, 2.0)
MU_RANGE = (0.3, 0.6)
F_RANGE = (13.0, 26.0)
K_RANGE = (100, 200)  # integer steps, inclusive
FQ_RANGE = (17.0, 22.0)  # E2/E3 query force (PILOT_SPEC L)
NEAR = 3.0  # N: "at least 3 N from F_q"
E3_SEP = 6.0  # N: E3 base forces at least this far apart


def sample_cubes(rng: np.random.Generator, n: int, m_range=M_RANGE) -> tuple[np.ndarray, np.ndarray]:
    m = np.exp(rng.uniform(np.log(m_range[0]), np.log(m_range[1]), n))
    mu = rng.uniform(*MU_RANGE, n)
    return m, mu


def sample_forces(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.uniform(*F_RANGE, shape)


def sample_ks(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.integers(K_RANGE[0], K_RANGE[1] + 1, shape).astype(np.int64)


def distinct_ks(rng: np.random.Generator, n: int, exclude: int | None = None) -> list[int]:
    """n pairwise-different integer durations from the training range, none equal to `exclude`."""
    pool = [k for k in range(K_RANGE[0], K_RANGE[1] + 1) if k != exclude]
    return [int(k) for k in rng.choice(pool, size=n, replace=False)]


def _from_measure(u: float, intervals: list[tuple[float, float]]) -> float:
    """Map u in [0, total length) to a point of the union of intervals."""
    for lo, hi in intervals:
        if u < hi - lo:
            return lo + u
        u -= hi - lo
    return intervals[-1][1]


def stratified_forces(rng: np.random.Generator, n: int, intervals: list[tuple[float, float]]) -> list[float]:
    """n forces spread evenly over the union of intervals: one uniform draw inside the middle
    80% of each of n equal-measure strata (so forces are distinct and >= 0.2*L/n apart)."""
    total = sum(hi - lo for lo, hi in intervals)
    width = total / n
    return [_from_measure((j + 0.1 + 0.8 * rng.random()) * width, intervals) for j in range(n)]


def far_intervals(Fq: float) -> list[tuple[float, float]]:
    """[13, Fq-3] U [Fq+3, 26]: forces at least 3 N from the query force."""
    return [(F_RANGE[0], Fq - NEAR), (Fq + NEAR, F_RANGE[1])]
