"""Simulator oracle and analytic true-parameter prediction (PILOT_SPEC §I1, §I2; PROTOCOL §2, §8).

Three distinct quantities (PILOT_SPEC §G3), never to be confused:
- simulator oracle: the clean MuJoCo query outcome at the true cube. It IS the scoring target,
  so the oracle just returns the stored value; it never recomputes or re-simulates.
- analytic true-parameter prediction: the ideal closed form at the true (m, mu).
- analytic discrepancy: simulator oracle minus analytic prediction (PROTOCOL §2 table).
"""

from __future__ import annotations

import numpy as np

from cs182_icl.physics import G, analytic_displacement
from cs182_icl.types import Action, Cube, Prediction, TestItem

__all__ = [
    "analytic_displacement",
    "simulator_oracle",
    "analytic_true_prediction",
    "analytic_discrepancy",
    "item_analytic_discrepancy",
]


def simulator_oracle(item: TestItem) -> Prediction:
    """Return the stored clean MuJoCo query displacement (the prediction ceiling)."""
    return Prediction(d_pred=float(item.d_query_clean), status="ok", reason="simulator oracle")


def analytic_true_prediction(cube: Cube, query: Action, g: float = G) -> Prediction:
    """Closed-form displacement at the true cube parameters (clipped below mu m g)."""
    d = analytic_displacement(cube.m, cube.mu, query.F, query.T, g=g)
    return Prediction(d_pred=float(d), status="ok", reason="analytic true-parameter")


def analytic_discrepancy(d_sim, m, mu, F, T, g: float = G):
    """Simulator minus analytic displacement, in meters (scalars or numpy arrays).

    Positive means MuJoCo slid farther than the ideal closed form.
    """
    diff = np.asarray(d_sim, dtype=np.float64) - analytic_displacement(m, mu, F, T, g=g)
    return float(diff) if np.ndim(diff) == 0 else diff


def item_analytic_discrepancy(item: TestItem, g: float = G) -> float:
    """analytic_discrepancy for one test item's query."""
    q = item.query
    return float(analytic_discrepancy(item.d_query_clean, item.cube.m, item.cube.mu, q.F, q.T, g=g))
