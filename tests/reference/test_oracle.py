"""Simulator oracle and analytic discrepancy (PILOT_SPEC §I1, §I2). Fast."""

import numpy as np
import pytest

from cs182_icl import physics as ph
from cs182_icl.reference.oracle import (
    analytic_discrepancy,
    analytic_true_prediction,
    item_analytic_discrepancy,
    simulator_oracle,
)
from cs182_icl.types import Action, Cube, TestItem

CUBE = Cube(m=0.8, mu=0.45, cube_id=7)
QUERY = Action(F=20.0, k=150)
D_ANA = ph.analytic_displacement(0.8, 0.45, 20.0, 0.15)


def _item(d_sim):
    return TestItem(cube=CUBE, query=QUERY, d_query_clean=d_sim, context=[], condition="E1")


def test_oracle_returns_stored_target_not_analytic():
    d_sim = D_ANA * 1.037 + 1e-4  # deliberately different from the closed form
    p = simulator_oracle(_item(d_sim))
    assert p.status == "ok"
    assert p.d_pred == d_sim  # exact: stored value, never recomputed


def test_analytic_true_prediction_matches_physics():
    p = analytic_true_prediction(CUBE, QUERY)
    assert p.status == "ok"
    assert p.d_pred == D_ANA


def test_analytic_true_prediction_clipped_below_threshold():
    p = analytic_true_prediction(Cube(1.5, 0.6, 0), Action(F=8.0, k=100))  # 8 < 0.6*1.5*9.81
    assert p.d_pred == 0.0


def test_discrepancy_sign_is_simulator_minus_analytic():
    assert item_analytic_discrepancy(_item(D_ANA + 0.01)) == pytest.approx(0.01, abs=1e-12)
    assert item_analytic_discrepancy(_item(D_ANA - 0.02)) == pytest.approx(-0.02, abs=1e-12)
    assert item_analytic_discrepancy(_item(D_ANA)) == 0.0


def test_discrepancy_vectorized():
    m = np.array([0.5, 1.0, 1.5])
    mu = np.array([0.3, 0.45, 0.6])
    F = np.array([26.0, 20.0, 13.0])
    T = np.array([0.2, 0.15, 0.1])
    ana = ph.analytic_displacement(m, mu, F, T)
    out = analytic_discrepancy(ana * 1.1, m, mu, F, T)
    np.testing.assert_allclose(out, 0.1 * ana, rtol=1e-12)
