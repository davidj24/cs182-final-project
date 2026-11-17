"""Closed-form physics tests (GATES.md H0, PILOT_SPEC §F). All fast."""

import numpy as np
import pytest

from cs182_icl import physics as ph
from cs182_icl.reference.bayes import bayes_predict
from cs182_icl.types import Action, Observation


def test_gravity_constant():
    assert ph.G == 9.81


def test_closed_form_light_slippery_long_push():
    # m=0.5, mu=0.3, F=26, T=0.2 -> 17.34 m (GATES.md H0)
    d = ph.analytic_displacement(0.5, 0.3, 26.0, 0.2)
    assert d == pytest.approx(17.34, abs=0.005)


def test_closed_form_heavy_grippy_short_push():
    # m=1.5, mu=0.6, F=13, T=0.1 -> 2.05 cm (GATES.md H0)
    d = ph.analytic_displacement(1.5, 0.6, 13.0, 0.1)
    assert d * 100 == pytest.approx(2.05, abs=0.005)


def test_closed_form_matches_A_B_definition():
    m, mu, F, T = 0.8, 0.45, 20.0, 0.15
    A, B = ph.line_coeffs(m, mu)
    assert A == pytest.approx(1 / (2 * m**2 * mu * ph.G), rel=1e-15)
    assert B == pytest.approx(1 / (2 * m), rel=1e-15)
    assert ph.analytic_displacement(m, mu, F, T) == pytest.approx(
        F * T**2 * (A * F - B), rel=1e-12
    )


def test_clipped_to_zero_at_and_below_threshold():
    m, mu = 1.5, 0.6
    thr = ph.motion_threshold(m, mu)
    assert thr == pytest.approx(mu * m * 9.81)
    assert ph.analytic_displacement(m, mu, thr, 0.2) == 0.0
    assert ph.analytic_displacement(m, mu, thr - 1.0, 0.2) == 0.0
    assert ph.analytic_displacement(m, mu, 1e-3, 0.2) == 0.0
    assert ph.analytic_displacement(m, mu, thr + 0.01, 0.2) > 0.0


def test_vectorized_float64():
    F = np.array([1.0, 13.0, 26.0])
    d = ph.analytic_displacement(1.5, 0.6, F, 0.1)
    assert isinstance(d, np.ndarray) and d.dtype == np.float64 and d.shape == (3,)
    assert d[0] == 0.0
    assert d[1] == pytest.approx(ph.analytic_displacement(1.5, 0.6, 13.0, 0.1))
    assert isinstance(ph.analytic_displacement(1.5, 0.6, 13.0, 0.1), float)


def test_same_force_duration_rule():
    m, mu, F = 0.7, 0.5, 18.0
    T1, Tq = ph.duration(120), ph.duration(190)
    d1 = ph.analytic_displacement(m, mu, F, T1)
    dq = ph.analytic_displacement(m, mu, F, Tq)
    assert ph.same_force_displacement(d1, T1, Tq) == pytest.approx(dq, rel=1e-12)
    assert ph.same_force_displacement(2.0, 0.1, 0.2) == pytest.approx(8.0)


def test_line_value_equals_AF_minus_B():
    m, mu = 1.1, 0.35
    A, B = ph.line_coeffs(m, mu)
    for F, T in [(13.0, 0.1), (19.5, 0.15), (26.0, 0.2)]:
        d = ph.analytic_displacement(m, mu, F, T)
        assert ph.line_value(d, F, T) == pytest.approx(A * F - B, rel=1e-12)


def test_params_from_line_recovers_m_mu_from_two_forces():
    m, mu = 0.9, 0.42
    (F1, T1), (F2, T2) = (14.0, 0.11), (24.0, 0.18)
    y1 = ph.line_value(ph.analytic_displacement(m, mu, F1, T1), F1, T1)
    y2 = ph.line_value(ph.analytic_displacement(m, mu, F2, T2), F2, T2)
    A = (y2 - y1) / (F2 - F1)
    B = A * F1 - y1
    m_hat, mu_hat = ph.params_from_line(A, B)
    assert m_hat == pytest.approx(m, rel=1e-10)
    assert mu_hat == pytest.approx(mu, rel=1e-10)


def test_params_from_line_inverts_line_coeffs():
    A, B = ph.line_coeffs(1.3, 0.55)
    m_hat, mu_hat = ph.params_from_line(A, B)
    assert m_hat == pytest.approx(1.3, rel=1e-14)
    assert mu_hat == pytest.approx(0.55, rel=1e-14)


def test_stop_time():
    m, mu, F, T = 0.5, 0.3, 26.0, 0.2
    assert ph.stop_time(m, mu, F, T) == pytest.approx(F * T / (m * mu * 9.81))
    assert ph.stop_time(m, mu, F, T) == pytest.approx(3.534, abs=1e-3)  # PROTOCOL §3: <= 3.5 s
    assert ph.stop_time(1.5, 0.6, 5.0, 0.2) == 0.0


def test_stop_time_consistent_with_constant_deceleration():
    # after the push, v0 = (F - mu m g) T / m decelerates at mu g, so t_stop = T + v0/(mu g)
    m, mu, F, T = 1.2, 0.4, 21.0, 0.17
    v0 = (F - mu * m * ph.G) * T / m
    assert ph.stop_time(m, mu, F, T) == pytest.approx(T + v0 / (mu * ph.G), rel=1e-12)


def test_duration_and_action_T():
    assert ph.duration(150) == pytest.approx(0.15)
    assert Action(F=20.0, k=150).T == pytest.approx(0.15)


def test_bayes_stub_returns_blocked_without_raising():
    obs = Observation(Action(20.0, 150), measurement_id=0, d_clean=1.0, d_obs=1.0)
    p = bayes_predict([obs], Action(18.0, 120))
    assert p.status == "blocked" and p.d_pred is None and p.reason
