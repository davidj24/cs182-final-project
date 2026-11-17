"""Ideal closed-form push model (PROTOCOL §2). Float64 throughout.

With A = 1/(2 m^2 mu g) and B = 1/(2 m):
    d      = F T^2 max(A F - B, 0)
    t_stop = F T / (m mu g)            (moving pushes)
The cube moves only if F > mu m g. All functions accept scalars or numpy arrays.
These formulas are exact for the ideal model only; agreement with MuJoCo is measured, not assumed.
"""

from __future__ import annotations

import numpy as np

from cs182_icl.types import DT

G = 9.81  # gravitational acceleration, m/s^2 (also MuJoCo's default |gravity|)

__all__ = [
    "G",
    "DT",
    "duration",
    "line_coeffs",
    "motion_threshold",
    "analytic_displacement",
    "stop_time",
    "same_force_displacement",
    "line_value",
    "params_from_line",
]


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _out(x):
    """Return a Python float for 0-d results, else the float64 array."""
    x = np.asarray(x, dtype=np.float64)
    return float(x) if x.ndim == 0 else x


def duration(k, dt: float = DT):
    """T = k * dt in seconds."""
    return _out(_f64(k) * dt)


def line_coeffs(m, mu, g: float = G):
    """(A, B) of the line form d/(F T^2) = A F - B."""
    m, mu = _f64(m), _f64(mu)
    return _out(1.0 / (2.0 * m * m * mu * g)), _out(1.0 / (2.0 * m))


def motion_threshold(m, mu, g: float = G):
    """Static force threshold mu m g; the cube moves only if F > threshold."""
    return _out(_f64(mu) * _f64(m) * g)


def analytic_displacement(m, mu, F, T, g: float = G):
    """Closed-form displacement in meters, clipped to 0 when F <= mu m g.

    Uses the algebraically equal form F T^2 (F - mu m g) / (2 m^2 mu g), so the clip is exact
    at the threshold.
    """
    m, mu, F, T = _f64(m), _f64(mu), _f64(F), _f64(T)
    excess = F - mu * m * g
    d = F * T * T * excess / (2.0 * m * m * mu * g)
    return _out(np.where(excess > 0.0, d, 0.0))


def stop_time(m, mu, F, T, g: float = G):
    """Time from push start until the cube stops, F T/(m mu g); 0.0 if the cube never moves."""
    m, mu, F, T = _f64(m), _f64(mu), _f64(F), _f64(T)
    t = F * T / (m * mu * g)
    return _out(np.where(F > mu * m * g, t, 0.0))


def same_force_displacement(d1, T1, Tq):
    """Same-force rule: if F_q = F_1 then d_q = d_1 (T_q / T_1)^2."""
    d1, T1, Tq = _f64(d1), _f64(T1), _f64(Tq)
    return _out(d1 * (Tq / T1) ** 2)


def line_value(d, F, T):
    """y = d / (F T^2); equals A F - B for moving pushes."""
    d, F, T = _f64(d), _f64(F), _f64(T)
    return _out(d / (F * T * T))


def params_from_line(A, B, g: float = G):
    """Invert the line coefficients: m = 1/(2B), mu = 2 B^2 / (A g)."""
    A, B = _f64(A), _f64(B)
    return _out(1.0 / (2.0 * B)), _out(2.0 * B * B / (A * g))
