"""Priors for the Bayes reference (PROTOCOL §8; ranges frozen in DECISIONS / PILOT_SPEC §C).

Both priors are independent: m log-uniform on [m_lo, m_hi], mu uniform on [mu_lo, mu_hi].
In the coordinates (u = log m, mu) the density is CONSTANT on the box, so a uniform grid in
(u, mu) has constant cell weights (no extra 1/m factor; that factor belongs to a grid in m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["Prior", "TRAIN_PRIOR", "BROAD_PRIOR", "get_prior", "base_grid"]


@dataclass(frozen=True)
class Prior:
    name: str
    m_lo: float
    m_hi: float
    mu_lo: float
    mu_hi: float

    @property
    def u_lo(self) -> float:
        return math.log(self.m_lo)

    @property
    def u_hi(self) -> float:
        return math.log(self.m_hi)

    def log_density(self, m, mu):
        """log p(m, mu) w.r.t. dm dmu: -log(m ln(m_hi/m_lo)) - log(mu_hi - mu_lo); -inf outside."""
        m, mu = np.asarray(m, np.float64), np.asarray(mu, np.float64)
        inside = (m >= self.m_lo) & (m <= self.m_hi) & (mu >= self.mu_lo) & (mu <= self.mu_hi)
        val = -np.log(m * math.log(self.m_hi / self.m_lo)) - math.log(self.mu_hi - self.mu_lo)
        return np.where(inside, val, -np.inf)

    def log_density_logm(self, u, mu):
        """log p(u, mu) w.r.t. du dmu (u = log m): constant on the box, -inf outside."""
        u, mu = np.asarray(u, np.float64), np.asarray(mu, np.float64)
        inside = (u >= self.u_lo) & (u <= self.u_hi) & (mu >= self.mu_lo) & (mu <= self.mu_hi)
        val = -math.log(self.u_hi - self.u_lo) - math.log(self.mu_hi - self.mu_lo)
        return np.where(inside, val, -np.inf)


# Training ranges (DECISIONS: frozen for the pilot).
TRAIN_PRIOR = Prior("train", 0.5, 1.5, 0.3, 0.6)
# Broad prior, OOD only (PROTOCOL §8): log-uniform m on [0.35, 2.0] kg, uniform mu on [0.2, 0.7].
BROAD_PRIOR = Prior("broad", 0.35, 2.0, 0.2, 0.7)

_PRIORS = {"train": TRAIN_PRIOR, "broad": BROAD_PRIOR}


def get_prior(name: str) -> Prior:
    try:
        return _PRIORS[name]
    except KeyError:
        raise ValueError(f"unknown prior {name!r}; expected one of {sorted(_PRIORS)}") from None


def base_grid(prior: Prior, n: int):
    """Midpoint grid n x n uniform in (u, mu): (u_centers, mu_centers, log_cell_weight).

    The cell weight is constant (prior density times cell area) because the prior is uniform
    in (u, mu).
    """
    hu = (prior.u_hi - prior.u_lo) / n
    hmu = (prior.mu_hi - prior.mu_lo) / n
    u = prior.u_lo + hu * (np.arange(n) + 0.5)
    mu = prior.mu_lo + hmu * (np.arange(n) + 0.5)
    logw = math.log(hu * hmu) - math.log(prior.u_hi - prior.u_lo) - math.log(prior.mu_hi - prior.mu_lo)
    return u, mu, logw
