"""Hard-case subset for Bayes validation (analytic noise-free observations, optionally with a
tiny deterministic relative perturbation mimicking the measured MuJoCo discrepancy ~1e-6).

Pilot domain: F in [13, 26] N, k in 100..200. Clipped (d = 0) observations cannot occur:
max mu m g is 8.83 N (training prior) and 11.8 N for heavy OOD m <= 2.0, mu <= 0.6, all < 13 N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cs182_icl.physics import analytic_displacement
from cs182_icl.types import Action, Observation


@dataclass(frozen=True)
class HardCase:
    name: str
    prior: str
    m: float
    mu: float
    acts: tuple  # ((F, k), ...)
    query: tuple  # (F, k)
    perturb: float = 0.0  # relative perturbation scale of the observations

    def context(self) -> list[Observation]:
        rng = np.random.default_rng(12345)  # fixed seed: perturbations are deterministic
        out = []
        for i, (F, k) in enumerate(self.acts):
            d = float(analytic_displacement(self.m, self.mu, F, k * 1e-3))
            e = d * (1.0 + self.perturb * float(rng.standard_normal())) if self.perturb else d
            out.append(Observation(Action(float(F), int(k)), 1000 + i, d, e))
        return out

    def qaction(self) -> Action:
        return Action(float(self.query[0]), int(self.query[1]))

    def single_force(self) -> bool:
        return len({F for F, _ in self.acts}) == 1


def _rand_acts(seed, c):
    rng = np.random.default_rng(seed)
    return tuple((round(float(rng.uniform(13, 26)), 6), int(rng.integers(100, 201))) for _ in range(c))


HARD_CASES = [
    HardCase("light_slippery_c10", "train", 0.52, 0.31, _rand_acts(1, 10), (21.3, 150)),
    HardCase("light_slippery_c2", "train", 0.52, 0.31, ((14.0, 120), (25.0, 180)), (19.5, 160)),
    HardCase("light_slippery_c1", "train", 0.52, 0.31, ((17.0, 140),), (24.0, 190)),
    HardCase("heavy_grippy_c10", "train", 1.45, 0.58, _rand_acts(2, 10), (13.4, 110)),
    HardCase("heavy_grippy_c2_close", "train", 1.45, 0.58, ((13.0, 150), (13.8, 170)), (20.0, 120)),
    HardCase("heavy_grippy_c1_F13", "train", 1.45, 0.58, ((13.0, 200),), (13.0, 100)),
    HardCase("middle_c2_sameforce", "train", 0.9, 0.45, ((18.0, 120), (18.0, 190)), (22.0, 150)),
    HardCase("middle_c2_veryclose", "train", 0.9, 0.45, ((18.0, 160), (18.3, 140)), (25.5, 200)),
    HardCase("corner_lo_c2", "train", 0.5001, 0.3001, ((15.0, 100), (23.0, 200)), (18.0, 150)),
    HardCase("corner_hi_c10", "train", 1.4999, 0.5999, _rand_acts(3, 10), (16.0, 130)),
    HardCase("middle_c10_perturbed", "train", 0.8, 0.4, _rand_acts(4, 10), (20.0, 175), 1e-6),
    HardCase("middle_c1_perturbed", "train", 0.8, 0.4, ((20.0, 150),), (14.0, 100), 1e-6),
    HardCase("light_grippy_c2", "train", 0.55, 0.59, ((13.5, 110), (17.5, 200)), (26.0, 200)),
    HardCase("heavy_slippery_c1", "train", 1.4, 0.31, ((25.0, 190),), (13.2, 105)),
    HardCase("ood_heavy_train_c2", "train", 1.8, 0.5, ((16.0, 150), (22.0, 120)), (19.0, 180)),
    HardCase("ood_heavy_train_c10", "train", 2.0, 0.6, _rand_acts(5, 10), (14.0, 160)),
    HardCase("ood_heavy_train_c1", "train", 1.7, 0.35, ((19.0, 170),), (15.0, 130)),
    HardCase("ood_heavy_broad_c10", "broad", 1.8, 0.5, _rand_acts(6, 10), (17.0, 140)),
    HardCase("ood_heavy_broad_c2_nearthr", "broad", 2.0, 0.6, ((13.0, 200), (14.0, 180)), (13.0, 150)),
    HardCase("ood_heavy_broad_c1_F13", "broad", 1.6, 0.4, ((13.0, 160),), (21.0, 120)),
    HardCase("ood_heavy_broad_c2_sameforce", "broad", 1.9, 0.45, ((15.0, 100), (15.0, 200)), (24.0, 150)),
    HardCase("middle_c4", "train", 1.1, 0.5, _rand_acts(7, 4), (23.0, 120)),
    HardCase("k_extremes_c10", "train", 0.7, 0.35,
             tuple((13.0 + 1.3 * i, 100 if i % 2 else 200) for i in range(10)), (26.0, 200)),
    HardCase("light_slippery_c10_perturbed", "train", 0.5, 0.3, _rand_acts(8, 10), (25.0, 200), 2e-6),
]
