"""physics_fit_predict signature (owner: T03 replaces the body; keep the signature)."""

from __future__ import annotations

from collections.abc import Sequence

from cs182_icl.types import Action, Observation, Prediction


def physics_fit_predict(context: Sequence[Observation], query: Action) -> Prediction:
    """Analytical physics fit (PROTOCOL §8, PILOT_SPEC §I3).

    Noise-free: least squares of d on (F^2 T^2, -F T^2) with >= 2 distinct forces, or the
    same-force rule if query.F was observed. Otherwise return
    Prediction(None, "not_determined", reason) -- never a pseudoinverse guess.
    """
    raise NotImplementedError("T03: physics_fit_predict")
