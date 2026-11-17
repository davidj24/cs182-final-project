"""Analytical physics fit reference (PROTOCOL §8, PILOT_SPEC §I3). Float64 throughout.

Model for moving pushes: d = A F^2 T^2 - B F T^2 (line form d/(F T^2) = A F - B).
Observations here come from MuJoCo, not the ideal model, so (A, B) are FITTED by ordinary
least squares; no exact analytic inversion is applied to the observations.

Rules, in order:
1. Deduplicate by measurement_id (first occurrence kept). Copies therefore change nothing.
   A repeated ID with a different action or value is corrupt data and raises ValueError.
2. Same-force rule: if the query force was observed (equal up to float storage,
   SAME_FORCE_RTOL), predict h T_q^2 with h = sum d T^2 / sum T^4 over the context pushes at
   F_q. With one such push this is d_1 (T_q/T_1)^2 exactly. Clipped pushes at F_q are kept:
   d = 0 at F_q means the cube does not move at that force, so the rule predicts 0.
3. Line fit on MOVING pushes only (d_obs > move_eps). A clipped push only tells us
   F <= mu m g, an inequality; it carries no information about the line and is excluded.
   Determined iff >= MIN_MOVING moving pushes whose forces span >= force_sep_min newtons.
   Otherwise status "not_determined" with a reason -- never a pseudoinverse guess.
4. Predict the query with the clipped closed form F_q T_q^2 max(A F_q - B, 0).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from cs182_icl.types import Action, Observation, Prediction

FORCE_SEP_MIN = 3.0  # N; min force span among moving pushes (matches the E2 "new force" gap)
MIN_MOVING = 2  # min number of distinct-ID moving pushes for a line fit
MOVE_EPS = 1e-6  # m; d_obs <= MOVE_EPS counts as a clipped (non-moving) push
SAME_FORCE_RTOL = 1e-9  # float-storage tolerance for "query force observed"; not physical

__all__ = [
    "FORCE_SEP_MIN",
    "MIN_MOVING",
    "MOVE_EPS",
    "SAME_FORCE_RTOL",
    "dedupe_by_measurement_id",
    "fit_line",
    "physics_fit_predict",
]


def dedupe_by_measurement_id(context: Sequence[Observation]) -> list[Observation]:
    """Keep the first observation per measurement_id; raise if a repeated ID disagrees."""
    seen: dict[int, Observation] = {}
    out: list[Observation] = []
    for obs in context:
        prev = seen.get(obs.measurement_id)
        if prev is None:
            seen[obs.measurement_id] = obs
            out.append(obs)
        elif prev.action != obs.action or prev.d_obs != obs.d_obs:
            raise ValueError(
                f"measurement_id {obs.measurement_id} repeated with a different action or "
                "d_obs; a copy must keep identical action, outcome and ID"
            )
    return out


def _same_force(F: float, Fq: float) -> bool:
    return abs(F - Fq) <= SAME_FORCE_RTOL * max(1.0, abs(Fq))


def fit_line(
    context: Sequence[Observation],
    *,
    force_sep_min: float = FORCE_SEP_MIN,
    move_eps: float = MOVE_EPS,
    fit_space: str = "d",
) -> tuple[float, float] | str:
    """OLS estimate (A, B) from the moving pushes, or a reason string if not determined.

    fit_space "d" (default, PROTOCOL §8): regress d on columns (F^2 T^2, -F T^2).
    fit_space "line": regress y = d/(F T^2) on columns (F, -1). Both are exact on
    noise-free analytic data; they weight MuJoCo discrepancies differently.
    """
    obs = dedupe_by_measurement_id(context)
    if not obs:
        return "empty context"
    moving = [o for o in obs if o.d_obs > move_eps]
    if not moving:
        return "no moving pushes (all clipped; a clipped push gives no line information)"
    if len(moving) < MIN_MOVING:
        return f"only {len(moving)} moving push(es); need >= {MIN_MOVING}"
    F = np.array([o.action.F for o in moving], dtype=np.float64)
    T = np.array([o.action.T for o in moving], dtype=np.float64)
    d = np.array([o.d_obs for o in moving], dtype=np.float64)
    span = float(F.max() - F.min())
    if span < force_sep_min:
        return (
            f"moving-push forces span {span:.3g} N < {force_sep_min} N "
            "(one force level; line not determined)"
        )
    if fit_space == "d":
        X = np.column_stack([F * F * T * T, -F * T * T])
        y = d
    elif fit_space == "line":
        X = np.column_stack([F, -np.ones_like(F)])
        y = d / (F * T * T)
    else:
        raise ValueError(f"fit_space must be 'd' or 'line', got {fit_space!r}")
    coef, _, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    if rank < 2:  # cannot happen when span > 0; checked so we never return a min-norm guess
        return "design matrix rank < 2"
    return float(coef[0]), float(coef[1])


def physics_fit_predict(
    context: Sequence[Observation],
    query: Action,
    *,
    force_sep_min: float = FORCE_SEP_MIN,
    move_eps: float = MOVE_EPS,
    fit_space: str = "d",
) -> Prediction:
    """Analytical physics fit (PROTOCOL §8, PILOT_SPEC §I3); query displacement in meters.

    Same-force rule if query.F was observed; else least squares of d on (F^2 T^2, -F T^2)
    over moving pushes spanning >= force_sep_min N. Otherwise return
    Prediction(None, "not_determined", reason) -- never a pseudoinverse guess.
    """
    obs = dedupe_by_measurement_id(context)
    Fq, Tq = float(query.F), float(query.T)

    same = [o for o in obs if _same_force(o.action.F, Fq)]
    if same:
        if len(same) == 1:
            o = same[0]
            d = o.d_obs * (Tq / o.action.T) ** 2  # physics.same_force_displacement, scalar form
        else:
            dd = np.array([o.d_obs for o in same], dtype=np.float64)
            TT = np.array([o.action.T for o in same], dtype=np.float64)
            h = float(np.sum(dd * TT**2) / np.sum(TT**4))
            d = h * Tq * Tq
        return Prediction(float(d), "ok", f"same-force rule ({len(same)} push(es) at F_q)")

    line = fit_line(obs, force_sep_min=force_sep_min, move_eps=move_eps, fit_space=fit_space)
    if isinstance(line, str):
        return Prediction(None, "not_determined", line)
    A, B = line
    d = Fq * Tq * Tq * max(A * Fq - B, 0.0)
    reason = f"line fit ({fit_space}-space OLS): A={A:.6g}, B={B:.6g}"
    if A <= 0.0 or B <= 0.0:
        reason += " [unphysical: A or B <= 0]"
    return Prediction(float(d), "ok", reason)
