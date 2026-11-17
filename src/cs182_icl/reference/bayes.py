"""bayes_predict interface (owner: T03). Bayes is OPTIONAL in the pilot.

This stub returns status "blocked" and never raises, so callers can always call it and must
handle the blocked case (PILOT_SPEC Precedence 3, §I4). T03 replaces the body only if
PREAPPROVAL R1 passes; otherwise it keeps returning "blocked" with the reason.
"""

from __future__ import annotations

from collections.abc import Sequence

from cs182_icl.types import Action, Observation, Prediction

BLOCKED_REASON = "Bayes not implemented: awaiting PREAPPROVAL R1 and H2 discrepancy parameters"


def bayes_predict(
    context: Sequence[Observation], query: Action, *, prior: str = "train"
) -> Prediction:
    """Posterior-mean query displacement in MuJoCo mode (PROTOCOL §8).

    prior: "train" (training ranges) or "broad" (OOD only). Likelihood deduplicates by
    measurement_id, groups by identical action, uses the discrepancy-aware variance.
    Returns Prediction(d_pred, "ok") or Prediction(None, "blocked", reason).
    """
    return Prediction(d_pred=None, status="blocked", reason=BLOCKED_REASON)
