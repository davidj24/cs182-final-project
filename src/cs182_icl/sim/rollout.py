"""simulate_push signature (owner: T01 replaces the body; keep the signature)."""

from __future__ import annotations

from cs182_icl.types import DT, Action, Cube, PushResult


def simulate_push(cube: Cube, action: Action, *, dt: float = DT) -> PushResult:
    """Clean deterministic MuJoCo rollout of one push (PROTOCOL §4).

    Restore the settled state for `cube`, apply (F, 0, 0) for exactly action.k steps of size dt
    (at dt != DT the caller passes the step count it wants), zero the force, step until the
    stopping rule fires or the cap is reached. Never drops flagged rollouts.
    """
    raise NotImplementedError("T01: simulate_push")
