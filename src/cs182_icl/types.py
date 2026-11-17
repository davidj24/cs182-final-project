"""Core data types shared by every task (see docs/INTERFACES.md).

Units are SI: mass in kg, force in N, displacement in meters, time in seconds.
Push duration is stored as an integer number of simulator steps k; T = k * DT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DT = 0.001  # simulator timestep in seconds (PILOT_SPEC §C, PROTOCOL §3)


@dataclass(frozen=True)
class Cube:
    m: float  # mass, kg
    mu: float  # sliding friction coefficient
    cube_id: int  # unique within a dataset version; disjoint across splits


@dataclass(frozen=True)
class Action:
    F: float  # horizontal force, N
    k: int  # number of steps the force is applied

    @property
    def T(self) -> float:
        """Push duration in seconds, T = k * DT."""
        return self.k * DT


@dataclass(frozen=True)
class Observation:
    action: Action
    measurement_id: int  # see docs/INTERFACES.md "measurement_id" (PROTOCOL §5)
    d_clean: float  # clean simulated displacement, m
    d_obs: float  # observed displacement, m (equals d_clean when noise is 0)


@dataclass(frozen=True)
class PushResult:
    d: float  # x displacement from the settled start, m
    t_stop: float  # time from push start until the stopping rule fires, s
    max_rotation_deg: float  # largest orientation change over the trajectory, degrees
    timed_out: bool  # True if the stopping rule never fired before the cap


@dataclass(frozen=True)
class TestItem:
    __test__ = False  # stop pytest from collecting this class as a test

    cube: Cube
    query: Action
    d_query_clean: float  # scoring target: clean MuJoCo query displacement, m
    context: list[Observation] = field(default_factory=list)
    condition: str = ""  # e.g. "E1", "E2/A", "E3/k=2", "E4/heavy"


Status = Literal["ok", "not_determined", "blocked"]


@dataclass(frozen=True)
class Prediction:
    """Result of a reference estimator.

    status "ok": d_pred is a float (meters).
    status "not_determined": the context does not identify the query (e.g. physics fit
        with fewer than two distinct forces and no same-force observation); d_pred is None.
    status "blocked": the estimator is not available (e.g. Bayes when PREAPPROVAL R1 fails);
        d_pred is None and reason says why.
    """

    d_pred: float | None
    status: Status
    reason: str = ""
