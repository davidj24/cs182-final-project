"""MuJoCo idealized 1D pusher and rollout (owner: T01), PROTOCOL 4 / PILOT_SPEC A4.

Usage:
    simulate_push(cube, Action(F, k))                      # dt = 0.001
    simulate_push(cube, action_for_dt(action, 0.0005), dt=0.0005)   # dt-halving rerun

dt convention: Action.T == k * 0.001 only at dt = 0.001. At another dt the CALLER passes the step
count it wants (use `action_for_dt`, which keeps the physical duration k_orig * 0.001 s); the force
is applied for exactly action.k steps of size dt. Analytic comparisons must then use
T = k_original * DT explicitly, never Action.T of the rescaled action. The stopping dwell is 50 steps
at dt = 0.001 and scales to keep 0.05 s (100 steps at dt = 0.0005).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from cs182_icl.types import DT, Action, Cube, PushResult

SIDE = 0.05  # full cube side s, m (half-size 0.025)
HALF = SIDE / 2
INERTIA_COEF = SIDE**2 / 6  # inertia = m * s^2 / 6 about every axis (irrelevant for 1D, kept for the model)
CAP_S = 15.0
LIN_TOL = 1e-4  # m/s
DWELL_S = 0.05  # 50 steps at dt = 0.001
G = 9.81  # same g as cs182_icl.physics
ASSET = Path(__file__).resolve().parents[3] / "assets" / "cube.xml"


def action_for_dt(action: Action, dt: float) -> Action:
    """Same physical push (F, k*DT seconds) expressed as steps of size dt."""
    return Action(action.F, int(round(action.k * DT / dt)))


class Simulator:
    """IDEALIZED 1D translational pusher in MuJoCo (PILOT_SPEC A4).

    One body on an x slide joint; the push force and Coulomb friction mu*m*g (assumes N = m*g) are
    applied through qfrc_applied by `_friction_step` before every mj_step. Floor/box geoms are visual
    only (contype = conaffinity = 0). This implements the declared ideal law; it is NOT an
    independent validation of contact physics. max_rotation_deg is vacuous (always 0.0).

    Why explicit friction and not MuJoCo dof frictionloss: frictionloss is a soft constraint and
    was tried first (default solref/solimp). It creeps below threshold: with m=1, mu=0.45
    (threshold 4.41 N) a push of F=4.0 N for 150 steps moved the cube 0.63 mm and peak speed
    4.2 mm/s (F=4.3: 0.68 mm), so d(below threshold) != 0, violating stiction. The explicit law
    below is exact in the sticking branch.

    Sticking rule (per step, dt): v_free = v + dt*F_push/m. If |v_free| <= dt*mu*g (the friction
    impulse can cancel all momentum) the cube sticks: qvel is set to 0 and net force is 0
    (qfrc = F_push - F_push). Otherwise net force = F_push - mu*m*g*sign(v_free).
    """

    def __init__(self, dt: float = DT):
        self.dt = float(dt)
        self.model = mujoco.MjModel.from_xml_path(str(ASSET))
        self.model.opt.timestep = self.dt
        self.data = mujoco.MjData(self.model)
        self.body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.dwell = max(1, int(round(DWELL_S / self.dt)))
        self.cap_steps = int(round(CAP_S / self.dt))
        self.cube_key: tuple[float, float] | None = None
        self.settled: dict | None = None
        self.settle_stats: dict | None = None
        self.mu = self.coulomb = 0.0

    # ---- cube setup -------------------------------------------------------------
    def set_cube(self, m: float, mu: float) -> None:
        key = (float(m), float(mu))
        if key == self.cube_key:
            return
        mdl = self.model
        mdl.body_mass[self.body] = m
        mdl.body_inertia[self.body] = m * INERTIA_COEF  # every axis
        self.mu, self.coulomb = float(mu), float(mu) * float(m) * G  # Coulomb force mu*m*g (N = m g)
        mujoco.mj_setConst(mdl, self.data)  # refresh derived quantities after mass/inertia change
        self._settle()
        self.cube_key = key

    def _settle(self) -> None:
        """Trivial: the initial state is exactly at rest (qvel = 0, nothing to settle)."""
        mdl, d = self.model, self.data
        mujoco.mj_resetData(mdl, d)
        mujoco.mj_forward(mdl, d)
        if np.any(d.qvel != 0.0) or d.qpos[0] != 0.0:
            raise RuntimeError("initial state is not exactly at rest")
        self.settled = dict(qpos=d.qpos.copy(), qvel=d.qvel.copy())
        self.settle_stats = dict(height=HALF, excursion=0.0, max_rot_deg=0.0, speed=0.0)

    def restore(self) -> None:
        s, d = self.settled, self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = s["qpos"]
        d.qvel[:] = s["qvel"]
        d.xfrc_applied[:] = 0.0
        d.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, d)

    def _friction_step(self, f_push: float) -> None:
        """Set qfrc_applied = push + Coulomb friction (with the stick rule) for the next mj_step."""
        d = self.data
        m = self.model.body_mass[self.body]
        v_free = d.qvel[0] + self.dt * f_push / m
        if abs(v_free) <= self.dt * self.mu * G:
            d.qvel[0] = 0.0  # stick: friction cancels the push and the remaining momentum
            d.qfrc_applied[0] = f_push - f_push  # exactly 0
        else:
            d.qfrc_applied[0] = f_push - np.sign(v_free) * self.coulomb

    # ---- rollout ----------------------------------------------------------------
    def rollout(self, action: Action, record: bool = False) -> tuple[PushResult, dict | None]:
        """Run one push from rest. With record=True also return the trajectory."""
        if self.settled is None:
            raise RuntimeError("call set_cube first")
        mdl, d = self.model, self.data
        self.restore()
        x0 = float(d.qpos[0])
        k = int(action.k)
        f_push = 0.0
        calm = 0
        timed_out, t_stop = True, CAP_S
        traj = {"t": [], "pos": [], "rot_deg": [], "force_on": [], "vel": []} if record else None
        for i in range(self.cap_steps):
            f_push = float(action.F) if i < k else 0.0  # push on for exactly k steps, then zero
            if record:
                traj["force_on"].append(i < k)
            self._friction_step(f_push)
            mujoco.mj_step(mdl, d)
            if record:
                traj["t"].append((i + 1) * self.dt)
                traj["pos"].append((d.qpos[0], 0.0, HALF))
                traj["vel"].append(float(d.qvel[0]))
                traj["rot_deg"].append(0.0)
            # the stopping rule is checked only once the push is over (force zero from step k)
            if i >= k and abs(d.qvel[0]) < LIN_TOL:
                calm += 1
                if calm >= self.dwell:
                    timed_out, t_stop = False, (i + 1) * self.dt
                    break
            else:
                calm = 0
        d.qfrc_applied[:] = 0.0
        res = PushResult(d=float(d.qpos[0] - x0), t_stop=float(t_stop),
                         max_rotation_deg=0.0, timed_out=bool(timed_out))
        if record:
            traj = {kk: np.asarray(v) for kk, v in traj.items()}
            traj["pos"] = traj["pos"].reshape(-1, 3)
        return res, traj


_SIMS: dict[float, Simulator] = {}


def _sim(dt: float) -> Simulator:
    if dt not in _SIMS:
        _SIMS[dt] = Simulator(dt)
    return _SIMS[dt]


def simulate_push(cube: Cube, action: Action, *, dt: float = DT) -> PushResult:
    """Clean deterministic MuJoCo rollout of one push (PROTOCOL 4).

    Restore the settled state for `cube`, apply (F, 0, 0) for exactly action.k steps of size dt
    (at dt != DT the caller passes the step count it wants), zero the force, step until the
    stopping rule fires or the cap is reached. Never drops flagged rollouts: the timeout flag is
    returned in the PushResult (max_rotation_deg is vacuous, 0.0, in this 1D scene).

    t_stop is the time (from push start, s) at which the stopping rule FIRES: the first step i >= k
    after which speed stayed below LIN_TOL (|v| < 1e-4 m/s) for 50 consecutive steps. So t_stop = (first calm step) + DWELL_S (0.05 s) (at the cap: CAP_S, timed_out).
    The first calm step is therefore t_stop - DWELL_S. d is the x displacement at the firing step.
    """
    s = _sim(dt)
    s.set_cube(cube.m, cube.mu)
    return s.rollout(action)[0]


# ---- batch helper -----------------------------------------------------------------
MAX_WORKERS = 16


def _work(args):
    dt, jobs = args
    return [(i, simulate_push(c, a, dt=dt)) for i, c, a in jobs]


def simulate_batch(items, *, workers: int = MAX_WORKERS, dt: float = DT, seed: int = 0,
                   verbose: bool = True) -> list[PushResult]:
    """Simulate a list of (Cube, Action); results in input order, flagged rollouts included.

    Jobs are grouped by (m, mu) so each cube settles once per worker; groups are shuffled with
    `seed` for load balance. Results do not depend on workers or seed (simulation is
    deterministic); the seed fixes the job layout so runs are reproducible bit-for-bit.
    Workers are capped at 16. The pool uses the fork start method: call this BEFORE any CUDA/torch
    initialisation in the process (forking after CUDA init is unsafe); otherwise run with workers=1.
    Prints seconds per rollout (wall clock and per-worker CPU).
    """
    import multiprocessing as mp

    items = list(items)
    workers = max(1, min(int(workers), MAX_WORKERS, mp.cpu_count()))
    groups: dict[tuple[float, float], list] = {}
    for i, (c, a) in enumerate(items):
        groups.setdefault((c.m, c.mu), []).append((i, c, a))
    order = np.random.default_rng(seed).permutation(len(groups))
    glist = list(groups.values())
    chunks = [(dt, glist[j]) for j in order]
    t0 = time.perf_counter()
    out: list[PushResult | None] = [None] * len(items)
    if workers == 1 or len(chunks) <= 1:
        parts = [_work(c) for c in chunks]
    else:
        with mp.get_context("fork" if "fork" in mp.get_all_start_methods() else None).Pool(workers) as pool:
            parts = pool.map(_work, chunks, chunksize=1)
    for part in parts:
        for i, r in part:
            out[i] = r
    wall = time.perf_counter() - t0
    if verbose and items:
        print(f"simulate_batch: {len(items)} rollouts, {workers} workers, {wall:.2f} s wall, "
              f"{wall / len(items):.4f} s/rollout wall, {wall * workers / len(items):.4f} s/rollout/worker")
    return out
