"""MuJoCo scene and rollout (owner: T01), PROTOCOL 4.

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

SIDE = 0.05  # full cube side s, m
HALF = SIDE / 2
INERTIA_COEF = SIDE**2 / 6  # inertia = m * s^2 / 6 about every axis
SETTLE_S = 0.5
CAP_S = 15.0
LIN_TOL = 1e-4  # m/s
ANG_TOL = 1e-3  # rad/s
DWELL_S = 0.05  # 50 steps at dt = 0.001
ASSET = Path(__file__).resolve().parents[3] / "assets" / "cube.xml"


def action_for_dt(action: Action, dt: float) -> Action:
    """Same physical push (F, k*DT seconds) expressed as steps of size dt."""
    return Action(action.F, int(round(action.k * DT / dt)))


def _rot_deg(q0_inv: np.ndarray, q: np.ndarray, tmp: np.ndarray) -> float:
    """Angle (deg) of the rotation taking the settled orientation to q (all axes)."""
    mujoco.mju_mulQuat(tmp, q0_inv, q)
    return float(np.degrees(2.0 * np.arctan2(np.linalg.norm(tmp[1:]), abs(tmp[0]))))


class Simulator:
    """One MuJoCo model; set_cube re-settles only when (m, mu) changes."""

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
        self._tmp = np.zeros(4)

    # ---- cube setup -------------------------------------------------------------
    def set_cube(self, m: float, mu: float) -> None:
        key = (float(m), float(mu))
        if key == self.cube_key:
            return
        mdl = self.model
        mdl.body_mass[self.body] = m
        mdl.body_inertia[self.body] = m * INERTIA_COEF  # every axis
        mdl.geom_friction[self.geom, 0] = mu  # sliding; floor friction is 0 so max() gives mu
        mujoco.mj_setConst(mdl, self.data)  # refresh derived quantities after mass/inertia change
        self._settle()
        self.cube_key = key

    def _settle(self) -> None:
        mdl, d = self.model, self.data
        mujoco.mj_resetData(mdl, d)
        d.qpos[:3] = (0.0, 0.0, HALF)
        mujoco.mj_forward(mdl, d)
        q0 = d.qpos[3:7].copy()
        q0_inv = np.zeros(4)
        mujoco.mju_negQuat(q0_inv, q0)
        zs, rots = [], []
        for _ in range(int(round(SETTLE_S / self.dt))):
            mujoco.mj_step(mdl, d)
            zs.append(d.qpos[2])
            rots.append(_rot_deg(q0_inv, d.qpos[3:7], self._tmp))
        zs = np.array(zs)
        speed = float(np.linalg.norm(d.qvel))
        if abs(d.qpos[2] - HALF) > 1e-4 or np.ptp(zs) > 1e-3 or max(rots) > 1.0 or speed > 1e-3:
            raise RuntimeError(
                f"cube did not settle: z={d.qpos[2]:.6f}, excursion={np.ptp(zs):.2e}, "
                f"rot={max(rots):.3f} deg, speed={speed:.2e}"
            )
        self.settled = dict(
            qpos=d.qpos.copy(), qvel=d.qvel.copy(), qacc_warmstart=d.qacc_warmstart.copy(),
            time=d.time, act=d.act.copy(),
        )
        self.settle_stats = dict(height=float(d.qpos[2]), excursion=float(np.ptp(zs)),
                                 max_rot_deg=float(max(rots)), speed=speed)

    def restore(self) -> None:
        s, d = self.settled, self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[:] = s["qpos"]
        d.qvel[:] = s["qvel"]
        d.qacc_warmstart[:] = s["qacc_warmstart"]
        d.time = s["time"]
        d.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, d)

    # ---- rollout ----------------------------------------------------------------
    def rollout(self, action: Action, record: bool = False) -> tuple[PushResult, dict | None]:
        """Run one push from the settled state. With record=True also return the trajectory."""
        if self.settled is None:
            raise RuntimeError("call set_cube first")
        mdl, d = self.model, self.data
        self.restore()
        x0 = float(d.qpos[0])
        q0_inv = np.zeros(4)
        mujoco.mju_negQuat(q0_inv, self.settled["qpos"][3:7])
        fx = d.xfrc_applied[self.body]
        k = int(action.k)
        max_rot = 0.0
        calm = 0
        timed_out, t_stop = True, CAP_S
        traj = {"t": [], "pos": [], "rot_deg": [], "force_on": []} if record else None
        for i in range(self.cap_steps):
            if i < k:
                fx[0] = action.F
            elif i == k:
                fx[:] = 0.0  # force removed after exactly k steps
            if record:
                traj["force_on"].append(float(fx[0]) != 0.0)
            mujoco.mj_step(mdl, d)
            rot = _rot_deg(q0_inv, d.qpos[3:7], self._tmp)
            max_rot = max(max_rot, rot)
            if record:
                traj["t"].append((i + 1) * self.dt)
                traj["pos"].append(d.qpos[:3].copy())
                traj["rot_deg"].append(rot)
            # the stopping rule is checked only once the push is over (force zero from step k)
            if i >= k and np.linalg.norm(d.qvel[:3]) < LIN_TOL and np.linalg.norm(d.qvel[3:6]) < ANG_TOL:
                calm += 1
                if calm >= self.dwell:
                    timed_out, t_stop = False, (i + 1) * self.dt
                    break
            else:
                calm = 0
        fx[:] = 0.0
        res = PushResult(d=float(d.qpos[0] - x0), t_stop=float(t_stop),
                         max_rotation_deg=float(max_rot), timed_out=bool(timed_out))
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
    stopping rule fires or the cap is reached. Never drops flagged rollouts: rotation and timeout
    flags are returned in the PushResult.

    t_stop is the time (from push start) at which the 50-step dwell completes, i.e. when the rule
    fires; it exceeds the physical stopping time by about DWELL_S.
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
    Workers are capped at 16. Prints seconds per rollout (wall clock and per-worker CPU).
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
        with mp.get_context("spawn").Pool(workers) as pool:
            parts = pool.map(_work, chunks, chunksize=1)
    for part in parts:
        for i, r in part:
            out[i] = r
    wall = time.perf_counter() - t0
    if verbose and items:
        print(f"simulate_batch: {len(items)} rollouts, {workers} workers, {wall:.2f} s wall, "
              f"{wall / len(items):.4f} s/rollout wall, {wall * workers / len(items):.4f} s/rollout/worker")
    return out
