"""Implementation check of the idealized 1D simulator against its declared law (PILOT_SPEC A4).

This checks that the code implements the ideal law; it is NOT independent validation of contact
physics. Nothing is tuned toward the analytic result; numbers are printed as measured.
Run: python scripts/sim_validate.py   (about 10 s)
"""

from __future__ import annotations

import time

import mujoco
import numpy as np

from cs182_icl.physics import G, analytic_displacement
from cs182_icl.sim.rollout import Simulator, action_for_dt
from cs182_icl.types import Action


def draw(rng, n, mlo, mhi):
    m = np.exp(rng.uniform(np.log(mlo), np.log(mhi), n))
    return m, rng.uniform(0.3, 0.6, n), rng.uniform(13, 26, n), rng.integers(100, 201, n)


def stats(name, x):
    x = np.asarray(x)
    print(f"  {name}: median {np.median(x):.3e}  p95 {np.percentile(x, 95):.3e}  max {x.max():.3e}")


def run_set(label, m, mu, F, k):
    sim = Simulator()
    d, rel, ab, acc_e, dec_e, to, secs = [], [], [], [], [], 0, []
    for mi, mui, Fi, ki in zip(m, mu, F, k):
        sim.set_cube(mi, mui)
        t0 = time.perf_counter()
        res, tr = sim.rollout(Action(float(Fi), int(ki)), record=True)
        secs.append(time.perf_counter() - t0)
        an = analytic_displacement(mi, mui, Fi, ki * 1e-3)
        to += res.timed_out
        ab.append(abs(res.d - an)); rel.append(abs(res.d - an) / an)
        v = tr["vel"]
        acc_e.append(abs(v[ki - 1] / (ki * 1e-3) - (Fi - mui * mi * G) / mi) / ((Fi - mui * mi * G) / mi))
        n = 10  # coast window right after release
        dec_e.append(abs((v[ki + n - 1] - v[ki - 1]) / (n * 1e-3) + mui * G) / (mui * G))
    print(f"{label}: n={len(m)}, timeouts={to}, seconds per rollout (incl. recording) mean {np.mean(secs):.4f}")
    stats("displacement relative error", rel)
    stats("displacement absolute error (m)", ab)
    stats("push-phase acceleration relative error", acc_e)
    stats("coast deceleration relative error", dec_e)


def main():
    rng = np.random.default_rng(20261116)
    run_set("training-domain pushes", *draw(rng, 300, 0.5, 1.5))
    run_set("heavy-OOD pushes (m in [1.5, 2.0])", *draw(rng, 50, 1.5, 2.0))

    m, mu, F, k = draw(rng, 30, 0.5, 1.5)
    k = np.minimum(k, 200)
    sim = Simulator(); r = []
    for mi, mui, Fi, ki in zip(m, mu, F, k):
        sim.set_cube(mi, mui)
        d1 = sim.rollout(Action(float(Fi), int(ki)))[0].d
        d2 = sim.rollout(Action(float(Fi), int(2 * ki)))[0].d
        r.append(d2 / d1)
    r = np.array(r)
    print(f"T^2 scaling (same cube, same F, T doubled; closed-form ratio is exactly 4): n={len(r)}, "
          f"ratio median {np.median(r):.5f} min {r.min():.5f} max {r.max():.5f}, max |ratio/4-1| {np.abs(r / 4 - 1).max():.2e}")

    m, mu, F, k = draw(rng, 50, 0.5, 1.5)
    s1, s2 = Simulator(0.001), Simulator(0.0005); ch, ch2 = [], []
    for mi, mui, Fi, ki in zip(m, mu, F, k):
        s1.set_cube(mi, mui); s2.set_cube(mi, mui)
        a = Action(float(Fi), int(ki))
        d1 = s1.rollout(a)[0].d
        d2 = s2.rollout(action_for_dt(a, 0.0005))[0].d
        an = analytic_displacement(mi, mui, Fi, ki * 1e-3)
        ch.append(abs(d2 - d1) / d1); ch2.append(abs(d2 - an) / an)
    print("dt=0.0005 subset (n=50):")
    stats("|d(0.0005) - d(0.001)| / d(0.001)", ch)
    stats("|d(0.0005) - analytic| / analytic", ch2)

    # below-threshold stiction and zero-force creep
    sim = Simulator(); dmax = 0.0
    for _ in range(50):
        mi, mui = rng.uniform(0.5, 1.5), rng.uniform(0.3, 0.6)
        sim.set_cube(mi, mui)
        dmax = max(dmax, abs(sim.rollout(Action(float(mui * mi * G * rng.uniform(0.0, 1.0)), 150))[0].d))
    print(f"below-threshold pushes (F = U(0,1) * mu m g, 50 cubes): max |d| = {dmax:.3e} m")

    # evidence for the design choice: MuJoCo dof frictionloss (soft constraint) below threshold
    sim = Simulator(); sim.set_cube(1.0, 0.45)
    mdl, d = sim.model, sim.data
    mdl.dof_frictionloss[0] = 0.45 * 1.0 * G
    out = []
    for F in (4.0, 4.3, 4.4):
        mujoco.mj_resetData(mdl, d)
        for i in range(150):
            d.qfrc_applied[0] = F
            mujoco.mj_step(mdl, d)
        d.qfrc_applied[0] = 0.0
        for i in range(2000):
            mujoco.mj_step(mdl, d)
        out.append((F, float(d.qpos[0])))
    print("MuJoCo frictionloss (default solref/solimp), m=1 mu=0.45, threshold 4.4145 N: (F, d after 2.15 s) =", out)


if __name__ == "__main__":
    main()
