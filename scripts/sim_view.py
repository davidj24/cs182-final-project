"""Save replays + frames for four representative pushes (PILOT_SPEC G1).

Idealized 1D simulator (PILOT_SPEC A4): the box is visual only and moves with the x joint.
Output: $CS282A_DATA/pilot/sim_frames/<timestamp>/ with <case>.npz (trajectory replay),
<case>_traj.png (position / velocity vs time) and, if MuJoCo EGL rendering works,
<case>_frame_*.png. Then times ~20 rollouts. Run: python scripts/sim_view.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from cs182_icl.sim.rollout import Simulator
from cs182_icl.types import Action

CASES = {
    "light_slippery": (0.5, 0.3, 26.0, 200),
    "middle": (1.0, 0.45, 20.0, 150),
    "heavy_grippy": (1.5, 0.6, 26.0, 200),
    "light_grippy": (0.5, 0.6, 26.0, 200),
}


def main():
    out = Path(os.environ["CS282A_DATA"]) / "pilot" / "sim_frames" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    sim = Simulator()
    try:
        renderer = mujoco.Renderer(sim.model, 240, 360)
        render = "mujoco_egl"
    except Exception as e:  # headless GL unavailable
        renderer, render = None, f"matplotlib_only ({type(e).__name__}: {e})"
    print("rendering:", render)
    for name, (m, mu, F, k) in CASES.items():
        sim.set_cube(m, mu)
        res, tr = sim.rollout(Action(F, k), record=True)
        np.savez(out / f"{name}.npz", m=m, mu=mu, F=F, k=k, **tr)
        print(f"{name}: m={m} mu={mu} F={F} k={k} -> {res}")
        fig, ax = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        ax[0].plot(tr["t"], tr["pos"][:, 0]); ax[0].set_ylabel("x (m)")
        ax[1].plot(tr["t"], tr["vel"]); ax[1].set_ylabel("velocity (m/s)")
        for a in ax:
            a.axvline(k * sim.dt, color="r", ls="--", lw=0.8)
        ax[1].set_xlabel("time (s); red = force off")
        ax[0].set_title(f"{name}: m={m} mu={mu} F={F} k={k}, d={res.d:.3f} m (idealized 1D)")
        fig.tight_layout(); fig.savefig(out / f"{name}_traj.png", dpi=110); plt.close(fig)
        if renderer is not None:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = (0, 0, 0.02); cam.distance = 0.6; cam.azimuth = 90; cam.elevation = -20
            sim.restore()
            steps = len(tr["t"])
            for j, frac in enumerate((0.0, 0.1, 0.5, 1.0)):
                target = int(frac * (steps - 1))
                while len(tr["t"]) and sim.data.time < tr["t"][target] - 1e-12:
                    sim._friction_step(F if sim.data.time < k * sim.dt - 1e-12 else 0.0)
                    mujoco.mj_step(sim.model, sim.data)
                mujoco.mj_kinematics(sim.model, sim.data)
                cam.lookat[0] = sim.data.qpos[0]
                renderer.update_scene(sim.data, cam)
                plt.imsave(out / f"{name}_frame_{j}.png", renderer.render())
            sim.data.qfrc_applied[:] = 0.0
    # timing: 20 rollouts, one cube setup each pair
    t0 = time.perf_counter()
    rng = np.random.default_rng(0)
    n = 20
    for i in range(n):
        sim.set_cube(*[(0.5, 0.3), (1.0, 0.45), (1.5, 0.6), (0.8, 0.35)][i % 4])
        sim.rollout(Action(float(rng.uniform(13, 26)), int(rng.integers(100, 201))))
    dt_ = time.perf_counter() - t0
    print(f"timing: {n} rollouts (incl. settles) {dt_:.2f} s = {dt_ / n:.3f} s/rollout (single process)")
    print("saved to", out)


if __name__ == "__main__":
    main()
