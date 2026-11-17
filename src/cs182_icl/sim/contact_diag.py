"""Contact-solver investigation for T01 repair (diagnosis + candidate table). Not used by data code.

Run: python -m cs182_icl.sim.contact_diag      (about 1-2 min, single process)
Tunes ONLY for flatness (no rotation, contact kept, no height excursion); d-vs-analytic is printed
afterwards as a measurement and was never a selection criterion.
"""

from __future__ import annotations

import mujoco
import numpy as np

from cs182_icl.physics import analytic_displacement
from cs182_icl.sim import rollout as R
from cs182_icl.types import Action

CORNERS = [(0.5, 0.6, 26, 200), (0.5, 0.3, 26, 200), (1.0, 0.45, 20, 150), (1.5, 0.6, 26, 200), (0.5, 0.6, 13, 100)]


def make(cfg: dict) -> R.Simulator:
    """Simulator with model knobs overridden by cfg (geom/option fields, or a contact pair)."""
    s = R.Simulator()
    if "pair_solreffriction" in cfg:  # contact pair carrying its own friction reference (not a geom knob)
        x = open(R.ASSET).read().replace(
            "</mujoco>",
            '<contact><pair geom1="floor" geom2="cube_geom" condim="3" friction="0.4 0.4 0.005 0.0001 0.0001" '
            f'solref="0.02 1" solreffriction="{cfg["pair_solreffriction"]}" solimp="0.99 0.999 0.001"/></contact></mujoco>')
        s.model = mujoco.MjModel.from_xml_string(x)
        s.model.opt.timestep = s.dt
        s.data = mujoco.MjData(s.model)
        orig = s.set_cube

        def set_cube(m, mu, _o=orig):
            s.model.pair_friction[0, :2] = mu
            s.cube_key = None
            _o(m, mu)
        s.set_cube = set_cube
    m = s.model
    for key, arr in (("solref", m.geom_solref), ("solimp", m.geom_solimp), ("margin", m.geom_margin), ("gap", m.geom_gap)):
        if key in cfg:
            arr[:] = cfg[key]
    for key in ("impratio", "iterations", "tolerance", "cone", "noslip_iterations", "noslip_tolerance", "solver"):
        if key in cfg:
            setattr(m.opt, key, cfg[key])
    if cfg.get("nowarm"):
        m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_WARMSTART
    return s


def run(s: R.Simulator, m, mu, F, k):
    """(max rotation deg, height excursion during push mm, min contacts during push, d, settled height)."""
    s.set_cube(m, mu)
    res, _ = s.rollout(Action(F, k))
    s.restore()
    fx = s.data.xfrc_applied[s.body]
    minc, zs = 99, []
    for i in range(k):  # contact count / height during the push (rollout() does not record them)
        fx[0] = F
        mujoco.mj_step(s.model, s.data)
        minc = min(minc, s.data.ncon)
        zs.append(s.data.qpos[2])
    fx[:] = 0
    return res.max_rotation_deg, (max(zs) - min(zs)) * 1e3, minc, res.d, s.settle_stats["height"]


def slide_test(s: R.Simulator, m=0.5, mu=0.6, v0=3.0, n=150):
    """Zero applied force, initial sliding speed v0: hop height (um) and min contacts."""
    s.set_cube(m, mu)
    s.restore()
    s.data.qvel[0] = v0
    zs, minc = [], 99
    for _ in range(n):
        mujoco.mj_step(s.model, s.data)
        zs.append(s.data.qpos[2])
        minc = min(minc, s.data.ncon)
    return (max(zs) - min(zs)) * 1e6, minc


def diagnose():
    s = R.Simulator()
    s.set_cube(0.5, 0.6)
    s.restore()
    d = s.data
    print("DIAGNOSIS m=0.5 mu=0.6 F=26 (push onset)")
    print("  settled ncon", d.ncon, "dist (m)", d.contact.dist[:d.ncon].round(8).tolist())
    print("  xipos == xpos (COM at body origin):", np.allclose(d.xipos[s.body], d.xpos[s.body]),
          " xfrc_applied torque after setting force:", end=" ")
    fx = d.xfrc_applied[s.body]
    fx[0] = 26.0
    print(fx[3:].tolist(), "(force is world-frame at the COM)")
    f6 = np.zeros(6)
    mujoco.mj_step(s.model, d)
    N, T = 0.0, 0.0
    for c in range(d.ncon):
        mujoco.mj_contactForce(s.model, d, c, f6)
        N += f6[0]
        T += np.hypot(f6[1], f6[2])
    print(f"  step 0: sum normal = {N:.2f} N (m g = {0.5 * 9.81:.2f} N), sum tangential = {T:.2f} N, "
          f"T/N = {T / max(N, 1e-9):.3f} (mu = 0.6), vz = {d.qvel[2]:.4f} m/s")
    print("  hand derivation: rigid cube, force at COM, friction mu*m*g at the base gives torque mu*m*g*h; the "
          "normal force can shift by at most s/2 (needs mu < s/(2h) = 1) so N = m g, pitch ~ 0.")
    fx[:] = 0


CANDIDATES = {
    "baseline (current cube.xml)": {},
    "solref 0.002 1": {"solref": [0.002, 1]},
    "solref 0.005 1": {"solref": [0.005, 1]},
    "solref 0.01 1": {"solref": [0.01, 1]},
    "solref 0.04 1": {"solref": [0.04, 1]},
    "solref 0.004 2": {"solref": [0.004, 2]},
    "solref 0.02 3": {"solref": [0.02, 3]},
    "solref 0.02 10": {"solref": [0.02, 10]},
    "solref -1e6 -1000 (direct)": {"solref": [-1e6, -1000]},
    "solref -1e6 -20 (direct)": {"solref": [-1e6, -20]},
    "solimp 0.9 0.99 0.01": {"solimp": [0.9, 0.99, 0.01, 0.5, 2]},
    "solimp 0.95 0.99 0.0001": {"solimp": [0.95, 0.99, 0.0001, 0.5, 2]},
    "solimp 0.99 0.999 0.01": {"solimp": [0.99, 0.999, 0.01, 0.5, 2]},
    "solimp 0.9999 0.9999 + solref 0.05": {"solimp": [0.9999, 0.9999, 0.001, 0.5, 2], "solref": [0.05, 1]},
    "solimp 0.9999 0.9999 + solref 0.2": {"solimp": [0.9999, 0.9999, 0.001, 0.5, 2], "solref": [0.2, 1]},
    "margin 0.001": {"margin": 0.001},
    "margin 0.001 gap 0.001": {"margin": 0.001, "gap": 0.001},
    "impratio 1": {"impratio": 1},
    "impratio 100": {"impratio": 100},
    "pyramidal cone": {"cone": mujoco.mjtCone.mjCONE_PYRAMIDAL},
    "pyramidal impratio 1 noslip 10": {"cone": mujoco.mjtCone.mjCONE_PYRAMIDAL, "impratio": 1, "noslip_iterations": 10},
    "noslip_iterations 20": {"noslip_iterations": 20},
    "iterations 1000 tol 1e-14": {"iterations": 1000, "tolerance": 1e-14},
    "PGS 500": {"solver": mujoco.mjtSolver.mjSOL_PGS, "iterations": 500},
    "CG 500 tol 1e-14": {"solver": mujoco.mjtSolver.mjSOL_CG, "iterations": 500, "tolerance": 1e-14},
    "no warmstart": {"nowarm": True},
    "[pair] solreffriction 0.2 1": {"pair_solreffriction": "0.2 1"},
    "[pair] solreffriction 0.5 1": {"pair_solreffriction": "0.5 1"},
    "[pair] solreffriction 1 1": {"pair_solreffriction": "1 1"},
    "[pair] solreffriction 2 1": {"pair_solreffriction": "2 1"},
}


def table():
    print(f"\n{'candidate':38s} {'maxrot_deg':>10s} {'exc_mm':>8s} {'minncon':>7s} {'z_err_um':>8s} "
          f"{'slide_hop_um':>12s} | {'max rel err d vs analytic (measurement)':s}")
    for name, cfg in CANDIDATES.items():
        try:
            s = make(cfg)
            out = [run(s, *c) for c in CORNERS]
            hop, _ = slide_test(s)
        except RuntimeError as e:
            print(f"{name:38s} FAIL: {str(e)[:70]}")
            continue
        err = max(abs(o[3] - analytic_displacement(c[0], c[1], c[2], c[3] * 1e-3)) /
                  analytic_displacement(c[0], c[1], c[2], c[3] * 1e-3) for o, c in zip(out, CORNERS))
        print(f"{name:38s} {max(o[0] for o in out):10.3f} {max(o[1] for o in out):8.3f} {min(o[2] for o in out):7d} "
              f"{max(abs(o[4] - 0.025) for o in out) * 1e6:8.1f} {hop:12.1f} | {err:.3f}")


if __name__ == "__main__":
    diagnose()
    table()
