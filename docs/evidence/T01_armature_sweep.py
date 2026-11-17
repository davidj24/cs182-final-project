"""T01 repair 2: translation-only candidate (x,y,z slides, z armature 1000 kg) on random pushes.
Run: python docs/evidence/T01_armature_sweep.py  (about 1-2 min). Prints d error vs analytic (MEASUREMENT only),
fraction of push steps in contact, mean/min/max normal force / mg."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__))
import T01_candidates as T, mujoco, numpy as np
from cs182_icl.physics import analytic_displacement
base = T.HEAD + T.J["ii_xyz_free_z"] + T.TAIL
X = base.replace('axis="0 0 1"/>', 'axis="0 0 1" armature="1000"/>')

def go(m, mu, F, k, dt=0.001):
    mdl = mujoco.MjModel.from_xml_string(X); mdl.opt.timestep = dt; d = mujoco.MjData(mdl)
    mdl.body_mass[1] = m; mdl.body_inertia[1] = m*0.05**2/6; mdl.geom_friction[1, 0] = mu; mujoco.mj_setConst(mdl, d)
    for _ in range(int(0.5/dt)): mujoco.mj_step(mdl, d)
    x0 = d.qpos[0]; f6 = np.zeros(6); N = []; incon = 0; calm = 0; dwell = int(0.05/dt)
    for i in range(int(15/dt)):
        d.xfrc_applied[1, 0] = F if i < k else 0
        mujoco.mj_step(mdl, d)
        if i < k:
            n = 0
            for c in range(d.ncon): mujoco.mj_contactForce(mdl, d, c, f6); n += f6[0]
            N.append(n/(m*9.81)); incon += d.ncon > 0
        elif abs(d.qvel[0]) < 1e-4:
            calm += 1
            if calm >= dwell: break
        else: calm = 0
    return d.qpos[0]-x0, np.array(N), incon/k

def sweep(name, n, mrange, seed, dt=0.001):
    rng = np.random.default_rng(seed); rel = []; fin = []; nmin = []; nmax = []; nmean = []
    for _ in range(n):
        m = float(np.exp(rng.uniform(np.log(mrange[0]), np.log(mrange[1])))); mu = rng.uniform(.3, .6)
        F = rng.uniform(13, 26); k = int(rng.integers(100, 201)); ks = int(round(k*0.001/dt))
        dd, N, f = go(m, mu, F, ks, dt); an = analytic_displacement(m, mu, F, k*0.001)
        rel.append(abs(dd/an-1)); fin.append(f); nmin.append(N.min()); nmax.append(N.max()); nmean.append(N.mean())
    rel = np.array(rel)
    print(f"{name} n={n} dt={dt}: |d/an-1| median {np.median(rel):.2e} p95 {np.percentile(rel,95):.2e} max {rel.max():.2e}; "
          f"frac of push steps in contact: min {min(fin):.2f} median {np.median(fin):.2f}; "
          f"N/mg: worst-min {min(nmin):.2f} worst-max {max(nmax):.1f} median-of-means {np.median(nmean):.2f}")
    return rel

if __name__ == "__main__":
    sweep("train-range", 300, (0.5, 1.5), 0)
    sweep("OOD heavy", 50, (1.5, 2.0), 1)
    r1 = sweep("dt=0.001 (same 50)", 50, (0.5, 1.5), 2)
    r2 = sweep("dt=0.0005 (same 50)", 50, (0.5, 1.5), 2, dt=0.0005)
