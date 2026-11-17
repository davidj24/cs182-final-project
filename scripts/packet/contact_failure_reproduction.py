"""REPRODUCTION (not the original run) of the free-cube contact scene's failure, at historical commit 5e0ae3f.

The original scene (assets/cube.xml at 9876867/5e0ae3f: free joint, box-plane contact, friction mu on the cube geom, floor friction 0,
elliptic cone, impratio 10, implicitfast, dt 0.001, solimp 0.99 0.999 0.001, force F at the centre of mass for k steps) is loaded
UNCHANGED from the isolated checkout given by PYTHONPATH; this script only adds per-step recording (z, rotation, velocity, ncon,
summed contact normal force / m g). Run:
  PYTHONPATH=<run-hist_contact>/src python contact_failure_reproduction.py <out_dir>
Outputs: reproduction_cases.npz, reproduction_summary.csv, random_1000.csv, three PNGs. Newly produced on 2026-11-16 at the time of
packet assembly; the HISTORICAL saved evidence (sim_frames dirs, docs/evidence, review 9876867) is stored separately."""
from __future__ import annotations
import sys, csv, json
from pathlib import Path
import numpy as np, mujoco
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from cs182_icl.sim import rollout as R
from cs182_icl.physics import analytic_displacement
from cs182_icl.types import Action, Cube

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False)
G = 9.81
CASES = {"light_slippery": (0.5, 0.3, 26.0, 200), "middle": (1.0, 0.45, 20.0, 150),
         "heavy_grippy": (1.5, 0.6, 26.0, 200), "light_grippy": (0.5, 0.6, 26.0, 200)}


def record(sim, F, k, v0=0.0, n_extra=1200):
    """Same stepping as Simulator.rollout, plus recording. v0 != 0 with F = 0: free sliding test (no applied force)."""
    mdl, d = sim.model, sim.data
    sim.restore(); d.qvel[0] = v0; mujoco.mj_forward(mdl, d)
    x0 = d.qpos[0]; q0inv = np.zeros(4); mujoco.mju_negQuat(q0inv, sim.settled["qpos"][3:7]); fx = d.xfrc_applied[sim.body]
    rec = {n: [] for n in ("t", "x", "z", "rot", "vx", "ncon", "Nsum", "Tsum", "on")}
    f6 = np.zeros(6)
    for i in range(k + n_extra):
        fx[0] = F if i < k else 0.0
        rec["on"].append(i < k and F != 0)
        mujoco.mj_step(mdl, d)
        N = T = 0.0
        for c in range(d.ncon):
            mujoco.mj_contactForce(mdl, d, c, f6); N += f6[0]; T += float(np.hypot(f6[1], f6[2]))
        rec["t"].append((i + 1) * sim.dt); rec["x"].append(d.qpos[0] - x0); rec["z"].append(d.qpos[2]); rec["vx"].append(d.qvel[0])
        rec["rot"].append(R._rot_deg(q0inv, d.qpos[3:7], sim._tmp)); rec["ncon"].append(d.ncon); rec["Nsum"].append(N); rec["Tsum"].append(T)
    fx[:] = 0.0
    return {k_: np.asarray(v) for k_, v in rec.items()}


sim = R.Simulator()
summ, store = [], {}
for name, (m, mu, F, k) in CASES.items():
    sim.set_cube(m, mu); r = record(sim, F, k); store[name] = r
    res = R.simulate_push(Cube(m, mu, 0), Action(F, k)); d_an = analytic_displacement(m, mu, F, k * 0.001)
    on = r["on"]; contact = r["ncon"] > 0
    summ.append(dict(case=name, m=m, mu=mu, F=F, k=k, d_sim_project_api=res.d, d_analytic=d_an, rel_err=(res.d - d_an) / d_an,
                     max_rot_deg=float(r["rot"].max()), max_height_mm=float((r["z"] - R.HALF).max() * 1e3),
                     steps_without_contact_during_push=int((~contact & on).sum()), push_steps=int(on.sum()),
                     Nmax_over_mg=float((r["Nsum"] / (m * G)).max()), first_step_N_over_mg=float(r["Nsum"][0] / (m * G)),
                     first_step_prediction=float((m * G + mu * F) / (1 + mu**2) / (m * G)),
                     friction_over_N_during_push_sliding_median=float(np.median((r["Tsum"] / np.maximum(r["Nsum"], 1e-12))[contact & on & (np.abs(r["vx"]) > 0.05)]))))
np.savez(out / "reproduction_cases.npz", **{f"{n}__{k_}": v for n, r in store.items() for k_, v in r.items()})
with open(out / "reproduction_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

# no applied force: the cube is given 3 m/s and left to slide (decelerates under friction only)
sim.set_cube(0.5, 0.6); free = record(sim, 0.0, 0, v0=3.0, n_extra=400)
fs = dict(z_max_mm=float((free["z"] - R.HALF).max() * 1e3), steps_no_contact=int((free["ncon"] == 0).sum()), steps=len(free["t"]), rot_max=float(free["rot"].max()))
json.dump(fs, open(out / "free_sliding_no_force.json", "w"), indent=1)


def shade(ax, t, on):
    if on.any(): ax.axvspan(t[on][0], t[on][-1], color="#f4d03f", alpha=0.18, lw=0)


# Figure 1: detailed light_grippy and middle, 4 panels each
fig, ax = plt.subplots(2, 4, figsize=(17, 6.4))
for row, name in enumerate(("light_grippy", "middle")):
    r = store[name]; m, mu, F, k = CASES[name]; t = r["t"]; on = r["on"]; lim = min(len(t), 2000)
    a = ax[row]
    a[0].plot(t, (r["z"] - R.HALF) * 1e3, color="#1f77b4"); a[0].set_ylabel("height above rest (mm)"); a0b = a[0].twinx(); a0b.plot(t, r["rot"], color="#c0392b", lw=1); a0b.set_ylabel("rotation (deg)", color="#c0392b")
    a[1].plot(t, r["vx"], color="#2a9d8f"); a[1].set_ylabel("horizontal velocity (m/s)"); a1b = a[1].twinx(); a1b.plot(t, r["x"], color="#555", lw=1); a1b.set_ylabel("horizontal displacement (m)")
    a[2].plot(t, r["Nsum"] / (m * G), color="#7b5ea7", lw=0.8); a[2].axhline(1, color="k", ls="--", lw=0.8); a[2].set_ylabel("summed normal force / (m g)"); a[2].set_ylim(-0.5, min(6, (r["Nsum"] / (m * G)).max() * 1.05))
    a[3].step(t, r["ncon"], color="#555", lw=0.8); a[3].set_ylabel("number of active contacts"); a[3].set_ylim(-0.2, 4.4)
    for j, x in enumerate(a):
        shade(x, t, on); x.set_xlabel("time (s)"); x.set_xlim(0, t[min(len(t) - 1, int(np.argmax(r["vx"] < 1e-3)) + 200 if r["vx"].max() > 0 else len(t) - 1)] if False else t[-1])
        x.grid(alpha=0.2)
    a[0].set_title(f"{name}: m={m} kg, mu={mu}, F={F} N, k={k} (force on: shaded)", fontsize=9, loc="left")
fig.text(0.01, 0.005, "REPRODUCTION at historical commit 5e0ae3f (free cube, box-plane contact, solimp 0.99 0.999 0.001). Shaded = applied force on. Dashed line in column 3 = N = m g (the value the ideal model assumes).", fontsize=8)
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out / "fig1_contact_failure_traces.png", dpi=150); plt.close(fig)

# Figure 2: displacement disagreement + rotation over 1000 random training-domain pushes (project API at the historical commit)
rng = np.random.default_rng(20261116); n = 1000
m = np.exp(rng.uniform(np.log(0.5), np.log(1.5), n)); mu = rng.uniform(0.3, 0.6, n); F = rng.uniform(13, 26, n); k = rng.integers(100, 201, n)
items = [(Cube(float(m[i]), float(mu[i]), i), Action(float(F[i]), int(k[i]))) for i in range(n)]
res = R.simulate_batch(items, workers=16, seed=0)
d_sim = np.array([r.d for r in res]); rot = np.array([r.max_rotation_deg for r in res]); to = np.array([r.timed_out for r in res])
d_an = np.array([analytic_displacement(m[i], mu[i], F[i], k[i] * 0.001) for i in range(n)]); rel = (d_sim - d_an) / d_an
with open(out / "random_1000.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["m", "mu", "F", "k", "d_sim", "d_analytic", "rel_err", "max_rot_deg", "timed_out"])
    for i in range(n): w.writerow([m[i], mu[i], F[i], k[i], d_sim[i], d_an[i], rel[i], rot[i], to[i]])
stats = dict(n=n, rot_gt_1deg=int((rot > 1).sum()), rot_gt_10deg=int((rot > 10).sum()), rot_gt_45deg=int((rot > 45).sum()), timeouts=int(to.sum()),
             rel_err_median_abs=float(np.median(np.abs(rel))), rel_err_p95_abs=float(np.percentile(np.abs(rel), 95)), rel_err_max_abs=float(np.abs(rel).max()),
             n_rel_gt_1pct=int((np.abs(rel) > 0.01).sum()))
json.dump(stats, open(out / "random_1000_stats.json", "w"), indent=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
ax[0].hist(np.log10(np.abs(rel) + 1e-9), bins=50, color="#c0392b"); ax[0].set_xlabel("log10 |d_sim / d_analytic - 1|"); ax[0].set_ylabel("pushes (of 1000)"); ax[0].set_title("displacement disagreement", fontsize=10)
sc = ax[1].scatter(m, mu, c=np.minimum(rot, 90), s=10, cmap="viridis"); plt.colorbar(sc, ax=ax[1], label="max rotation (deg), clipped at 90"); ax[1].set_xlabel("m (kg)"); ax[1].set_ylabel("mu"); ax[1].set_title("where the cube tumbles", fontsize=10)
ax[2].scatter(d_an, d_sim, s=8, c=np.minimum(rot, 90), cmap="viridis"); lim = [d_an.min(), d_an.max()]; ax[2].plot(lim, lim, "k--", lw=0.8); ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("analytic d (m)"); ax[2].set_ylabel("simulated d (m)"); ax[2].set_title("simulated vs analytic", fontsize=10)
fig.text(0.01, 0.005, f"REPRODUCTION at 5e0ae3f, 1000 seeded training-domain pushes. {stats['rot_gt_1deg']} rotate >1 deg, {stats['rot_gt_45deg']} >45 deg; median |rel err| {stats['rel_err_median_abs']:.1e}, p95 {stats['rel_err_p95_abs']:.1e}, max {stats['rel_err_max_abs']:.2f}.", fontsize=8)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(out / "fig2_random_1000.png", dpi=150); plt.close(fig)

# Figure 3: no applied force, 3 m/s initial slide
t = free["t"]; fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
ax[0].plot(t, (free["z"] - R.HALF) * 1e3); ax[0].set_ylabel("height above rest (mm)"); ax[1].plot(t, free["vx"]); ax[1].set_ylabel("horizontal velocity (m/s)")
ax[2].step(t, free["ncon"], lw=0.8); ax[2].set_ylabel("active contacts")
for x in ax: x.set_xlabel("time (s)"); x.grid(alpha=0.2)
fig.text(0.01, 0.005, f"REPRODUCTION at 5e0ae3f: m=0.5, mu=0.6, NO applied force, initial speed 3 m/s. Max height {fs['z_max_mm']:.2f} mm; {fs['steps_no_contact']} of {fs['steps']} steps without contact.", fontsize=8)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(out / "fig3_no_force_sliding.png", dpi=150); plt.close(fig)
print(json.dumps(stats)); print(json.dumps(fs)); [print(s) for s in summ]
