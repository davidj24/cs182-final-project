"""Computes the verified numbers quoted in the guide (control-authored; run from the integration/release checkout with PYTHONPATH=<checkout>/src:<checkout>/scripts).
Writes JSON/CSV into the given output dir (must not exist)."""
import json, sys, os
from pathlib import Path
import numpy as np, torch
from cs182_icl.physics import analytic_displacement, stop_time
from cs182_icl.sim.rollout import simulate_push
from cs182_icl.types import Action, Cube
import pilot_load_run as L
from cs182_icl.train.data import RowSampler, load_dataset
from cs182_icl.model.tokens import encode, standardize_target

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False); DATA = Path(os.environ["CS282A_DATA"]); G = 9.81
res = {}
# ---- worked example, plain python (independent of project code) ----
m, mu, F, k = 1.0, 0.45, 20.0, 150; T = k * 0.001
fmax = mu * m * G; a_push = (F - fmax) / m; v = a_push * T; x1 = 0.5 * a_push * T**2; a_dec = mu * G; x2 = v**2 / (2 * a_dec); t_coast = v / a_dec
A = 1 / (2 * m**2 * mu * G); B = 1 / (2 * m); d_formula = F * T**2 * (A * F - B)
res["worked"] = dict(m=m, mu=mu, F=F, k=k, T=T, friction_force_N=fmax, a_push=a_push, v_at_release=v, x_push=x1, a_decel=a_dec, x_coast=x2, d_two_phase=x1 + x2,
                     A=A, B=B, d_formula=d_formula, t_stop=T + t_coast, t_stop_formula=F * T / (m * mu * G))
sim = simulate_push(Cube(m, mu, 0), Action(F, k))
res["worked"].update(d_project_analytic=analytic_displacement(m, mu, F, T), d_simulator=sim.d, t_stop_simulator_incl_0p05_dwell=sim.t_stop)
# same-force rule: second push at same F, T2 = 0.20 s
T2 = 0.20; res["same_force"] = dict(T2=T2, d_rule=d_formula * (T2 / T) ** 2, d_direct=analytic_displacement(m, mu, F, T2), d_simulator=simulate_push(Cube(m, mu, 0), Action(F, 200)).d)
# ambiguity example: two different cubes that agree on d at one (F,T) but differ at another F
def brentq(f, a, b, tol=1e-14):
    fa, fb = f(a), f(b); assert fa * fb < 0
    for _ in range(200):
        c = 0.5 * (a + b); fc = f(c)
        if fa * fc <= 0: b, fb = c, fc
        else: a, fa = c, fc
        if b - a < tol: break
    return 0.5 * (a + b)
F1, T1 = 20.0, 0.15; d1 = analytic_displacement(1.0, 0.45, F1, T1)
m2 = 1.2; y1 = d1 / (F1 * T1**2); mu2 = F1 / (m2 * G * (2 * m2 * y1 + 1)); assert 0.3 <= mu2 <= 0.6  # exact: y1 = F1/(2 m^2 mu g) - 1/(2m)
Fq = 24.0; res["ambiguity"] = dict(cube_a=(1.0, 0.45), cube_b=(m2, mu2), d_at_F1=(d1, analytic_displacement(m2, mu2, F1, T1)),
          d_at_Fq=(analytic_displacement(1.0, 0.45, Fq, T1), analytic_displacement(m2, mu2, Fq, T1)), F1=F1, Fq=Fq, T=T1)
# ---- decoded saved examples ----
ds = DATA / "datasets" / "pilot_v1"; tr = np.load(ds / "train.npz"); e1 = np.load(ds / "e1.npz"); meta = json.load(open(ds / "meta.json"))
i = 0; res["train_cube0"] = dict(cube_id=int(tr["cube_id"][i]), m=float(tr["m"][i]), mu=float(tr["mu"][i]),
          first5=[dict(F=float(tr["F"][i, j]), k=int(tr["k"][i, j]), T=float(tr["k"][i, j] * 0.001), d_clean=float(tr["d_clean"][i, j]), measurement_id=int(tr["measurement_id"][i, j])) for j in range(5)])
j = 3; res["e1_item3"] = dict(cube_id=int(e1["cube_id"][j]), m=float(e1["m"][j]), mu=float(e1["mu"][j]), query=dict(F=float(e1["query_F"][j]), k=int(e1["query_k"][j]), d_query_clean=float(e1["d_query_clean"][j])),
          context=[dict(F=float(e1["ctx_F"][j, q]), k=int(e1["ctx_k"][j, q]), d_obs=float(e1["ctx_d_obs"][j, q]), mid=int(e1["ctx_measurement_id"][j, q])) for q in range(3)])
# a training row: sample exactly like RowSampler (seeded) -> shows tokens
dsl = load_dataset(ds); norm = dsl["norm"]; dev = torch.device("cpu")
samp = RowSampler(dsl["train"], dev, seed=0); Fb, Tb, dob, dcl = samp.sample(1)
tok = encode(norm, Fb.float(), Tb.float(), dob.float())[0]
res["token_example"] = dict(F=Fb[0].tolist(), T=Tb[0].tolist(), d_obs=dob[0].tolist(), tokens_first6=tok[:6].tolist(), target_std_first3=standardize_target(norm, dcl[0, :3]).tolist())
# ---- per-position validation loss for best.pt and last.pt ----
run = DATA / "runs" / "pilot_v1_n0_s0"; rows = []
for ck in ("best.pt", "last.pt"):
    model, nrm, info, _ = L.load_checked(run, ck, "cpu"); vs = RowSampler(dsl["val"], dev, seed=0); rc, ri = dsl["val_rows"]
    rc_t = torch.as_tensor(rc); ri_t = torch.as_tensor(ri); sq = []
    with torch.no_grad():
        for s in range(0, len(rc_t), 500):
            Fv, Tv, dov, dcv = vs.gather(rc_t[s:s + 500], ri_t[s:s + 500])
            p = model.push_predictions(encode(nrm, Fv.float(), Tv.float(), dov.float())); y = standardize_target(nrm, dcv).float(); sq.append(((p - y) ** 2).numpy())
    sq = np.concatenate(sq); per = sq.mean(0)
    rows.append(dict(ckpt=ck, step=info["step"], sha256=info["ckpt_sha256"], mean=float(per.mean()), pos0_share=float(per[0] / per.sum()), per_position=per.tolist(),
                     mean_pos1_to_10=float(per[1:].mean())))
res["per_position_val"] = rows
json.dump(res, open(out / "guide_facts.json", "w"), indent=1, default=float)
import csv
with open(out / "per_position_val_loss.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ckpt", "step", "position"] + ["std_mse"]); [w.writerow([r["ckpt"], r["step"], q, r["per_position"][q]]) for r in rows for q in range(11)]
print(json.dumps({k: v for k, v in res.items() if k in ("worked", "same_force", "ambiguity")}, indent=1, default=float))
print([(r["ckpt"], r["step"], round(r["mean"], 6), round(r["pos0_share"], 4), round(r["mean_pos1_to_10"], 6), [round(x, 7) for x in r["per_position"][:3]]) for r in rows])
