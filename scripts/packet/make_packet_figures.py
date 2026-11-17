"""Control-authored: generates the packet figures that do not come from the T06/pilot_make_figures pipeline. All data are read from saved artifacts.
Run from a checkout with PYTHONPATH=<checkout>/src:<checkout>/scripts and CS282A_DATA set:  python make_packet_figures.py <packet_dir>"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PK = Path(sys.argv[1]); FIG = PK / "figures"; CSV = PK / "csv"; DATA = Path(os.environ["CS282A_DATA"]); G = 9.81
CAP = "IDEALIZED 1D COULOMB DYNAMICS (not realistic contact). "


def cap(fig, t, fs=7):
    import textwrap; fig.text(0.01, 0.005, textwrap.fill(t, 170), fontsize=fs, va="bottom")


def new(name):  # never overwrite
    p = FIG / name; assert not p.exists(), p; return p


# ---------- F01 pipeline diagram ----------
steps = ["Sample cube parameters (m, mu) and push actions (F, k)", "Simulate clean push outcomes (idealized 1D Coulomb dynamics in MuJoCo)",
         "Build disjoint training / validation / test datasets (cube-level split)", "Encode histories as push / outcome tokens (training-only normalization)",
         "Train ONE causal transformer (150k steps, seed 0, noise 0)", "Freeze its weights; select a validation checkpoint (best.pt, step 118,000)",
         "Evaluate the SAME frozen checkpoint on E1, E2, E3, E4 (weights never change while reading a history)",
         "Compare with references: oracle, analytic, physics fit, Bayes (training / broad prior)", "Interpret results; choose the next experiment"]
fig, ax = plt.subplots(figsize=(7.6, 7.4)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, len(steps) * 1.25)
for i, s in enumerate(steps):
    y = (len(steps) - 1 - i) * 1.25 + 0.2; col = "#e8f1fa" if i not in (4, 5, 6) else "#fbe9e7"
    ax.add_patch(FancyBboxPatch((0.4, y), 9.2, 0.85, boxstyle="round,pad=0.02,rounding_size=0.12", fc=col, ec="#555", lw=1))
    ax.text(5, y + 0.43, f"{i + 1}. {s}", ha="center", va="center", fontsize=9)
    if i < len(steps) - 1: ax.add_patch(FancyArrowPatch((5, y), (5, y - 0.4), arrowstyle="-|>", mutation_scale=12, color="#555"))
fig.tight_layout(); cap(fig, "Red boxes = the single training run and its frozen use. " + CAP); fig.savefig(new("F01_pipeline.png"), dpi=170); plt.close(fig)

# ---------- F02 token diagram ----------
fig, ax = plt.subplots(figsize=(11, 3.6)); ax.axis("off"); ax.set_xlim(0, 22); ax.set_ylim(0, 6)
labels = ["push 0\n(F,T)", "out 0\n(d)", "push 1\n(F,T)", "out 1\n(d)", "query push\n(F,T)", "query out\n(d = 0, blank)", "...", "..."]
for i, l in enumerate(labels):
    blank = i == 5; x = 0.5 + i * 2.6
    ax.add_patch(FancyBboxPatch((x, 2.2), 2.2, 1.4, boxstyle="round,pad=0.02", fc="#ddd" if blank else ("#e8f1fa" if i % 2 == 0 else "#fdf2d8"), ec="#555", ls="--" if blank else "-"))
    ax.text(x + 1.1, 2.9, l, ha="center", va="center", fontsize=8)
ax.annotate("", xy=(0.5 + 4 * 2.6 + 1.1, 3.7), xytext=(0.5 + 4 * 2.6 + 1.1, 5.2), arrowprops=dict(arrowstyle="<|-", color="#c0392b", lw=2))
ax.text(0.5 + 4 * 2.6 + 1.1, 5.45, "prediction of the query displacement is READ HERE (position 2c = 4 for c = 2)", ha="center", fontsize=9, color="#c0392b")
ax.text(0.5, 1.4, "Causal mask: position t attends only to positions <= t. The query's own outcome (d) comes AFTER the readout position, so it cannot influence the prediction.", fontsize=9)
ax.text(0.5, 0.7, "Every push token also reads out a prediction during training (11 per row); the evaluation uses the readout at the query push token with c context pairs before it.", fontsize=9)
fig.tight_layout(); fig.savefig(new("F02_token_diagram.png"), dpi=170); plt.close(fig)

# ---------- F03 worked example: analytic piecewise vs final simulator replay ----------
m, mu, F, k = 1.0, 0.45, 20.0, 150; T = k * 1e-3; a1 = (F - mu * m * G) / m; a2 = mu * G
z = np.load(DATA / "pilot" / "sim_frames" / "20261116_205732" / "middle.npz", allow_pickle=True)
t = z["t"]; pos = z["pos"][:, 0]; vel = z["vel"]; x0 = pos[0]
def an_v(tt): return np.where(tt <= T, a1 * tt, np.maximum(a1 * T - a2 * (tt - T), 0))
def an_x(tt):
    vt = a1 * T; xT = 0.5 * a1 * T**2; tc = np.clip(tt - T, 0, vt / a2); return np.where(tt <= T, 0.5 * a1 * tt**2, xT + vt * tc - 0.5 * a2 * tc**2)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
ax[0].plot(t, vel, color="#2a9d8f", lw=2, label="simulator replay (final idealized scene)"); ax[0].plot(t, an_v(t), "k--", lw=1, label="hand-derived piecewise"); ax[0].set_ylabel("velocity (m/s)")
ax[1].plot(t, pos - x0, color="#555", lw=2, label="simulator replay"); ax[1].plot(t, an_x(t), "k--", lw=1, label="hand-derived piecewise"); ax[1].set_ylabel("displacement (m)")
for a in ax: a.axvspan(0, T, color="#f4d03f", alpha=0.18, lw=0); a.set_xlabel("time (s)"); a.grid(alpha=0.2); a.legend(fontsize=8, frameon=False); a.spines[["top", "right"]].set_visible(False)
ax[0].annotate("force on: a = (F - mu m g)/m = %.3f m/s^2" % a1, (0.01, a1 * 0.1), fontsize=8, xytext=(0.02, 1.2)); ax[0].annotate("force off: a = -mu g = -%.3f m/s^2" % a2, (0.3, 1.2), fontsize=8, xytext=(0.28, 1.6))
cap(fig, f"Worked example m=1.0 kg, mu=0.45, F=20 N, k=150 (T=0.15 s): shaded = force on. Final displacement {an_x(np.array([5.0]))[0]:.5f} m (hand) vs {pos[-1] - x0:.5f} m (replay). " + CAP)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(new("F03_worked_example.png"), dpi=160); plt.close(fig)

# ---------- F04 ambiguity ----------
def dan(m, mu, F, T): return np.where(F - mu * m * G > 0, F * T * T * (F - mu * m * G) / (2 * m * m * mu * G), 0.0)
Fs = np.linspace(13, 26, 300); T1 = 0.15; d1 = dan(1.0, 0.45, 20, T1); y1 = d1 / (20 * T1**2); mu2 = 20 / (1.2 * G * (2 * 1.2 * y1 + 1))
fig, ax = plt.subplots(1, 2, figsize=(11, 3.9)); a, b = dan(1.0, 0.45, Fs, T1), dan(1.2, mu2, Fs, T1)
ax[0].plot(Fs, a, color="#c0392b", label="cube A: m=1.0, mu=0.450"); ax[0].plot(Fs, b, color="#1f77b4", ls="--", label=f"cube B: m=1.2, mu={mu2:.3f}"); ax[0].axvline(20, color="gray", lw=0.8); ax[0].set_xlabel("push force F (N), T = 0.15 s"); ax[0].set_ylabel("displacement d (m)"); ax[0].legend(frameon=False, fontsize=8)
ax[1].plot(Fs, 100 * (b - a) / a, color="#555"); ax[1].axhline(0, color="gray", lw=0.8); ax[1].axvline(20, color="gray", lw=0.8); ax[1].set_xlabel("push force F (N)"); ax[1].set_ylabel("(d_B - d_A) / d_A  (%)")
for x in ax: x.grid(alpha=0.2); x.spines[["top", "right"]].set_visible(False)
cap(fig, "Two different cubes give the SAME displacement at F=20 N, T=0.15 s (one observation cannot tell them apart) but differ by under 1% elsewhere in the force range: parameters can be ambiguous while a query prediction is still accurate. Closed form only. " + CAP)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(new("F04_parameter_ambiguity.png"), dpi=160); plt.close(fig)

# ---------- F05 training curves, F06 per-position val ----------
run = DATA / "runs" / "pilot_v1_n0_s0"; rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
tr = [(r["step"], r["train_loss"]) for r in rows if r["type"] == "train"]; va = [(r["step"], r["val_loss"]) for r in rows if r["type"] == "val"]; best = min(va, key=lambda x: x[1])
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for kk, (lo, hi) in enumerate([(0, 150000), (0, 6000)]):
    tt = [x for x in tr if lo <= x[0] <= hi]; vv = [x for x in va if lo <= x[0] <= hi]
    ax[kk].plot(*zip(*tt), color="#c0392b", lw=1.2, label="train (mean over 50-step window)"); ax[kk].plot(*zip(*vv), color="#1f77b4", marker="o", ms=3, lw=1.2, label="validation (1000 fixed rows)")
    ax[kk].set_yscale("log"); ax[kk].set_xlabel("step"); ax[kk].grid(alpha=0.25); ax[kk].spines[["top", "right"]].set_visible(False)
ax[0].axvline(best[0], color="gray", ls="--", lw=1); ax[0].annotate(f"best val {best[1]:.5f}\nstep {best[0]}", (best[0], best[1]), xytext=(-110, 40), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color="gray"))
ax[0].set_ylabel("standardized MSE, mean over all 11 positions"); ax[0].set_title("full run: 150,000 steps", fontsize=10); ax[1].set_title("first 6,000 steps", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
cap(fig, "Source: runs/pilot_v1_n0_s0/metrics.jsonl. Both curves are flat after ~3k steps because ~99.9% of the validation loss is push 0 (no context; see F06). " + CAP)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(new("F05_training_curves.png"), dpi=160); plt.close(fig)
pp = {}
for r in csv.DictReader(open(sorted((PK / "guide_assets").glob("facts_v*/per_position_val_loss.csv"))[-1])): pp.setdefault(r["ckpt"], []).append(float(r["std_mse"]))
fig, ax = plt.subplots(figsize=(8, 4)); w = 0.38; xs = np.arange(11)
ax.bar(xs - w / 2, pp["best.pt"], w, color="#c0392b", label="best.pt (step 118,000)"); ax.bar(xs + w / 2, pp["last.pt"], w, color="#1f77b4", label="last.pt (step 150,000)")
ax.set_yscale("log"); ax.set_xlabel("position in the 11-push row (= number of context pushes before it)"); ax.set_ylabel("validation standardized MSE"); ax.set_xticks(xs); ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.2, axis="y"); ax.spines[["top", "right"]].set_visible(False)
cap(fig, "Source: guide_assets per_position_val_loss.csv (1000 fixed validation rows). Position 0 dominates the mean, so best-checkpoint selection is driven by the no-context error. " + CAP)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(new("F06_per_position_val_loss.png"), dpi=160); plt.close(fig)

# ---------- F08 c=1 audit: transformer vs exact Bayes (1D curve limit, plain numpy) ----------
from cs182_icl.eval.cases import prefix_cases, load_npz_rows
from cs182_icl.model.predict import model_predict
import pilot_load_run as L
ds = DATA / "datasets" / "pilot_v1"; e1 = np.load(ds / "e1.npz"); model, norm, info, _ = L.load_checked(run, "best.pt", "cpu")
cs = [x for x in prefix_cases(load_npz_rows(ds / "e1.npz"), [1]) if x.c == 1]; p_tf = model_predict(model, [x.item for x in cs], norm)
Fq, Tq, dq = e1["query_F"], e1["query_k"] * 1e-3, e1["d_query_clean"]; F1, T1, d1 = e1["ctx_F"][:, 0], e1["ctx_k"][:, 0] * 1e-3, e1["ctx_d_obs"][:, 0]
assert np.allclose([x.target for x in cs], dq)
def ana(m, mu, F, T): return np.where(F - mu * m * G > 0, F * T * T * (F - mu * m * G) / (2 * m * m * mu * G), 0.0)
M = np.exp(np.linspace(np.log(0.5), np.log(1.5), 400001)); bm = np.zeros(100); bsd = np.zeros(100)
for i in range(100):
    a_ = F1[i]**2 * T1[i]**2 / (2 * M * M * G); b_ = F1[i] * T1[i]**2 / (2 * M); mus = a_ / (d1[i] + b_); ok = (mus >= 0.3) & (mus <= 0.6)
    w_ = np.where(ok, (1 / M) * mus**2 / a_, 0.0) * np.gradient(M); w_ /= w_.sum(); dqs = ana(M, np.clip(mus, 1e-9, None), Fq[i], Tq[i])
    bm[i] = (w_ * dqs).sum(); bsd[i] = np.sqrt((w_ * (dqs - bm[i]) ** 2).sum())
dF = np.abs(Fq - F1)
with open(CSV / "c1_audit_transformer_vs_exact_bayes.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["item", "F1", "Fq", "abs_dF", "d_query_clean", "transformer", "exact_bayes_1d_curve", "posterior_sd_m", "rel_err_transformer", "rel_err_bayes"])
    for i in range(100): w.writerow([i, F1[i], Fq[i], dF[i], dq[i], p_tf[i], bm[i], bsd[i], abs(p_tf[i] - dq[i]) / dq[i], abs(bm[i] - dq[i]) / dq[i]])
VD = json.load(open(ds / "meta.json"))["normalization"]["d_obs"]["std"] ** 2
summ = dict(mse_transformer=float(np.mean((p_tf - dq) ** 2)), mse_bayes=float(np.mean((bm - dq) ** 2)), smse_transformer=float(np.mean((p_tf - dq) ** 2) / VD), smse_bayes=float(np.mean((bm - dq) ** 2) / VD),
            median_rel_transformer=float(np.median(np.abs(p_tf - dq) / dq)), median_rel_bayes=float(np.median(np.abs(bm - dq) / dq)), transformer_beats_bayes_items=int((np.abs(p_tf - dq) < np.abs(bm - dq)).sum()),
            error_corr=float(np.corrcoef(p_tf - dq, bm - dq)[0, 1]), checkpoint_sha256=info["ckpt_sha256"], ckpt_step=info["step"])
json.dump(summ, open(CSV / "c1_audit_summary.json", "w"), indent=1)
bins = [(0, 1), (1, 3), (3, 6), (6, 13)]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax[0].scatter(np.abs(bm - dq) / dq * 100, np.abs(p_tf - dq) / dq * 100, s=14, c="#c0392b"); lim = [1e-3, 30]; ax[0].plot(lim, lim, "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("exact-Bayes |relative error| at c=1 (%)"); ax[0].set_ylabel("transformer |relative error| at c=1 (%)"); ax[0].set_title("per item (100 E1 items); dashed = equal", fontsize=10)
xs = np.arange(len(bins)); mt = [np.median(np.abs(p_tf - dq)[(dF >= lo) & (dF < hi)] / dq[(dF >= lo) & (dF < hi)]) * 100 for lo, hi in bins]
mb = [np.median(np.abs(bm - dq)[(dF >= lo) & (dF < hi)] / dq[(dF >= lo) & (dF < hi)]) * 100 for lo, hi in bins]; ms = [np.median((bsd / bm)[(dF >= lo) & (dF < hi)]) * 100 for lo, hi in bins]
ax[1].bar(xs - 0.27, mt, 0.27, color="#c0392b", label="transformer"); ax[1].bar(xs, mb, 0.27, color="#1f77b4", label="exact Bayes"); ax[1].bar(xs + 0.27, ms, 0.27, color="#aaa", label="Bayes posterior sd (relative)")
ax[1].set_xticks(xs); ax[1].set_xticklabels([f"[{lo},{hi}) N" for lo, hi in bins]); ax[1].set_xlabel("|F_query - F_context| (N)"); ax[1].set_ylabel("median relative error (%)"); ax[1].legend(frameon=False, fontsize=8)
for x in ax: x.grid(alpha=0.2); x.spines[["top", "right"]].set_visible(False)
cap(fig, f"One observation (c=1). Transformer MSE {summ['mse_transformer']:.2e} m^2 vs exact Bayes {summ['mse_bayes']:.2e} m^2; transformer closer than Bayes on {summ['transformer_beats_bayes_items']}/100 items. Exact Bayes = sigma->0 posterior on the constraint curve computed here in plain numpy (not the discrepancy-aware project implementation). Being worse than Bayes does not by itself exclude leakage; see counterfactual checks. " + CAP)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(new("F08_c1_audit.png"), dpi=160); plt.close(fig)

# ---------- F12 E4 prior comparison ----------
sp = list(csv.DictReader(open(CSV / "supp_broad_prior_bayes" / "bayes_prior_comparison.csv")))
v3 = DATA / "results" / "v3"; import pandas as pd
m4 = pd.read_csv(v3 / "e4_metrics.csv", comment="#"); tf = m4[(m4.method == "transformer")]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for kk, setname in enumerate(("E1 in-distribution", "E4 heavy OOD")):
    cc = [0, 2, 10]
    for pname, col, lab in (("bayes_training_prior", "#1f77b4", "Bayes, training prior"), ("bayes_broad_prior", "#2a9d8f", "Bayes, broad prior")):
        ys = [float([r for r in sp if r["set"] == setname and int(r["c"]) == c and r["method"] == pname][0]["median_rel_err"]) for c in cc]; ax[kk].plot(cc, ys, marker="s", color=col, label=lab)
    g = tf[tf.group.str.contains("heavy" if kk else "in-distribution")]; ax[kk].plot(g.c, g.median_rel_err, marker="o", color="#c0392b", lw=2, label="transformer")
    ax[kk].set_yscale("log"); ax[kk].set_xlabel("context length c"); ax[kk].set_xticks(cc); ax[kk].set_ylabel("median relative error"); ax[kk].set_title(setname + (" (m in [1.5, 2.0] kg)" if kk else " (training range)"), fontsize=10); ax[kk].grid(alpha=0.2); ax[kk].spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False, fontsize=8)
cap(fig, "Sources: results/v3/e4_metrics.csv (transformer) and csv/supp_broad_prior_bayes (control-authored, outside the T06 harness). Training prior m in [0.5,1.5]; broad prior m in [0.35,2.0], mu in [0.2,0.7]. preliminary heavy-OOD behavior (50 cubes, 1 seed). " + CAP)
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(new("F12_e4_prior_comparison.png"), dpi=160); plt.close(fig)
print("wrote", sorted(p.name for p in FIG.iterdir()))
