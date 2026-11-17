"""Control-authored (pilot): training/validation curves from a run's metrics.jsonl (to be committed as scripts/pilot_plot_training.py).
  python pilot_plot_training.py --run-dir <run> --out <new png>
Train loss = mean over the logging window (every 50 steps); val loss every 1000 steps. Standardized MSE over all 11 push positions
(position 0 has no context, so its irreducible error dominates the plateau)."""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--run-dir", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
run = Path(a.run_dir); rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
tr = [(r["step"], r["train_loss"]) for r in rows if r["type"] == "train"]; va = [(r["step"], r["val_loss"]) for r in rows if r["type"] == "val"]
best = min(va, key=lambda x: x[1]); out = Path(a.out); assert not out.exists(), "refusing to overwrite"
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, (lo, hi) in enumerate([(0, 150000), (0, 6000)]):
    t = [x for x in tr if lo <= x[0] <= hi]; v = [x for x in va if lo <= x[0] <= hi]
    ax[k].plot(*zip(*t), color="#c0392b", lw=1.2, label="train (window mean)"); ax[k].plot(*zip(*v), color="#1f77b4", marker="o", ms=3, lw=1.2, label="validation")
    ax[k].set_yscale("log"); ax[k].set_xlabel("step"); ax[k].grid(True, alpha=0.25); ax[k].spines[["top", "right"]].set_visible(False)
ax[0].axvline(best[0], color="gray", ls="--", lw=1); ax[0].annotate(f"best val {best[1]:.5f}\nstep {best[0]}", (best[0], best[1]), xytext=(-110, 40), textcoords="offset points", fontsize=8, arrowprops=dict(arrowstyle="-", color="gray"))
ax[0].set_ylabel("standardized MSE (all 11 positions)"); ax[0].set_title("pilot_v1_n0_s0: full run (150k steps)", fontsize=10); ax[1].set_title("first 6k steps", fontsize=10); ax[0].legend(frameon=False, fontsize=8)
fig.text(0.01, 0.005, "Idealized 1D Coulomb dynamics; 1 seed; noise 0. Plateau ~0.06 is dominated by position 0 (no context).", fontsize=7); fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=160)
print("best", best, "n train recs", len(tr), "n val recs", len(va))
