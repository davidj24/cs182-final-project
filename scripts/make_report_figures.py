"""Regenerate the V1 report figures from the committed pilot result tables.

    uv run python scripts/make_report_figures.py            # or: python scripts/make_report_figures.py

Reads only results/pilot_v1/ (CSVs written by the pilot evaluation, plus the pilot's saved training
plot) and writes report/v1/figures/. No model, GPU or dataset is needed. Error bars are the `se`
column: one standard error over test cubes for a single trained model (not over training seeds).
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "results" / "pilot_v1"
CSV = PILOT / "csv"
RES = CSV / "results_v3_best_ckpt"
OUT = REPO / "report" / "v1" / "figures"

# method -> (color, marker, label)
STYLE = {
    "transformer": ("#c0392b", "o", "Transformer"),
    "bayes_train": ("#1f77b4", "s", "Bayes (training prior)"),
    "physics_fit": ("#2a9d8f", "^", "Physics fit (where determined)"),
}
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "legend.frameon": False})


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def series(rows, **match):
    """Sorted (x, y, se) for the rows matching `match`, skipping undetermined/non-positive points."""
    pts = []
    for r in rows:
        if all(r[k] == v for k, v in match.items()):
            y = float(r["value"])
            if y > 0:  # NaN (not determined) also fails this test
                pts.append((float(r["x"]), y, float(r["se"])))
    return sorted(pts)


def draw(ax, pts, method, ls="-", label=None, clip_for_log=True):
    if not pts:
        return
    x, y, se = (list(v) for v in zip(*pts))
    color, marker, default = STYLE[method]
    lower = [min(s, 0.9 * v) if clip_for_log else s for s, v in zip(se, y)]  # keep bars positive on log axes
    ax.errorbar(x, y, yerr=[lower, se], color=color, marker=marker, ls=ls, lw=1.4, ms=4, capsize=2,
                label=label or default)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name)
    plt.close(fig)


def fig_e1():
    ser, met = read(RES / "e1_error_vs_context.csv"), read(RES / "e1_metrics.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for m in STYLE:
        draw(ax[0], series(ser, panel="main", method=m), m)
        pts = sorted((float(r["c"]), 100 * float(r["median_rel_err"])) for r in met
                     if r["method"] == m and int(r["n_ok"]) > 0)
        if pts:
            ax[1].plot(*zip(*pts), color=STYLE[m][0], marker=STYLE[m][1], ms=4, lw=1.4, label=STYLE[m][2])
    ax[0].set_ylabel("standardized MSE (log)")
    ax[1].set_ylabel("median relative error, % (log)")
    for a in ax:
        a.set_yscale("log")
        a.set_xlabel("context pushes c")
        a.set_xticks([0, 1, 2, 4, 8, 10])
        a.grid(alpha=0.25)
    ax[0].legend(fontsize=6.5, loc="center right")
    save(fig, "fig_e1.pdf")


def fig_e2():
    rows = read(RES / "e2_same_vs_distinct.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for a, panel in zip(ax, ["familiar-force query", "new-force query"]):
        for style, ls in (("same-force", "-"), ("distinct-force", "--")):
            for m in STYLE:
                draw(a, series(rows, panel=panel, style=style, method=m), m, ls=ls,
                     label=f"{STYLE[m][2].split(' (where')[0]}, {style.split('-')[0]}")
        a.set_title(panel, fontsize=8)
        a.set_yscale("log")
        a.set_xlabel("context pushes c")
        a.set_xticks([1, 2, 4])
        a.grid(alpha=0.25)
    ax[0].set_ylabel("standardized MSE (log)")
    ax[1].legend(fontsize=5.5, ncol=2, loc="center left")
    save(fig, "fig_e2.pdf")


def fig_e3_e4():
    e3, e4 = read(RES / "e3_redundancy.csv"), read(RES / "e4_heavy_ood.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    pts = sorted((float(r["x"]), float(r["value"]), float(r["se"])) for r in e3
                 if r["panel"] == "prediction change" and r["method"] == "transformer")
    draw(ax[0], pts, "transformer", clip_for_log=False)
    ax[0].set_xlabel("appended exact copies k")
    ax[0].set_ylabel("|change in prediction| / train std (linear)")
    ax[0].set_xticks([0, 1, 2, 4])
    ax[0].set_ylim(bottom=0)
    ax[0].set_title("E3: effect of duplicates", fontsize=8)
    for style, ls in (("in-distribution", "-"), ("heavy OOD", "--")):
        for m in ("transformer", "bayes_train"):
            draw(ax[1], series(e4, style=style, method=m), m, ls=ls,
                 label=f"{STYLE[m][2].split(' (')[0]}, {'in range' if style == 'in-distribution' else 'heavy'}")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("context pushes c")
    ax[1].set_ylabel("standardized MSE (log)")
    ax[1].set_xticks([0, 2, 10])
    ax[1].set_title("E4: heavier cubes", fontsize=8)
    ax[1].legend(fontsize=6)
    for a in ax:
        a.grid(alpha=0.25)
    save(fig, "fig_e3_e4.pdf")


def fig_per_position():
    rows = read(CSV / "per_position_val_loss.csv")
    fig, ax = plt.subplots(figsize=(4.3, 2.2))
    for ck, color, label in (("best.pt", "#c0392b", "selected checkpoint (step 118k)"),
                             ("last.pt", "#7b5ea7", "final checkpoint (step 150k)")):
        pts = sorted((int(r["position"]), float(r["std_mse"])) for r in rows if r["ckpt"] == ck)
        ax.plot(*zip(*pts), color=color, marker="o", ms=3, lw=1.2, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("position = number of context pushes")
    ax.set_ylabel("validation std. MSE (log)")
    ax.set_xticks(range(11))
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5)
    save(fig, "fig_per_position.pdf")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_e1()
    fig_e2()
    fig_e3_e4()
    fig_per_position()
    # The training curve was plotted on the training machine from its metrics log, which is not committed.
    shutil.copyfile(PILOT / "figures" / "F05_training_curves.png", OUT / "fig_training.png")
    print("wrote", ", ".join(sorted(p.name for p in OUT.iterdir())))


if __name__ == "__main__":
    main()
