"""Control-authored (pilot): log-scale meeting figures drawn ONLY from the saved result CSVs.

  python scripts/pilot_make_figures.py --results-dir $CS282A_DATA/results/vN --out-dir <new dir>

Why: the T06 default figures use a symlog axis, so references at ~1e-13 and exact zeros (the oracle
is the target itself, so its error is 0 by definition) flatten the transformer curve. Here the y axis
is log; zero values are not drawn (stated in the caption); every point is a number from a CSV.
Bayes appears only if the results CSVs contain it (valid Bayes only; T06 drops it otherwise).
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STYLE = {  # method -> (color, marker, label)
    "transformer": ("#c0392b", "o", "transformer"),
    "bayes_train": ("#1f77b4", "s", "Bayes (training prior)"),
    "physics_fit": ("#2a9d8f", "^", "physics fit (where determined)"),
    "analytic_true": ("#7b5ea7", "D", "analytic, true (m, mu)"),
    "oracle": ("#555555", "x", "simulator oracle (=target, error 0)"),
}
NOTE = "IDEALIZED 1D COULOMB DYNAMICS (not realistic contact). log y axis; value-0 points (oracle = target itself) not drawn. Error vs clean simulator target. Single seed."


def _series(ax, d, xlabel, ylabel, title):
    for m, (col, mk, lab) in STYLE.items():
        s = d[d.method == m].sort_values("x")
        s = s[(s.value > 0) & s.value.notna()]
        if s.empty:
            continue
        ax.errorbar(s.x, s.value, yerr=None, color=col, marker=mk, lw=2 if m == "transformer" else 1.2,
                    ms=6, label=lab, ls="-" if m == "transformer" else "--")
        if m == "transformer" and "se" in s and s.se.notna().all():
            lo = np.clip(s.value - s.se, s.value * 0.05, None)
            ax.fill_between(s.x, lo, s.value + s.se, color=col, alpha=0.15, lw=0)
        if "n_ok" in s and "n" in s:
            for _, r in s[s.n_ok < s.n].iterrows():
                ax.annotate(f"{int(r.n_ok)}/{int(r.n)}", (r.x, r.value), fontsize=7, xytext=(3, 4), textcoords="offset points", color=col)
    ax.set_yscale("log"); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=10)
    ax.grid(True, which="major", alpha=0.25); ax.spines[["top", "right"]].set_visible(False)


def cap(fig, txt, fs=7, bold=False):
    fig.text(0.01, 0.005, textwrap.fill(txt, 170), fontsize=fs, va="bottom", weight="bold" if bold else "normal")


def _coverage(d):
    """e.g. 'physics_fit c=0: 0/100 determined' for every series point with n_ok < n (those points are not drawn when n_ok=0)."""
    if "n_ok" not in d or "n" not in d:
        return ""
    bad = d[(d.n_ok < d.n) & (d.method != "transformer")]
    return "; ".join(f"{r.method} x={r.x:g}: {int(r.n_ok)}/{int(r.n)} determined" for _, r in bad.iterrows())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results-dir", required=True); ap.add_argument("--out-dir", required=True)
    a = ap.parse_args(); R = Path(a.results_dir); O = Path(a.out_dir); O.mkdir(parents=True, exist_ok=False)
    # E1: standardized MSE and median relative error vs c
    ser = pd.read_csv(R / "e1_error_vs_context.csv", comment="#"); met = pd.read_csv(R / "e1_metrics.csv", comment="#")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    _series(ax[0], ser[ser.panel == "main"], "context length c (first c of one 10-push list, paired)", "standardized MSE (÷ train Var(d))", "E1: error vs context")
    mm = met.rename(columns={"median_rel_err": "value", "c": "x"}).assign(se=np.nan)
    mm = mm[["method", "x", "value", "n_ok", "n"]]
    _series(ax[1], mm, "context length c", "median relative error", "E1: median relative error")
    ax[0].legend(fontsize=7, frameon=False, loc="upper right"); cap(fig, NOTE + "  Undetermined: " + _coverage(ser[ser.panel == "main"]), fs=6); fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(O / "E1_error_vs_context_log.png", dpi=160); plt.close(fig)
    # E2
    d = pd.read_csv(R / "e2_same_vs_distinct.csv", comment="#"); panels = list(dict.fromkeys(d.panel))
    fig, ax = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.2), squeeze=False)
    for i, p in enumerate(panels):
        for st, ls in (("same-force", "-"), ("distinct-force", ":")):
            sub = d[(d.panel == p) & (d["style"] == st)]
            for m, (col, mk, lab) in STYLE.items():
                if m in ("analytic_true", "oracle"):
                    continue
                t = sub[(sub.method == m) & (sub.value > 0)].sort_values("x")
                if t.empty:
                    continue
                ax[0, i].plot(t.x, t.value, color=col, marker=mk, ls=ls, lw=2.2 if m == "transformer" else 1.3, ms=6,
                              label=f"{lab} — {st}")
        ax[0, i].set_yscale("log"); ax[0, i].set_title(f"E2: {p}", fontsize=10); ax[0, i].set_xlabel("context length c")
        ax[0, i].set_xticks(sorted(set(d[d.panel == p].x))); ax[0, i].grid(True, alpha=0.25); ax[0, i].spines[["top", "right"]].set_visible(False)
        ax[0, i].set_ylabel("standardized MSE (÷ train Var(d))"); ax[0, i].legend(fontsize=7, frameon=False)
    cap(fig, "Solid = same-force context, dotted = distinct-force context (every method). 50 paired items. " + NOTE + "  Undetermined: " + _coverage(d), fs=6)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(O / "E2_same_vs_distinct_log.png", dpi=160); plt.close(fig)
    # E3
    d = pd.read_csv(R / "e3_redundancy.csv", comment="#"); panels = list(dict.fromkeys(d.panel))
    fig, ax = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4), squeeze=False)
    for i, p in enumerate(panels):
        _series(ax[0, i], d[d.panel == p], "number of appended exact copies", "mean |change| vs k=0 (÷ train std(d))" if "change" in p else "standardized MSE (÷ train Var(d))", f"E3: {p}")
        ax[0, i].set_xticks(sorted(set(d.x)))
    ax[0, 0].legend(fontsize=7, frameon=False, loc="best")
    cap(fig, "Preliminary no-noise redundancy test. This does not test the benefit of independent noisy repeated measurements.", fs=8, bold=True)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(O / "E3_redundancy_log.png", dpi=160); plt.close(fig)
    # E4
    d = pd.read_csv(R / "e4_heavy_ood.csv", comment="#"); fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for k, st in enumerate(["in-distribution", "heavy OOD"]):
        _series(ax[k], d[d["style"] == st], "context length c", "standardized MSE (÷ train Var(d))" if k == 0 else "", f"E4: {st}" + (" (m in [1.5, 2.0] kg)" if k else " (training range)"))
        ax[k].set_xticks([0, 2, 10])
    ax[0].legend(fontsize=7, frameon=False)
    cap(fig, "preliminary heavy-OOD behavior (heavy panel: 50 cubes; in-distribution panel: the 100 E1 items; no retraining). " + NOTE, fs=7)
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(O / "E4_heavy_ood_log.png", dpi=160); plt.close(fig)
    print("wrote", sorted(p.name for p in O.iterdir()))


if __name__ == "__main__":
    main()
