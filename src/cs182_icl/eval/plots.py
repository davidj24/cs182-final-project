"""Matplotlib figures, one per file, with the plotted series saved as CSV beside each figure.

Only methods with at least one valid (status ok) point are drawn; invalid cells are gaps.
Error axis: standardized MSE (denominator = training-set Var(d)), symlog below 1e-3 so exact
zeros (oracle) stay visible. Raw physical MSE is in the metrics CSVs.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from cs182_icl.eval.metrics import write_csv  # noqa: E402

E3_LABEL = (
    "Preliminary no-noise redundancy test. This does not test the benefit of independent "
    "noisy repeated measurements."
)
E4_LABEL = "preliminary heavy-OOD behavior"

_COLORS = {
    "transformer": "#d62728",
    "bayes_train": "#1f77b4",
    "physics_fit": "#2ca02c",
    "analytic_true": "#9467bd",
    "oracle": "#7f7f7f",
}
_MARKERS = {"transformer": "o", "bayes_train": "s", "physics_fit": "^", "analytic_true": "D", "oracle": "x"}
SERIES_COLUMNS = ["panel", "style", "method", "x", "value", "se", "n_ok", "n"]


def _style(method: str) -> dict:
    return {
        "color": _COLORS.get(method, "#444444"),
        "marker": _MARKERS.get(method, "o"),
        "lw": 2.4 if method == "transformer" else 1.6,
    }


def _clean_axes(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    if title:
        ax.set_title(title, fontsize=10)


def _lines(ax, rows, panel, *, linestyle_of=lambda r: "-"):
    """rows: series dicts; draw one line per (method, style) in this panel."""
    keys = sorted({(r["method"], r["style"]) for r in rows if r["panel"] == panel})
    for method, style in keys:
        pts = sorted(
            (r for r in rows if r["panel"] == panel and r["method"] == method and r["style"] == style),
            key=lambda r: r["x"],
        )
        pts = [p for p in pts if p["n_ok"] > 0 and np.isfinite(p["value"])]
        if not pts:
            continue
        x = np.array([p["x"] for p in pts], dtype=float)
        y = np.array([p["value"] for p in pts], dtype=float)
        se = np.array([p["se"] if np.isfinite(p["se"]) else 0.0 for p in pts], dtype=float)
        label = method if style in ("", "all") else f"{method} ({style})"
        ax.plot(x, y, linestyle=linestyle_of(pts[0]), label=label, **_style(method))
        ax.fill_between(x, np.maximum(y - se, 0), y + se, color=_style(method)["color"], alpha=0.12, lw=0)


def _save(fig, path: Path, rows, header):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    write_csv(path.with_suffix(".csv"), rows, SERIES_COLUMNS, header)


def plot_e1(rows, path, header, title="E1: does the transformer use previous interactions?"):
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _lines(ax, rows, "main")
    _clean_axes(ax, "context length c (first c of the same 10 pushes)", "standardized MSE (train Var(d))", title)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_xticks(sorted({r["x"] for r in rows}))
    fig.text(0.01, -0.02, "error vs clean MuJoCo target; band = +-1 SE over items", fontsize=7)
    _save(fig, path, rows, header)


def plot_e2(rows, path, header):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, panel in zip(axes, ("familiar-force query", "new-force query")):
        _lines(ax, rows, panel, linestyle_of=lambda r: "--" if r["style"] == "same-force" else "-")
        _clean_axes(ax, "context length c", "standardized MSE (train Var(d))", panel)
        ax.set_xticks(sorted({r["x"] for r in rows if r["panel"] == panel}))
    axes[1].legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("E2: same-force (dashed) vs distinct-force (solid) context", fontsize=11)
    _save(fig, path, rows, header)


def plot_e3(rows, path, header):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, panel, ylabel in zip(
        axes,
        ("error", "prediction change"),
        ("standardized MSE (train Var(d))", "mean |pred(k) - pred(k=0)| / std_train(d)"),
    ):
        _lines(ax, rows, panel)
        _clean_axes(ax, "appended copies k", ylabel, panel)
        ax.set_xticks(sorted({r["x"] for r in rows}))
    axes[1].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.suptitle("E3: robustness to duplicated context", fontsize=11)
    fig.text(0.5, -0.02, E3_LABEL, ha="center", fontsize=8, style="italic")
    _save(fig, path, rows, header)


def plot_e4(rows, path, header):
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    _lines(ax, rows, "main", linestyle_of=lambda r: "--" if r["style"] == "heavy OOD" else "-")
    _clean_axes(ax, "context length c", "standardized MSE (train Var(d))", f"E4: {E4_LABEL}")
    ax.set_xticks(sorted({r["x"] for r in rows}))
    ax.legend(frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.text(0.01, -0.03, "solid: in-distribution E1 cubes; dashed: heavy cubes (different cubes, not paired).\n"
             "50 cubes, one seed: not evidence of generalization.", fontsize=7)
    _save(fig, path, rows, header)
