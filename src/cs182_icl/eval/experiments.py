"""Experiment runners E1-E4 (PILOT_SPEC K, L, M, N). Each writes metrics CSVs and one figure
(+ plotted-series CSV beside it) into out_dir, which must be a fresh results directory.
No model or simulator is needed: any Predictor works, and Bayes is optional (bayes=None)."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from cs182_icl.eval import plots
from cs182_icl.eval.cases import Case, assert_clean_targets, assert_prefix_paired
from cs182_icl.eval.metrics import (
    COLUMNS,
    Scores,
    bayes_denominator,
    evaluate_all,
    header_lines,
    metric_table,
    tertile_edges,
    unique_item_targets,
    write_csv,
)
from cs182_icl.eval.paired import paired_across_keys
from cs182_icl.eval.predictors import Predictor
from cs182_icl.types import TestItem
from dataclasses import replace

E1_CS = (0, 1, 2, 4, 8, 10)
E4_CS = (0, 2, 10)
E2_CELLS = {  # cell -> (query type, context type), PILOT_SPEC L1
    "A": ("familiar-force query", "same-force"),
    "B": ("new-force query", "same-force"),
    "C": ("familiar-force query", "distinct-force"),
    "D": ("new-force query", "distinct-force"),
}
PAIRED_COLUMNS = ["comparison", "group", "c", "method_a", "method_b", "n", "n_pairs_total", "mean_diff_sqerr_m2", "se", "ci_lo", "ci_hi"]


# ---------------------------------------------------------------- shared helpers
def _c0_cases(cases: Sequence[Case]) -> list[Case]:
    """Empty-context cases of the unique items: the set on which D (Bayes c=0) is computed."""
    seen, out = set(), []
    for cs in cases:
        if cs.item_id not in seen:
            seen.add(cs.item_id)
            out.append(Case(cs.item_id, 0, replace(cs.item, context=[], condition="c0-denominator")))
    return out


def _bayes_setup(bayes: Predictor | None, denom_cases, cache):
    """-> (D or None, note). D only if the Bayes c=0 set on the denominator items is all ok."""
    if bayes is None:
        return None, "bayes: unavailable"
    sc = evaluate_all(_c0_cases(denom_cases), [bayes], cache)[bayes.name]
    D = bayes_denominator(sc)
    if D is None:
        return None, "bayes: unavailable (c=0 Bayes set not fully valid)"
    return D, ""


def _series(scores: dict[str, Scores], var_d: float, groups: dict) -> list[dict]:
    """groups: {(panel, style, x): case-index array} -> plotted rows (standardized MSE, SE)."""
    rows = []
    for (panel, style, x), idx in groups.items():
        idx = np.asarray(idx)
        for name, s in scores.items():
            ok = s.ok[idx]
            e = s.sq_err[idx][ok]
            rows.append(
                {
                    "panel": panel,
                    "style": style,
                    "method": name,
                    "x": x,
                    "value": float(e.mean() / var_d) if len(e) else float("nan"),
                    "se": float(e.std(ddof=1) / np.sqrt(len(e)) / var_d) if len(e) > 1 else float("nan"),
                    "n_ok": int(ok.sum()),
                    "n": len(idx),
                }
            )
    return rows


def _fresh(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _paired_row(comparison, group, c, a, b, stats):
    return {
        "comparison": comparison, "group": group, "c": c, "method_a": a, "method_b": b,
        "n": stats["n"], "n_pairs_total": stats.get("n_pairs_total", stats["n"]),
        "mean_diff_sqerr_m2": stats["mean_diff"], "se": stats["se"],
        "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
    }


def print_examples(cases: Sequence[Case], scores: dict[str, Scores], c: int, n: int = 10) -> str:
    """Print ~n item-level examples at context length c: query F,T, true d, each method."""
    names = list(scores)
    idx = [i for i, cs in enumerate(cases) if cs.c == c][:n]
    lines = ["condition   item  F[N]   T[s]   d_true[m]  " + "  ".join(f"{m:>14s}" for m in names)]
    for i in idx:
        cs = cases[i]
        cells = []
        for m in names:
            s = scores[m]
            cells.append(f"{s.d_pred[i]:14.6f}" if s.status[i] == "ok" else f"{s.status[i]:>14s}")
        lines.append(
            f"{cs.item.condition:10s} {cs.item_id:4d} {cs.item.query.F:6.2f} {cs.item.query.T:6.3f} {cs.target:11.6f}  "
            + "  ".join(cells)
        )
    if "bayes_train" not in scores:
        lines.append("(bayes: unavailable)")
    text = f"examples at c={c}\n" + "\n".join(lines)
    print(text)
    return text


# ---------------------------------------------------------------- E1
def run_e1(cases, predictors, out_dir, var_d, *, bayes=None, cache=None, seed=0, print_c=2):
    """E1: error vs context length. `cases` from prefix_cases(rows, E1_CS)."""
    assert_prefix_paired(cases)
    assert_clean_targets(cases)
    out = _fresh(out_dir)
    preds = list(predictors) + ([bayes] if bayes is not None else [])
    D, note = _bayes_setup(bayes, cases, cache)
    scores = evaluate_all(cases, preds, cache)
    first = next(iter(scores.values()))
    edges = tertile_edges(unique_item_targets(first))
    rows = metric_table(
        scores, var_d, bayes=scores[bayes.name] if D is not None else None, bayes_D=D, edges=edges
    )
    header = header_lines(var_d, D, edges, ["experiment: E1, context = first c of one ordered 10-push list (paired)"])
    write_csv(out / "e1_metrics.csv", rows, COLUMNS, header)
    # paired: each method at c vs c=0 (same items), and model vs each other method at each c
    paired = []
    for name, s in scores.items():
        base = s.c == 0
        for c in sorted(set(s.c.tolist()) - {0}):
            st = paired_across_keys(s, s, s.c == c, base, seed=seed)
            paired.append(_paired_row("c vs c=0", "E1", c, name, name, st))
    models = [n for n, s in scores.items() if s.kind == "model"]
    for m in models:
        for name, s in scores.items():
            if name == m:
                continue
            for c in sorted(set(first.c.tolist())):
                msk = first.c == c
                st = paired_across_keys(scores[m], s, msk, msk, seed=seed)
                paired.append(_paired_row("model vs method", "E1", c, m, name, st))
    write_csv(out / "e1_paired.csv", paired, PAIRED_COLUMNS, header)
    groups = {("main", "all", c): np.where(first.c == c)[0] for c in sorted(set(first.c.tolist()))}
    series = _series(scores, var_d, groups)
    plots.plot_e1(series, out / "e1_error_vs_context.png", header)
    ex = print_examples(cases, scores, print_c)
    return {"scores": scores, "table": rows, "paired": paired, "series": series, "bayes_D": D, "note": note, "examples": ex}


# ---------------------------------------------------------------- E2
def run_e2(cases, predictors, out_dir, var_d, *, bayes=None, cache=None, seed=0):
    """E2: cells A-D. Condition strings 'E2/A'..'E2/D'; c = context length of the row."""
    assert_clean_targets(cases)
    assert_prefix_paired(cases)
    out = _fresh(out_dir)
    preds = list(predictors) + ([bayes] if bayes is not None else [])
    D, note = _bayes_setup(bayes, cases, cache)
    scores = evaluate_all(cases, preds, cache)
    first = next(iter(scores.values()))
    cell = np.array([cs.item.condition.split("/")[-1] for cs in cases])
    edges = tertile_edges(unique_item_targets(first))
    rows = metric_table(scores, var_d, bayes=scores[bayes.name] if D is not None else None,
                        bayes_D=D, edges=edges)
    header = header_lines(var_d, D, edges, ["experiment: E2 cells A-D (A,B same-force history; C,D distinct-force; A,C familiar query; B,D new query)"])
    write_csv(out / "e2_metrics.csv", rows, COLUMNS, header)
    paired = []
    for name, s in scores.items():
        for same, dist, qtype in (("A", "C", "familiar"), ("B", "D", "new")):
            for c in sorted(set(first.c[(cell == same) | (cell == dist)].tolist())):
                ma, mb = (cell == same) & (first.c == c), (cell == dist) & (first.c == c)
                if ma.any() and mb.any():
                    st = paired_across_keys(s, s, ma, mb, seed=seed)
                    paired.append(_paired_row(f"same-force minus distinct-force ({qtype}-force query)", f"{same} vs {dist}", c, name, name, st))
    write_csv(out / "e2_paired.csv", paired, PAIRED_COLUMNS, header)
    groups = {}
    for ck, (panel, style) in E2_CELLS.items():
        for c in sorted(set(first.c[cell == ck].tolist())):
            groups[(panel, style, c)] = np.where((cell == ck) & (first.c == c))[0]
    series = _series(scores, var_d, groups)
    plots.plot_e2(series, out / "e2_same_vs_distinct.png", header)
    ex = print_examples(cases, scores, 2)
    return {"scores": scores, "table": rows, "paired": paired, "series": series, "bayes_D": D, "note": note, "examples": ex}


# ---------------------------------------------------------------- E3
def _k_of(cs: Case) -> int:
    m = re.search(r"k=(\d+)", cs.item.condition)
    if not m:
        raise ValueError(f"E3 condition without k=: {cs.item.condition}")
    return int(m.group(1))


def run_e3(cases, predictors, out_dir, var_d, *, bayes=None, cache=None, seed=0):
    """E3 no-noise redundancy. Conditions 'E3/k=<copies>'. References deduplicate by
    measurement_id (FunctionPredictor does this); the model sees the full sequence."""
    assert_clean_targets(cases)
    assert_prefix_paired(cases, group_key=lambda cs: (cs.item_id,))
    out = _fresh(out_dir)
    preds = list(predictors) + ([bayes] if bayes is not None else [])
    D, note = _bayes_setup(bayes, cases, cache)
    scores = evaluate_all(cases, preds, cache)
    first = next(iter(scores.values()))
    k = np.array([_k_of(cs) for cs in cases])
    edges = tertile_edges(unique_item_targets(first))
    rows = metric_table(scores, var_d, group_fn=lambda key: key[1], bayes=scores[bayes.name] if D is not None else None, bayes_D=D, edges=edges)
    header = header_lines(var_d, D, edges, [plots.E3_LABEL])
    write_csv(out / "e3_metrics.csv", rows, COLUMNS, header)
    # prediction change vs k=0 on the same item
    change_rows, series = [], []
    sd = np.sqrt(var_d)
    ks = sorted(set(k.tolist()))
    for name, s in scores.items():
        base = {i: (p, o) for i, p, o, kk in zip(s.item_id, s.d_pred, s.ok, k) if kk == 0}
        for kk in ks:
            idx = np.where(k == kk)[0]
            ch = np.array([abs(s.d_pred[i] - base[s.item_id[i]][0]) for i in idx
                           if s.ok[i] and s.item_id[i] in base and base[s.item_id[i]][1]])
            change_rows.append({"method": name, "k": kk, "n_changes": len(ch),
                                "mean_abs_change_m": float(ch.mean()) if len(ch) else float("nan"),
                                "max_abs_change_m": float(ch.max()) if len(ch) else float("nan")})
            series.append({"panel": "prediction change", "style": "all", "method": name, "x": kk,
                           "value": float(ch.mean() / sd) if len(ch) else float("nan"),
                           "se": float(ch.std(ddof=1) / np.sqrt(len(ch)) / sd) if len(ch) > 1 else float("nan"),
                           "n_ok": len(ch), "n": len(idx)})
    write_csv(out / "e3_prediction_change.csv", change_rows,
              ["method", "k", "n_changes", "mean_abs_change_m", "max_abs_change_m"], header)
    series += _series(scores, var_d, {("error", "all", kk): np.where(k == kk)[0] for kk in ks})
    plots.plot_e3(series, out / "e3_redundancy.png", header)
    ex = print_examples(cases, scores, int(first.c.min()))
    return {"scores": scores, "table": rows, "changes": change_rows, "series": series, "bayes_D": D, "note": note, "examples": ex}


# ---------------------------------------------------------------- E4
def run_e4(id_cases, ood_cases, predictors, out_dir, var_d, *, bayes=None, cache=None, seed=0):
    """E4: in-distribution E1 cases vs heavy-OOD cases at c in E4_CS (different cubes, so the
    comparison is descriptive, not paired). D for NMSE comes from the in-distribution c=0 set."""
    for cs in (id_cases, ood_cases):
        assert_clean_targets(cs)
        assert_prefix_paired(cs)
    out = _fresh(out_dir)
    preds = list(predictors) + ([bayes] if bayes is not None else [])
    D, note = _bayes_setup(bayes, id_cases, cache)
    s_id = evaluate_all(id_cases, preds, cache)
    s_ood = evaluate_all(ood_cases, preds, cache)
    f_id = next(iter(s_id.values()))
    edges = tertile_edges(unique_item_targets(f_id))
    rows = []
    for tag, sc in (("E1 in-distribution", s_id), ("E4 heavy OOD", s_ood)):
        t = metric_table(sc, var_d, group_fn=lambda key, tag=tag: tag,
                         bayes=sc[bayes.name] if D is not None else None, bayes_D=D, edges=edges)
        rows += t
    header = header_lines(var_d, D, edges, [
        "experiment: E4 " + plots.E4_LABEL + " (heavy cubes, no retraining; 50 cubes, one seed)",
        "tertile edges come from the in-distribution item targets and are applied to both sets",
        "NMSE denominator D is the in-distribution Bayes c=0 MSE, shared with the OOD set",
    ])
    write_csv(out / "e4_metrics.csv", rows, COLUMNS, header)
    groups_series = []
    for style, sc in (("in-distribution", s_id), ("heavy OOD", s_ood)):
        f = next(iter(sc.values()))
        g = {("main", style, c): np.where(f.c == c)[0] for c in sorted(set(f.c.tolist()))}
        groups_series += _series(sc, var_d, g)
    # gain from context: MSE(c)/MSE(0) per method and set (descriptive)
    gain = []
    for style, sc in (("in-distribution", s_id), ("heavy OOD", s_ood)):
        for name in sc:
            r0 = next(r for r in groups_series if r["method"] == name and r["style"] == style and r["x"] == 0)
            for r in groups_series:
                if r["method"] == name and r["style"] == style and r["x"] != 0:
                    gain.append({"set": style, "method": name, "c": r["x"],
                                 "smse_c": r["value"], "smse_c0": r0["value"],
                                 "ratio_c_over_c0": r["value"] / r0["value"] if r0["value"] > 0 else float("nan")})
    write_csv(out / "e4_context_gain.csv", gain, ["set", "method", "c", "smse_c", "smse_c0", "ratio_c_over_c0"], header)
    plots.plot_e4(groups_series, out / "e4_heavy_ood.png", header)
    return {"scores_id": s_id, "scores_ood": s_ood, "table": rows, "gain": gain, "series": groups_series, "bayes_D": D, "note": note}
