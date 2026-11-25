"""Print every data table and quoted result number of the V1 report from the committed pilot CSVs.

    python scripts/make_report_tables.py          # standard library only

Table and section names follow report/v1/main.tex.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

CSV = Path(__file__).resolve().parents[1] / "results" / "pilot_v1" / "csv"
BEST, LAST = CSV / "results_v3_best_ckpt", CSV / "results_v4_last_ckpt_sensitivity"


def read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def pick(rows, **match):
    out = [r for r in rows if all(r[k] == str(v) for k, v in match.items())]
    if len(out) != 1:
        raise ValueError(f"expected one row for {match}, found {len(out)}")
    return out[0]


def title(text):
    print(f"\n== {text}")


def main():
    e1 = read(BEST / "e1_metrics.csv")
    title("E1 table: median relative error (%) and standardized MSE")
    for c in (0, 1, 2, 4, 8, 10):
        for m in ("transformer", "bayes_train"):
            r = pick(e1, c=c, method=m)
            print(f"c={c:<2} {m:<12} median_rel={100 * float(r['median_rel_err']):.3g}%  "
                  f"std_mse={float(r['smse_trainvar']):.3g}  mse_m2={float(r['mse_m2']):.3g}")
    r = pick(read(BEST / "e1_paired.csv"), comparison="model vs method", c=0, method_b="bayes_train")
    print(f"c=0 transformer - Bayes (m^2): {float(r['mean_diff_sqerr_m2']):.3g} "
          f"[{float(r['ci_lo']):.3g}, {float(r['ci_hi']):.3g}]")
    print("physics-fit coverage:", {c: pick(e1, c=c, method="physics_fit")["n_ok"] for c in (2, 4, 8, 10)})

    title("E2 paired table: same-force minus distinct-force, new-force query (1e-5 m^2, 95% bootstrap CI)")
    for label, folder, method in (("Bayes", BEST, "bayes_train"), ("Transformer selected", BEST, "transformer"),
                                  ("Transformer final", LAST, "transformer")):
        rows = read(folder / "e2_paired.csv")
        for c in (2, 4):
            r = pick(rows, group="B vs D", c=c, method_a=method)
            print(f"{label:<21} c={c}: {1e5 * float(r['mean_diff_sqerr_m2']):+.2f} "
                  f"[{1e5 * float(r['ci_lo']):.2f}, {1e5 * float(r['ci_hi']):.2f}]")

    title("E2 appendix table: standardized MSE")
    for folder, label in ((BEST, "selected"), (LAST, "final")):
        for r in read(folder / "e2_same_vs_distinct.csv"):
            if r["method"] in ("transformer", "bayes_train") and (label == "selected" or r["method"] == "transformer"):
                print(f"{label:<8} {r['panel']:<20} {r['style']:<14} c={r['x']} {r['method']:<12} {float(r['value']):.3g}")

    title("E3: mean and max |change| of the prediction (mm), and transformer standardized MSE")
    for r in read(BEST / "e3_prediction_change.csv"):
        if r["method"] == "transformer":
            print(f"k={r['k']} mean={1000 * float(r['mean_abs_change_m']):.1f} max={1000 * float(r['max_abs_change_m']):.1f}")
    for r in read(BEST / "e3_redundancy.csv"):
        if r["panel"] == "error" and r["method"] == "transformer":
            print(f"k={r['x']} std_mse={float(r['value']):.3g} se={float(r['se']):.2g}")
    print("final checkpoint, mean change (mm):",
          [round(1000 * float(r["mean_abs_change_m"]), 1) for r in read(LAST / "e3_prediction_change.csv")
           if r["method"] == "transformer"])

    title("E4 table: median relative error (%)")
    for r in read(CSV / "supp_broad_prior_bayes" / "bayes_prior_comparison.csv"):
        print(f"{r['set']:<18} c={r['c']:<2} {r['method']:<22} {100 * float(r['median_rel_err']):.3g}")
    for folder, label in ((BEST, "selected"), (LAST, "final")):
        for r in read(folder / "e4_metrics.csv"):
            if r["method"] == "transformer":
                print(f"transformer {label:<8} {r.get('group', ''):<18} c={r['c']:<2} "
                      f"{100 * float(r['median_rel_err']):.3g}")

    title("Sensitivity: E1 MSE, final vs selected checkpoint")
    last = read(LAST / "e1_metrics.csv")
    for c in (1, 2, 4, 8, 10):
        a, b = (float(pick(rows, c=c, method="transformer")["mse_m2"]) for rows in (e1, last))
        print(f"c={c}: {100 * (b / a - 1):+.0f}%")

    title("Training section: per-position validation loss")
    rows = read(CSV / "per_position_val_loss.csv")
    for ck in ("best.pt", "last.pt"):
        v = [float(r["std_mse"]) for r in sorted((r for r in rows if r["ckpt"] == ck), key=lambda r: int(r["position"]))]
        print(f"{ck}: pos0={v[0]:.3f} share={100 * v[0] / sum(v):.2f}% pos1..10 range={min(v[1:]):.2g}..{max(v[1:]):.2g} "
              f"mean_all={sum(v) / len(v):.4f}")

    title("One-push audit")
    print(json.dumps(json.loads((CSV / "c1_audit_summary.json").read_text()), indent=2))


if __name__ == "__main__":
    main()
