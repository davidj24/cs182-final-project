"""Compare the two loss-weighting ablation checkpoints on the same E1 and E2 test cases.

    PYTHONPATH=src:scripts python scripts/compare_ablation.py \
        --dataset-dir $CS282A_DATA/datasets/pilot_v1 \
        --run-a $CS282A_DATA/runs/ablation_A_all_positions \
        --run-b $CS282A_DATA/runs/ablation_B_context_only \
        --out-dir results/ablation

Run A trains on all 11 positions, run B on positions 1-10 only (report/v1/ABLATION_PLAN.md). Both use the
final checkpoint (last.pt), as fixed in that plan. Writes paired_a_vs_b.csv: A's squared error minus B's on
identical test cases (positive means B has the smaller error), with a paired bootstrap 95% interval over
test cubes. Also copies each run's validation records to val_per_position.csv. The intervals describe
test-cube sampling for these two trained models; they do not describe variation across training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cs182_icl.eval import experiments as ex
from cs182_icl.eval.cases import load_npz_rows, prefix_cases, rows_as_cases
from cs182_icl.eval.metrics import score_cases, train_d_variance_with_key
from cs182_icl.eval.paired import paired_across_keys
from cs182_icl.eval.predictors import ModelPredictor, predict_all
from pilot_load_run import load_checked


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--run-a", type=Path, required=True)
    ap.add_argument("--run-b", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((a.dataset_dir / "meta.json").read_text())
    var_d, _ = train_d_variance_with_key(meta)
    cases = {
        "E1": prefix_cases(load_npz_rows(a.dataset_dir / "e1.npz"), ex.E1_CS),
        "E2": rows_as_cases(load_npz_rows(a.dataset_dir / "e2.npz")),
    }
    models = {}
    for key, run in (("A", a.run_a), ("B", a.run_b)):
        model, norm, info, _ = load_checked(run, "last.pt", a.device)
        print(key, run.name, "step", info["step"], "sha256", info["ckpt_sha256"])
        models[key] = ModelPredictor(model, norm)

    out, per_case = [], {}
    for exp, cs in cases.items():
        s = {k: score_cases(m.name, "model", cs, predict_all(m, cs)) for k, m in models.items()}
        per_case[exp] = s
        first = s["A"]
        if exp == "E1":
            groups = [("all", c, first.c == c) for c in sorted(set(first.c.tolist()))]
        else:
            cell = np.array([case.item.condition.split("/")[-1] for case in cs])  # E2 cells A-D
            groups = [(g, c, (cell == g) & (first.c == c))
                      for g, c in sorted(set(zip(cell.tolist(), first.c.tolist())))]
        for g, c, mask in groups:
            st = paired_across_keys(s["A"], s["B"], mask, mask, seed=0)
            out.append({"experiment": exp, "group": g, "c": c, "n": st["n"],
                        "smse_a": s["A"].sq_err[mask].mean() / var_d,
                        "smse_b": s["B"].sq_err[mask].mean() / var_d,
                        "mean_diff_a_minus_b_m2": st["mean_diff"], "ci_lo": st["ci_lo"], "ci_hi": st["ci_hi"]})

    with open(a.out_dir / "paired_a_vs_b.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    with open(a.out_dir / "val_per_position.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["run", "step", "val_loss_all_positions", "val_loss_positions_1_to_10"] + [f"pos{i}" for i in range(11)])
        for key, run in (("A", a.run_a), ("B", a.run_b)):
            for line in (run / "metrics.jsonl").read_text().splitlines():
                r = json.loads(line)
                if r["type"] == "val":
                    w.writerow([key, r["step"], r["val_loss"], r["val_loss_ctx"]] + r["val_loss_per_pos"])
    for row in out:
        print({k: (f"{v:.3g}" if isinstance(v, float) else v) for k, v in row.items()})


if __name__ == "__main__":
    main()
