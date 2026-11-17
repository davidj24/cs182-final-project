"""SUPPLEMENTARY (control-authored, not through the T06 harness): training-prior vs BROAD-prior Bayes on E1 (in-distribution) and E4 (heavy OOD)
items at c = 0, 2, 10, using the integrated reviewed bayes_predict / bayes_predict_broad. Metrics are against the clean simulator target.
Standardized MSE denominator: training-set Var(d) (meta.json normalization.d_obs.std squared). Writes a CSV (dir must not exist)."""
import csv, json, os, sys
from pathlib import Path
import numpy as np
from cs182_icl.eval.cases import prefix_cases, load_npz_rows
from cs182_icl.reference.bayes import bayes_predict, bayes_predict_broad
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=False); ds = Path(os.environ["CS282A_DATA"]) / "datasets" / "pilot_v1"
var_d = json.load(open(ds / "meta.json"))["normalization"]["d_obs"]["std"] ** 2; rows = []
for name, f in (("E1 in-distribution", "e1.npz"), ("E4 heavy OOD", "e4.npz")):
    cases = prefix_cases(load_npz_rows(ds / f), [0, 2, 10])
    for c in (0, 2, 10):
        cs = [x for x in cases if x.c == c]
        for pname, fn in (("bayes_training_prior", bayes_predict), ("bayes_broad_prior", bayes_predict_broad)):
            preds = [fn(list(x.item.context), x.item.query) for x in cs]; ok = np.array([p.status == "ok" for p in preds])
            y = np.array([x.target for x in cs]); p = np.array([p.d_pred if p.status == "ok" else np.nan for p in preds])
            e = (p - y)[ok]; rel = np.abs(e) / y[ok]
            rows.append(dict(set=name, c=c, method=pname, n=len(cs), n_ok=int(ok.sum()), mse_m2=float(np.mean(e**2)), smse=float(np.mean(e**2) / var_d), median_rel_err=float(np.median(rel))))
with open(out / "bayes_prior_comparison.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows: print({k: (round(v, 8) if isinstance(v, float) else v) for k, v in r.items()})
