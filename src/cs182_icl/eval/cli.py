"""Command line entry for the evaluation harness (scripts/eval_run.py calls main)."""

from __future__ import annotations

import argparse
import json
import os
from importlib import import_module
from pathlib import Path

from cs182_icl.eval import experiments as ex
from cs182_icl.eval.cache import RefCache, next_results_version
from cs182_icl.eval.cases import load_npz_rows, prefix_cases, rows_as_cases
from cs182_icl.eval.metrics import train_d_variance
from cs182_icl.eval.predictors import (
    AnalyticTruePredictor,
    FunctionPredictor,
    ModelPredictor,
    OraclePredictor,
    bayes_predictor_or_none,
)
from cs182_icl.types import Action, Observation


def reference_predictors() -> list:
    """oracle + analytic always; physics fit if T03 has implemented it."""
    preds = [OraclePredictor(), AnalyticTruePredictor()]
    try:
        from cs182_icl.reference.physics_fit import physics_fit_predict

        physics_fit_predict([Observation(Action(15.0, 200), 1, 0.01, 0.01)], Action(18.0, 200))
        preds.append(FunctionPredictor("physics_fit", physics_fit_predict, cache_key="physics_fit_v1"))
    except NotImplementedError:
        print("physics_fit: unavailable (not implemented); omitted")
    return preds


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True, help="$CS282A_DATA/datasets/vN")
    ap.add_argument("--out-dir", help="new results dir; default: next $CS282A_DATA/results/vN")
    ap.add_argument("--experiments", default="e1,e2,e3,e4")
    ap.add_argument("--model-loader", help="module:function returning (model, norm); omit for references only")
    ap.add_argument("--reuse-cache", action="append", default=[], help="earlier results/vN whose ref_cache is read (never written)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ds = Path(args.dataset_dir)
    meta = json.loads((ds / "meta.json").read_text())
    var_d = train_d_variance(meta)
    out = Path(args.out_dir) if args.out_dir else next_results_version(os.environ["CS282A_DATA"])
    out.mkdir(parents=True, exist_ok=True)
    cache = RefCache(out, args.reuse_cache)
    preds = reference_predictors()
    bayes = bayes_predictor_or_none()
    if bayes is None:
        print("bayes: unavailable")
    if args.model_loader:
        mod, fn = args.model_loader.split(":")
        model, norm = getattr(import_module(mod), fn)()
        preds = [ModelPredictor(model, norm)] + preds
    else:
        print("no model supplied: references only")
    wanted = set(args.experiments.split(","))
    kw = dict(bayes=bayes, cache=cache, seed=args.seed)
    if "e1" in wanted:
        ex.run_e1(prefix_cases(load_npz_rows(ds / "e1.npz"), ex.E1_CS), preds, out, var_d, **kw)
    if "e2" in wanted:
        ex.run_e2(rows_as_cases(load_npz_rows(ds / "e2.npz")), preds, out, var_d, **kw)
    if "e3" in wanted:
        ex.run_e3(rows_as_cases(load_npz_rows(ds / "e3.npz")), preds, out, var_d, **kw)
    if "e4" in wanted:
        id_cases = prefix_cases(load_npz_rows(ds / "e1.npz"), ex.E4_CS)
        ood = prefix_cases(load_npz_rows(ds / "e4.npz"), ex.E4_CS)
        ex.run_e4(id_cases, ood, preds, out, var_d, **kw)
    print(f"results written to {out}")
    return out
