"""Command line entry for the evaluation harness (scripts/eval_run.py calls main)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from importlib import import_module
from pathlib import Path

from cs182_icl.eval import experiments as ex
from cs182_icl.eval.cache import RefCache, next_results_version
from cs182_icl.eval.cases import load_npz_rows, prefix_cases, rows_as_cases
from cs182_icl.eval.metrics import train_d_variance_with_key
from cs182_icl.eval.predictors import (
    AnalyticTruePredictor,
    FunctionPredictor,
    ModelPredictor,
    OraclePredictor,
    bayes_status,
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


def _find_hashes(meta: dict) -> dict:
    """{filename: sha256} from any meta.json key containing 'sha256' that maps names to hashes."""
    for k, v in meta.items():
        if "sha256" in k.lower() and isinstance(v, dict):
            flat = {kk: vv for kk, vv in v.items() if isinstance(vv, str)}
            if flat:
                return flat
    return {}


def provenance(ds: Path, meta: dict, var_key: str, model_loader: str | None, files) -> list[str]:
    """Header lines: dataset version/source SHA, verified npz hashes (raises on mismatch)."""
    hashes = _find_hashes(meta)
    lines = [
        f"dataset: {ds} (meta version = {meta.get('version', 'n/a')}; source git SHA = "
        f"{next((meta[k] for k in ('source_git_sha', 'git_sha', 'source_sha', 'source_commit') if k in meta), 'n/a')})",
        f"Var(d) read from meta.json key {var_key}",
        f"model: {model_loader or 'none (references only)'}",
    ]
    checked = []
    for f in files:
        want = hashes.get(f) or hashes.get(Path(f).stem)
        if want is None:
            continue
        got = hashlib.sha256((ds / f).read_bytes()).hexdigest()
        if got != want:
            raise RuntimeError(f"{f}: sha256 {got} != meta.json {want}; refusing to evaluate")
        checked.append(f)
    lines.append("npz sha256 verified against meta.json: " + (", ".join(checked) or
                 "NOT VERIFIED (no per-file hash found in meta.json)"))
    return lines


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
    var_d, var_key = train_d_variance_with_key(meta)
    if args.out_dir:
        out = Path(args.out_dir)
        if out.exists():
            raise SystemExit(f"--out-dir {out} already exists; results dirs are write-once")
        out.mkdir(parents=True)
    else:
        out = next_results_version(os.environ["CS282A_DATA"])
    wanted = set(args.experiments.split(","))
    prov = provenance(ds, meta, var_key, args.model_loader, sorted({f"{e}.npz" for e in wanted} | ({"e1.npz"} if "e4" in wanted else set())))
    cache = RefCache(out, args.reuse_cache)
    preds = reference_predictors()
    bayes, why = bayes_status()
    if bayes is None:
        print(f"bayes: unavailable ({why})")
    if args.model_loader:
        mod, fn = args.model_loader.split(":")
        model, norm = getattr(import_module(mod), fn)()
        preds = [ModelPredictor(model, norm)] + preds
    else:
        print("no model supplied: references only")
    kw = dict(bayes=bayes, cache=cache, seed=args.seed, header_extra=prov)
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
