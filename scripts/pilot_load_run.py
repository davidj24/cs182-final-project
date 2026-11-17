"""Control-authored glue (pilot): load a trained run for evaluation and verify provenance (PILOT_SPEC §O7).

  eval loader : CS282A_RUN_DIR=<run dir> [CS282A_CKPT=best.pt] python scripts/eval_run.py \
                    --dataset-dir <ds> --model-loader pilot_load_run:load      (needs PYTHONPATH incl. scripts/)
  verification: python scripts/pilot_load_run.py --run-dir <run dir> [--ckpt best.pt]
Checks: checkpoint sha256 printed; checkpoint norm == run meta.json norm == dataset meta.json norm (train only);
dataset file sha256 in config.json == files on disk; saved best_val reproduced by re-evaluating the checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cs182_icl.model.gpt2 import ModelConfig, PushGPT  # noqa: E402
from cs182_icl.train.data import RowSampler, file_sha256, load_dataset  # noqa: E402
from cs182_icl.train.loop import eval_val  # noqa: E402


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _same(a, b) -> bool:
    return json.dumps(a, sort_keys=True, default=str) == json.dumps(b, sort_keys=True, default=str)


def load_checked(run_dir: Path, ckpt_name: str = "best.pt", device: str = "cpu"):
    ck = torch.load(run_dir / ckpt_name, map_location=device, weights_only=False)
    cfg = json.loads((run_dir / "config.json").read_text())
    meta = json.loads((run_dir / "meta.json").read_text())
    ds_dir = Path(cfg["dataset"])
    ds_meta = json.loads((ds_dir / "meta.json").read_text())
    norm = ck["norm"]
    assert _same(norm, meta["normalization"]), "checkpoint norm != run meta.json norm"
    assert _same(norm, ds_meta["normalization"]), "checkpoint norm != dataset meta.json norm"
    for name, h in cfg["dataset_files_sha256"].items():
        fn = name if name.endswith(".npz") else name + ".npz"
        assert file_sha256(ds_dir / fn) == h, f"dataset file changed since training: {fn}"
    model = PushGPT(ModelConfig(**ck["model_cfg"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    info = {"run_dir": str(run_dir), "ckpt": ckpt_name, "ckpt_sha256": _sha(run_dir / ckpt_name), "step": ck["step"],
            "source_git": cfg["source_git"], "dataset": str(ds_dir), "dataset_version": cfg.get("dataset_version"),
            "config_sha256": cfg["cmd_config_sha256"], "steps_planned": cfg["steps"], "seed": cfg["seed"], "noise": cfg["noise"]}
    return model, norm, info, ds_meta


def load():
    """eval_run.py --model-loader hook: returns (model, norm); prints the provenance record (and writes it if CS282A_PROVENANCE_OUT is set)."""
    run = Path(os.environ["CS282A_RUN_DIR"])
    model, norm, info, _ = load_checked(run, os.environ.get("CS282A_CKPT", "best.pt"))
    print("MODEL PROVENANCE " + json.dumps(info, default=str))
    out = os.environ.get("CS282A_PROVENANCE_OUT")
    if out:
        Path(out).write_text(json.dumps(info, indent=2, default=str))
    return model, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    a = ap.parse_args()
    run = Path(a.run_dir)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, norm, info, ds_meta = load_checked(run, a.ckpt, dev)
    print(json.dumps(info, indent=2, default=str))
    ds = load_dataset(info["dataset"])
    assert _same(norm, ds["norm"]), "checkpoint norm != norm recomputed from the dataset train split"
    val_s = RowSampler(ds["val"], torch.device(dev), seed=0)
    v = eval_val(model, norm, val_s, *ds["val_rows"])
    rows = [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
    at = [r["val_loss"] for r in rows if r.get("type") == "val" and r["step"] == info["step"]]
    print(f"re-evaluated val loss: {v:.8f}; logged at step {info['step']}: {at}")
    if a.ckpt == "best.pt" and at:
        assert abs(v - at[0]) < 1e-6, "checkpoint does not reproduce its logged val loss"
        vals = [(r["val_loss"], r["step"]) for r in rows if r.get("type") == "val"]
        assert info["step"] == min(vals)[1], "best.pt step is not the argmin of the logged val loss"
    print("VERIFY OK")


if __name__ == "__main__":
    main()
