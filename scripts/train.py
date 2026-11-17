"""Train the pilot transformer on ONE GPU (honours CUDA_VISIBLE_DEVICES).

python scripts/train.py --dataset $CS282A_DATA/datasets/vN --steps N --seed 0 [--max-seconds S]
Writes $CS282A_DATA/runs/<run_id>/ (never overwrites; a new id is used if it exists).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from cs182_icl.model.gpt2 import ModelConfig, PushGPT, count_params  # noqa: E402
from cs182_icl.train.data import RowSampler, load_dataset, make_synthetic  # noqa: E402
from cs182_icl.train.loop import git_info, new_run_dir, train, warmup_steps  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--noise", type=float, default=0.0, help="pilot: only 0 is supported")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--runs-root", default=None)
    args = ap.parse_args()
    if args.noise != 0.0:
        sys.exit("noise > 0 not implemented in the pilot")

    ds = load_dataset(args.dataset)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = PushGPT(ModelConfig()).to(dev)
    n_params = count_params(model)
    print(f"parameters: {n_params:,}  device: {dev}  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    git = git_info(REPO)
    wu = warmup_steps(args.steps)
    config = {"dataset": str(Path(args.dataset).resolve()), "dataset_version": Path(args.dataset).name,
              "dataset_files_sha256": ds["files"], "dataset_meta": ds["meta"].get("version"),
              "steps": args.steps, "seed": args.seed, "noise": args.noise, "batch": args.batch,
              "lr": args.lr, "weight_decay": args.wd, "warmup_steps": wu, "schedule": "linear warmup + cosine to 0",
              "eval_every": args.eval_every, "max_seconds": args.max_seconds, "model": model.cfg.to_dict(),
              "params": n_params, "source_git": git, "argv": sys.argv,
              "cmd_config_sha256": None}
    config["cmd_config_sha256"] = hashlib.sha256(json.dumps(
        {k: v for k, v in config.items() if k not in ("argv", "cmd_config_sha256")}, sort_keys=True,
        default=str).encode()).hexdigest()
    root = Path(args.runs_root or os.path.join(os.environ["CS282A_DATA"], "runs"))
    rid = args.run_id or f"{Path(args.dataset).name}_n{args.noise:g}_s{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir, rid = new_run_dir(root, rid)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (run_dir / "meta.json").write_text(json.dumps({"normalization": ds["norm"], "run_id": rid,
                                                   "normalization_source": "train.npz only"}, indent=2))
    print("run dir:", run_dir)

    train_s = RowSampler(ds["train"], dev, seed=args.seed)
    val_s = RowSampler(ds["val"], dev, seed=0)
    recs = train(model, ds["norm"], train_s, val_s, ds["val_rows"], steps=args.steps, lr=args.lr, wd=args.wd,
                 batch=args.batch, warmup=wu, eval_every=args.eval_every, max_seconds=args.max_seconds,
                 run_dir=run_dir, config=config)
    print(json.dumps(recs[-1]))


if __name__ == "__main__":
    main()
