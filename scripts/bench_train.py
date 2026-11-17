"""Benchmark training speed on synthetic rows (SYNTHETIC; timing only, no checkpoints).

python scripts/bench_train.py [--steps 300]
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cs182_icl.model.gpt2 import ModelConfig, PushGPT, count_params  # noqa: E402
from cs182_icl.model.tokens import compute_norm  # noqa: E402
from cs182_icl.train.data import RowSampler, make_synthetic  # noqa: E402
from cs182_icl.train.loop import batch_loss, lr_at  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    data = make_synthetic(1000, seed=0)
    norm = compute_norm(data["F"], data["k"], data["d_obs"])
    s = RowSampler(data, dev, seed=0)
    model = PushGPT(ModelConfig()).to(dev)
    print(f"parameters: {count_params(model):,}  device: {dev} (SYNTHETIC rows, benchmark only)")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    def step(i):
        for g in opt.param_groups:
            g["lr"] = lr_at(i, args.steps, 20, 1e-4)
        loss = batch_loss(model, norm, s.sample(args.batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        return loss

    for i in range(20):
        step(i)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for i in range(args.steps):
        loss = step(i)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    sps = args.steps / dt
    print(f"steps/s: {sps:.2f}  tokens/s: {sps * args.batch * 22:,.0f}  final loss: {float(loss.detach()):.4f}")
    if dev.type == "cuda":
        print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
    print(f"projected minutes per 1000 steps: {1000 / sps / 60:.2f}")


if __name__ == "__main__":
    main()
