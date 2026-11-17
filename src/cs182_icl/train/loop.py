"""Training loop: AdamW, linear warmup, cosine over the actual budget, best/last checkpoints."""

from __future__ import annotations

import json
import math
import subprocess
import time
from pathlib import Path

import torch

from cs182_icl.model.gpt2 import ModelConfig, PushGPT, count_params
from cs182_icl.model.tokens import encode, standardize_target
from cs182_icl.train.data import RowSampler


def lr_at(step: int, steps: int, warmup: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    prog = (step - warmup) / max(1, steps - warmup)
    return 0.5 * lr * (1 + math.cos(math.pi * min(1.0, prog)))


def warmup_steps(steps: int) -> int:
    return min(2000, max(1, round(0.05 * steps)))


def git_info(repo: Path) -> dict:
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()
    return {"sha": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def new_run_dir(root: Path, run_id: str) -> tuple[Path, str]:
    root.mkdir(parents=True, exist_ok=True)
    cand, i = run_id, 0
    while True:
        try:
            (root / cand).mkdir()
            return root / cand, cand
        except FileExistsError:
            i += 1
            cand = f"{run_id}_{i}"


def batch_loss(model, norm, sampler_out) -> torch.Tensor:
    F, T, d_obs, d_clean = sampler_out
    tok = encode(norm, F.float(), T.float(), d_obs.float())
    y = standardize_target(norm, d_clean).float()
    return ((model.push_predictions(tok) - y) ** 2).mean()


@torch.no_grad()
def eval_val(model, norm, val_sampler: RowSampler, row_cube, row_idx, bs: int = 500) -> float:
    model.eval()
    tot, n = 0.0, 0
    dev = val_sampler.device
    rc = torch.as_tensor(row_cube, device=dev)
    ri = torch.as_tensor(row_idx, device=dev)
    for s in range(0, len(rc), bs):
        loss = batch_loss(model, norm, val_sampler.gather(rc[s : s + bs], ri[s : s + bs]))
        tot += loss.item() * len(rc[s : s + bs])
        n += len(rc[s : s + bs])
    model.train()
    return tot / n


def train(model, norm, train_s: RowSampler, val_s: RowSampler | None, val_rows, *, steps, lr=1e-4,
          wd=0.01, batch=256, warmup=None, eval_every=250, log_every=50, max_seconds=None,
          run_dir: Path | None = None, config: dict | None = None, seed_torch: int = 0):
    """Returns the list of metric records. Writes metrics.jsonl / best.pt / last.pt if run_dir."""
    dev = train_s.device
    warmup = warmup_steps(steps) if warmup is None else warmup
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    use_cuda = dev.type == "cuda"
    records, best = [], float("inf")
    mf = open(run_dir / "metrics.jsonl", "a") if run_dir else None

    def emit(rec):
        records.append(rec)
        if mf:
            mf.write(json.dumps(rec) + "\n")
            mf.flush()

    def ckpt(name, step):
        if run_dir:
            torch.save({"model": model.state_dict(), "model_cfg": model.cfg.to_dict(), "norm": norm,
                        "step": step, "config": config}, run_dir / name)

    t0 = time.time()
    gpu_s, win_loss, win_n, win_t, win_tok = 0.0, 0.0, 0, time.time(), 0
    ev_s = torch.cuda.Event(enable_timing=True) if use_cuda else None
    ev_e = torch.cuda.Event(enable_timing=True) if use_cuda else None
    last_step = 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, warmup, lr)
        if use_cuda and win_n == 0:
            ev_s.record()
        loss = batch_loss(model, norm, train_s.sample(batch))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        last_step = step + 1
        win_loss += loss.detach(); win_n += 1; win_tok += batch * 22
        if last_step % log_every == 0 or last_step == steps:
            if use_cuda:
                ev_e.record()
                torch.cuda.synchronize()
                gpu_s += ev_s.elapsed_time(ev_e) / 1000  # GPU-timeline seconds of training windows
            wl = float(win_loss) / win_n
            if not math.isfinite(wl):
                raise FloatingPointError(f"non-finite train loss at step {last_step}")
            now = time.time()
            emit({"type": "train", "step": last_step, "train_loss": wl, "lr": opt.param_groups[0]["lr"],
                  "grad_norm": float(gn), "tokens_per_s": win_tok / (now - win_t),
                  "wall_s": now - t0, "gpu_s": gpu_s})
            win_loss, win_n, win_t, win_tok = 0.0, 0, now, 0
        stop = max_seconds is not None and time.time() - t0 > max_seconds
        if val_s is not None and (last_step % eval_every == 0 or last_step == steps or stop):
            v = eval_val(model, norm, val_s, *val_rows)
            emit({"type": "val", "step": last_step, "val_loss": v, "wall_s": time.time() - t0})
            if v < best:
                best = v
                ckpt("best.pt", last_step)
        if stop:
            emit({"type": "stopped_early", "step": last_step, "reason": f"max_seconds={max_seconds}",
                  "wall_s": time.time() - t0})
            break
    if use_cuda:
        torch.cuda.synchronize()
    ckpt("last.pt", last_step)
    emit({"type": "final", "step": last_step, "wall_s": time.time() - t0, "gpu_s": gpu_s, "best_val": best,
          "params": count_params(model)})
    if mf:
        mf.close()
    return records
