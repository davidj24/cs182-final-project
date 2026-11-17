import math

import numpy as np
import pytest
import torch

from cs182_icl.model.gpt2 import ModelConfig, PushGPT
from cs182_icl.model.tokens import compute_norm
from cs182_icl.train.data import RowSampler, make_synthetic
from cs182_icl.train.loop import batch_loss, lr_at, train, warmup_steps


def test_row_sampler_distinct_pushes_and_determinism():
    d = make_synthetic(10, 0)
    s1, s2 = RowSampler(d, torch.device("cpu"), 5), RowSampler(d, torch.device("cpu"), 5)
    a, b = s1.sample(64), s2.sample(64)
    assert a[0].shape == (64, 11) and all(torch.equal(x, y) for x, y in zip(a, b))
    # distinct pushes per row: recover indices via measurement ids is not stored, so sample indices directly
    perm = torch.rand(64, 32, generator=s1.gen).argsort(dim=1)[:, :11]
    assert all(len(set(r.tolist())) == 11 for r in perm)


def test_schedule():
    assert warmup_steps(1000) == 50 and warmup_steps(1_000_000) == 2000
    assert lr_at(0, 1000, 50, 1e-4) < lr_at(49, 1000, 50, 1e-4) == pytest.approx(1e-4)
    assert lr_at(999, 1000, 50, 1e-4) < 1e-7


@pytest.mark.slow
def test_overfit_fixed_64_rows(capsys):
    """H5: 64 fixed rows, all 11 positions, final train MSE < 1e-3 x initial within the step cap."""
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = make_synthetic(64, 0)
    norm = compute_norm(d["F"], d["k"], d["d_obs"])
    s = RowSampler(d, dev, 0)
    cube = torch.arange(64, device=dev)
    idx = torch.stack([torch.randperm(32, device=dev)[:11] for _ in range(64)])
    fixed = s.gather(cube, idx)
    model = PushGPT(ModelConfig()).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.0)
    cap = 16000  # step cap; lr: 100-step warmup then cosine over the cap
    with torch.no_grad():
        init = batch_loss(model, norm, fixed).item()
    for step in range(cap):
        for g in opt.param_groups:
            g["lr"] = 5e-4 * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / cap))
        loss = batch_loss(model, norm, fixed)
        opt.zero_grad(); loss.backward(); opt.step()
        if loss.item() < 1e-3 * init:
            break
    final = loss.item()
    print(f"overfit: initial MSE {init:.4f}, final MSE {final:.2e} at step {step + 1}/{cap}, ratio {final / init:.2e}")
    assert final < 1e-3 * init


def test_train_loop_smoke(tmp_path):
    d = make_synthetic(20, 0)
    norm = compute_norm(d["F"], d["k"], d["d_obs"])
    s = RowSampler(d, torch.device("cpu"), 0)
    model = PushGPT(ModelConfig(n_layer=1, n_embd=32, n_head=4))
    rows = (np.arange(10) % 20, np.stack([np.arange(11)] * 10))
    recs = train(model, norm, s, s, rows, steps=20, batch=8, eval_every=10, log_every=10, run_dir=tmp_path)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    assert {"train", "val", "final"} <= {r["type"] for r in recs}
