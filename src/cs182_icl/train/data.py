"""Datasets, row sampler, and synthetic rows for development.

Real data: $CS282A_DATA/datasets/vN/{train,val}.npz (docs/INTERFACES.md), built by T04.
SYNTHETIC data (make_synthetic) comes from physics.analytic_displacement only, exists to
test plumbing, and must never be used for the real run (train.py refuses it).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from cs182_icl import physics
from cs182_icl.model.gpt2 import N_PUSHES
from cs182_icl.model.tokens import compute_norm
from cs182_icl.types import DT

P = 32  # pushes per cube


def make_synthetic(n_cubes: int, seed: int, id_offset: int = 0) -> dict:
    """SYNTHETIC rows (analytic physics, noise 0) in the train.npz/val.npz layout."""
    rng = np.random.default_rng(seed)
    m = np.exp(rng.uniform(np.log(0.5), np.log(1.5), n_cubes))
    mu = rng.uniform(0.3, 0.6, n_cubes)
    F = rng.uniform(13, 26, (n_cubes, P))
    k = rng.integers(100, 201, (n_cubes, P))
    d = physics.analytic_displacement(m[:, None], mu[:, None], F, k * DT)
    return {
        "cube_id": np.arange(n_cubes, dtype=np.int64) + id_offset,
        "m": m, "mu": mu, "F": F, "k": k.astype(np.int64),
        "measurement_id": np.arange(n_cubes * P, dtype=np.int64).reshape(n_cubes, P),
        "d_clean": d, "d_obs": d.copy(),
        "synthetic": np.array(True),
    }


def fixed_rows(n_cubes: int, n_rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic (row_cube, row_idx) for validation when the dataset has none."""
    rng = np.random.default_rng(seed)
    row_cube = rng.integers(0, n_cubes, n_rows)
    row_idx = np.stack([rng.permutation(P)[:N_PUSHES] for _ in range(n_rows)])
    return row_cube.astype(np.int64), row_idx.astype(np.int64)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(path: str | Path) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


class RowSampler:
    """Samples rows of 11 distinct pushes (random order) from each cube's 32-push pool.

    Tensors live on `device`; randomness comes from an explicit seeded torch.Generator.
    Targets are the CLEAN displacement; inputs use d_obs (identical at noise 0).
    """

    def __init__(self, split: dict, device, seed: int):
        t = lambda a, dt: torch.as_tensor(np.asarray(a), dtype=dt, device=device)
        self.F = t(split["F"], torch.float64)
        self.T = t(split["k"], torch.float64) * DT
        self.d_obs = t(split["d_obs"], torch.float64)
        self.d_clean = t(split["d_clean"], torch.float64)
        self.n = self.F.shape[0]
        self.device = device
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def sample(self, batch: int) -> tuple[torch.Tensor, ...]:
        cube = torch.randint(0, self.n, (batch,), generator=self.gen, device=self.device)
        perm = torch.rand(batch, P, generator=self.gen, device=self.device).argsort(dim=1)[:, :N_PUSHES]
        return self.gather(cube, perm)

    def gather(self, cube, idx):
        g = lambda a: a[cube[:, None], idx]
        return g(self.F), g(self.T), g(self.d_obs), g(self.d_clean)


def load_dataset(dataset_dir: str | Path, allow_synthetic: bool = False) -> dict:
    d = Path(dataset_dir)
    train, val = load_split(d / "train.npz"), load_split(d / "val.npz")
    if "synthetic" in train and not allow_synthetic:
        raise RuntimeError("synthetic dataset refused for a real run")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    if "row_idx" in val:
        row_cube, row_idx = val["row_cube"], val["row_idx"]
    else:
        row_cube, row_idx = fixed_rows(val["F"].shape[0], 1000, seed=12345)
    # normalization: train split only
    norm = compute_norm(train["F"], train["k"], train["d_obs"])
    return {"train": train, "val": val, "val_rows": (row_cube, row_idx), "norm": norm, "meta": meta,
            "files": {n: file_sha256(d / f"{n}.npz") for n in ("train", "val")}}
