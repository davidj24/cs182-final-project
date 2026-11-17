"""npz writing, hashing, train-only normalization and git provenance (PILOT_SPEC H2)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from cs182_icl.types import DT

ASINH_SCALE = 0.01  # same constant as model/tokens.py (PROTOCOL 5)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_sha256(arrays: dict) -> str:
    """Deterministic hash of array names, dtypes, shapes and bytes (npz zip timestamps differ)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        a = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def compute_normalization(F, k, d_obs) -> dict:
    """mean/std of F, T, d_obs, asinh(d_obs/0.01) over ALL pushes of the TRAIN split.
    Same format and values as model.tokens.compute_norm (tested for equality)."""
    d = np.asarray(d_obs, np.float64)
    vals = {
        "F": np.asarray(F, np.float64),
        "T": np.asarray(k, np.float64) * DT,
        "d_obs": d,
        "asinh_d_obs": np.arcsinh(d / ASINH_SCALE),
    }
    return {n: {"mean": float(v.mean()), "std": float(v.std())} for n, v in vals.items()}


def git_provenance(repo: Path, require_clean: bool = True) -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()

    sha = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty and require_clean:
        raise RuntimeError("git tree is not clean (commit first); refusing to record a source SHA:\n" + dirty)
    return {"source_git_sha": sha, "source_tree_clean": not dirty}
