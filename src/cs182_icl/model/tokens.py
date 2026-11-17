"""PROTOCOL §5 token encoding and training-only normalization.

push    [F~, T~, 0, 0, 0]
outcome [0, 0, d~_obs, asinh~(d_obs/0.01), 1]
Row layout: push_0, outcome_0, push_1, outcome_1, ... The prediction for push j is read at
position 2j (its push token), which causal attention keeps blind to outcome_j.
Target for push j: (d_clean_j - mean_d) / std_d (raw standardized displacement).

Norm dict format (saved as `normalization` in meta.json of a run):
  {"F": {"mean","std"}, "T": {...}, "d_obs": {...}, "asinh_d_obs": {...}}
"""

from __future__ import annotations

import numpy as np
import torch

from cs182_icl.types import DT

ASINH_SCALE = 0.01
KEYS = ("F", "T", "d_obs", "asinh_d_obs")


def compute_norm(F: np.ndarray, k: np.ndarray, d_obs: np.ndarray) -> dict:
    """Mean/std over ALL pushes of the TRAIN split only (arrays any shape)."""
    vals = {
        "F": np.asarray(F, np.float64),
        "T": np.asarray(k, np.float64) * DT,
        "d_obs": np.asarray(d_obs, np.float64),
        "asinh_d_obs": np.arcsinh(np.asarray(d_obs, np.float64) / ASINH_SCALE),
    }
    return {n: {"mean": float(v.mean()), "std": float(v.std())} for n, v in vals.items()}


def encode(norm: dict, F, T, d_obs) -> torch.Tensor:
    """F, T, d_obs: tensors (B, 11) -> tokens (B, 22, 5) float32. Pure function of the inputs."""
    def z(x, key):
        return (x - norm[key]["mean"]) / norm[key]["std"]

    B, P = F.shape
    push = torch.zeros(B, P, 5, dtype=torch.float32, device=F.device)
    out = torch.zeros_like(push)
    push[..., 0] = z(F, "F")
    push[..., 1] = z(T, "T")
    out[..., 2] = z(d_obs, "d_obs")
    out[..., 3] = z(torch.asinh(d_obs / ASINH_SCALE), "asinh_d_obs")
    out[..., 4] = 1.0
    return torch.stack([push, out], dim=2).reshape(B, 2 * P, 5)


def standardize_target(norm: dict, d):
    return (d - norm["d_obs"]["mean"]) / norm["d_obs"]["std"]


def destandardize(norm: dict, y):
    return y * norm["d_obs"]["std"] + norm["d_obs"]["mean"]
