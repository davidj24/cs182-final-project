"""model_predict (owner: T05)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from cs182_icl.model.gpt2 import N_PUSHES
from cs182_icl.model.tokens import destandardize, encode
from cs182_icl.types import TestItem


@torch.no_grad()
def model_predict(model: Any, items: Sequence[TestItem], norm: dict, batch_size: int = 512) -> np.ndarray:
    """Transformer query predictions in meters, float64 array of shape (len(items),).

    Encodes each item's context then its query push (PROTOCOL §5 tokens, train-only `norm`),
    reads the prediction at the query push token (position 2c), and de-standardizes it.
    The query outcome slot and everything after the query is zero padding; by causality it
    cannot influence the readout at the query push token.
    """
    was_training = model.training
    model.eval()
    dev = next(model.parameters()).device
    out = np.zeros(len(items), np.float64)
    for s in range(0, len(items), batch_size):
        chunk = items[s : s + batch_size]
        n = len(chunk)
        Fm = np.zeros((n, N_PUSHES))
        Tm = np.zeros((n, N_PUSHES))
        dm = np.zeros((n, N_PUSHES))
        pos = np.zeros(n, np.int64)
        for i, it in enumerate(chunk):
            c = len(it.context)
            if c > N_PUSHES - 1:
                raise ValueError(f"context length {c} > {N_PUSHES - 1}")
            for j, ob in enumerate(it.context):
                Fm[i, j], Tm[i, j], dm[i, j] = ob.action.F, ob.action.T, ob.d_obs
            Fm[i, c], Tm[i, c] = it.query.F, it.query.T  # query outcome stays 0 (invisible anyway)
            pos[i] = 2 * c
        tok = encode(
            norm,
            torch.tensor(Fm, device=dev),
            torch.tensor(Tm, device=dev),
            torch.tensor(dm, device=dev),
        )
        # zero the (padding) tokens that come after the query push so they are truly blank
        keep = torch.arange(2 * N_PUSHES, device=dev)[None, :] <= torch.tensor(pos, device=dev)[:, None]
        # but outcome tokens inside the context must be kept: they have index < pos, so keep covers them
        tok = tok * keep[..., None]
        # padded push tokens beyond query are removed by `keep`; query push token itself is kept
        pred = model(tok.float()).double().cpu().numpy()
        y = pred[np.arange(n), pos]
        out[s : s + n] = destandardize(norm, y)
    model.train(was_training)
    return out
