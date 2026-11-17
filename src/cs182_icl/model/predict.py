"""model_predict signature (owner: T05 replaces the body; keep the signature)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cs182_icl.types import TestItem


def model_predict(model: Any, items: Sequence[TestItem], norm: dict) -> np.ndarray:
    """Transformer query predictions in meters, float64 array of shape (len(items),).

    Encodes each item's context then its query push (PROTOCOL §5 tokens, train-only `norm`),
    reads the prediction at the query push token, and de-standardizes it.
    """
    raise NotImplementedError("T05: model_predict")
