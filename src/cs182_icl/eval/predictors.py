"""Predictor protocol and wrappers: any method runs over the same Cases.

A predictor returns types.Prediction per Case. Reference predictors (kind "reference") are
deterministic, are deduplicated by measurement_id before prediction (INTERFACES: copies carry
the same ID) and may be cached. The model (kind "model") sees the full sequence, copies included.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import replace
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

import numpy as np

from cs182_icl.eval.cases import Case
from cs182_icl.physics import analytic_displacement
from cs182_icl.types import Action, Observation, Prediction


@runtime_checkable
class Predictor(Protocol):
    name: str
    kind: str  # "model" or "reference"

    def cache_key(self) -> str | None:
        """Stable key for the reference cache, or None if results must not be cached."""

    def predict(self, case: Case) -> Prediction: ...


def dedupe_by_measurement_id(context: Sequence[Observation]) -> list[Observation]:
    """Keep the first observation of every measurement_id (padding id -1 dropped)."""
    seen: dict[int, Observation] = {}
    out = []
    for o in context:
        if o.measurement_id == -1:
            continue
        if o.measurement_id in seen:
            first = seen[o.measurement_id]
            if (first.action, first.d_obs) != (o.action, o.d_obs):
                raise ValueError(
                    f"measurement_id {o.measurement_id} repeated with a different action/d_obs: "
                    "copies must be exact copies"
                )
            continue
        seen[o.measurement_id] = o
        out.append(o)
    return out


def predict_all(predictor: Predictor, cases: Sequence[Case]) -> list[Prediction]:
    """Use predictor.predict_batch if present, else loop predict()."""
    if hasattr(predictor, "predict_batch"):
        preds = predictor.predict_batch(list(cases))
    else:
        preds = [predictor.predict(cs) for cs in cases]
    if len(preds) != len(cases):
        raise ValueError(f"{predictor.name}: {len(preds)} predictions for {len(cases)} cases")
    for p in preds:
        if p.status == "ok" and (p.d_pred is None or not np.isfinite(p.d_pred)):
            raise ValueError(f"{predictor.name}: status ok with non-finite d_pred")
    return preds


class OraclePredictor:
    """Returns the clean target itself. Harness sanity (error is exactly 0); the real
    simulator oracle (T03) re-simulates and should be wrapped with FunctionPredictor."""

    name = "oracle"
    kind = "reference"

    def cache_key(self):
        return None

    def predict(self, case: Case) -> Prediction:
        return Prediction(case.item.d_query_clean, "ok")


class AnalyticTruePredictor:
    """Analytic closed form with the true (m, mu) of the cube; ignores the context."""

    name = "analytic_true"
    kind = "reference"

    def cache_key(self):
        return "analytic_v1"

    def predict(self, case: Case) -> Prediction:
        it = case.item
        d = analytic_displacement(it.cube.m, it.cube.mu, it.query.F, it.query.T)
        return Prediction(float(d), "ok")


class FunctionPredictor:
    """Wrap fn(context, query) -> Prediction (physics_fit_predict, bayes_predict, ...)."""

    kind = "reference"

    def __init__(
        self,
        name: str,
        fn: Callable[[list[Observation], Action], Prediction],
        *,
        cache_key: str | None = None,
        dedupe: bool = True,
    ):
        self.name, self._fn, self._key, self._dedupe = name, fn, cache_key, dedupe

    def cache_key(self):
        return self._key

    def predict(self, case: Case) -> Prediction:
        ctx = list(case.item.context)
        if self._dedupe:
            ctx = dedupe_by_measurement_id(ctx)
        return self._fn(ctx, case.item.query)


class ModelPredictor:
    """Wrap T05's model_predict(model, items, norm) -> meters array. Sees the full context."""

    kind = "model"

    def __init__(self, model: Any, norm: dict, *, name: str = "transformer", batch_size: int = 512):
        self.model, self.norm, self.name, self.batch_size = model, norm, name, batch_size

    def cache_key(self):
        return None  # never cache model outputs here (checkpoint identity is T07's job)

    def predict_batch(self, cases: Sequence[Case]) -> list[Prediction]:
        from cs182_icl.model.predict import model_predict

        out: list[Prediction] = []
        for i in range(0, len(cases), self.batch_size):
            chunk = cases[i : i + self.batch_size]
            # mask clean values: the model input must never contain d_clean or the query target
            items = [
                replace(cs.item, d_query_clean=float("nan"),
                        context=[replace(o, d_clean=float("nan")) for o in cs.item.context])
                for cs in chunk
            ]
            arr = np.asarray(model_predict(self.model, items, self.norm))
            if arr.shape != (len(chunk),) or not np.all(np.isfinite(arr)):
                raise ValueError("model_predict returned wrong shape or non-finite values")
            out += [Prediction(float(v), "ok") for v in arr]
        return out

    def predict(self, case: Case) -> Prediction:
        return self.predict_batch([case])[0]


def bayes_status() -> tuple[FunctionPredictor | None, str]:
    """(predictor or None, reason). Reason is "" when available, else why Bayes is blocked.

    Supply one with env CS282A_BAYES_PREDICTOR="module:attr" (attr is a Predictor object or a
    fn(context, query)). Otherwise cs182_icl.reference.bayes.bayes_predict is probed; if it
    returns status "blocked" (or is unimplemented) Bayes is unavailable. Never raises.
    """
    spec = os.environ.get("CS282A_BAYES_PREDICTOR")
    if spec:
        mod, attr = spec.split(":")
        obj = getattr(import_module(mod), attr)
        if isinstance(obj, Predictor):
            return obj, ""  # type: ignore[return-value]
        return FunctionPredictor("bayes_train", obj, cache_key=f"env:{spec}"), ""
    try:
        from cs182_icl.reference.bayes import bayes_predict

        probe = bayes_predict(
            [
                Observation(Action(15.0, 200), 1, 0.01, 0.01),
                Observation(Action(20.0, 200), 2, 0.05, 0.05),
            ],
            Action(18.0, 200),
        )
    except NotImplementedError:
        return None, "bayes_predict not implemented"
    if probe.status == "blocked":
        return None, str(probe.reason or "blocked")
    return FunctionPredictor("bayes_train", bayes_predict, cache_key="bayes_predict_default"), ""


def bayes_predictor_or_none() -> FunctionPredictor | None:
    """The Bayes predictor if one is valid, else None ("bayes: unavailable")."""
    return bayes_status()[0]
