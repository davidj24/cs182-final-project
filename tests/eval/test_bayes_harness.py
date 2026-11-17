"""Bayes-as-model harness tests (always collected).

1. A synthetic stub Bayes (deterministic, context-dependent) is run both as a reference and
   wrapped as kind="model" through RefCache and the same metric path: metric rows must be identical.
2. The real Bayes variant skips visibly (pytest.skip with the reason) when Bayes is blocked.
"""

import pytest

from cs182_icl.eval import experiments as ex
from cs182_icl.eval.cache import RefCache
from cs182_icl.eval.cases import prefix_cases
from cs182_icl.eval.metrics import bayes_denominator, evaluate_all, metric_table
from cs182_icl.eval.predictors import FunctionPredictor, dedupe_by_measurement_id
from cs182_icl.types import Prediction
from tests.eval.conftest import BAYES_PREDICTOR, BAYES_REASON
from tests.eval.synth import make_e1_rows


def stub_bayes_fn(context, query):
    ctx = dedupe_by_measurement_id(context)
    base = sum(o.d_obs for o in ctx) / len(ctx) if ctx else 0.01
    return Prediction(0.5 * base + 0.0004 * query.F, "ok")


class AsModel:
    """The same function exposed as a kind="model" predictor (cacheable, sees full context)."""

    kind = "model"

    def __init__(self, name):
        self.name = name

    def cache_key(self):
        return "stub-as-model"

    def predict(self, case):
        return stub_bayes_fn(case.item.context, case.item.query)


def _identical_rows(tmp_path, ref, model):
    cases = prefix_cases(make_e1_rows(6), ex.E1_CS)
    cache = RefCache(tmp_path)
    sc = evaluate_all(cases, [ref, model], cache)
    D = bayes_denominator(evaluate_all([c for c in cases if c.c == 0], [ref], cache)[ref.name])
    assert D is not None
    rows = metric_table(sc, 1.0, bayes=sc[ref.name], bayes_D=D)
    by = {m: [r for r in rows if r["method"] == m] for m in (ref.name, model.name)}
    assert len(by[ref.name]) == len(by[model.name]) > 0
    drop = {"method", "kind"}
    for a, b in zip(by[ref.name], by[model.name]):
        assert {k: v for k, v in a.items() if k not in drop} == {k: v for k, v in b.items() if k not in drop}
        assert float(a["gap_to_bayes_m2"]) == 0.0 and float(b["gap_to_bayes_m2"]) == 0.0
    r0 = next(r for r in by[model.name] if r["c"] == 0)
    assert r0["nmse_bayes_c0"] == pytest.approx(1.0)
    return sc, cache


def test_stub_bayes_as_model_reproduces_bayes_metrics_exactly(tmp_path):
    ref = FunctionPredictor("bayes_train", stub_bayes_fn, cache_key="stub")
    sc, cache = _identical_rows(tmp_path, ref, AsModel("bayes_as_model"))
    a, b = sc["bayes_train"], sc["bayes_as_model"]
    assert (a.d_pred == b.d_pred).all() and a.status == b.status
    assert a.kind == "reference" and b.kind == "model"
    assert len(list((tmp_path / "ref_cache").glob("bayes*.json"))) >= 1
    assert len(list((tmp_path / "ref_cache").glob("bayes_as_model*.json"))) == 1  # model went through RefCache


def test_real_bayes_as_model_reproduces_bayes_metrics_exactly(tmp_path):
    if BAYES_PREDICTOR is None:
        pytest.skip(f"bayes blocked: {BAYES_REASON}")
    model = AsModel("bayes_as_model")
    model.predict = lambda case: BAYES_PREDICTOR.predict(case)
    _identical_rows(tmp_path, BAYES_PREDICTOR, model)
