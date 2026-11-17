"""Bayes-as-model harness test. Collected only when a Bayes predictor is supplied
(see tests/eval/conftest.py); otherwise this file is not collected ("bayes unavailable")."""

import pytest

from cs182_icl.eval import experiments as ex
from cs182_icl.eval.cache import RefCache
from cs182_icl.eval.metrics import bayes_denominator, evaluate_all, metric_table
from cs182_icl.eval.cases import prefix_cases
from tests.eval.conftest import BAYES_PREDICTOR as BAYES
from tests.eval.synth import make_e1_rows


def test_bayes_as_model_reproduces_bayes_metrics_exactly(tmp_path):
    cases = prefix_cases(make_e1_rows(6), ex.E1_CS)  # synthetic items
    direct = [BAYES.predict(c) for c in cases]
    sc = evaluate_all(cases, [BAYES], RefCache(tmp_path))[BAYES.name]
    for p, d_pred, st in zip(direct, sc.d_pred, sc.status):
        assert st == p.status
        if p.status == "ok":
            assert d_pred == p.d_pred
    D = bayes_denominator(
        evaluate_all([c for c in cases if c.c == 0], [BAYES])[BAYES.name]
    )
    if D is not None:
        rows = metric_table({BAYES.name: sc}, 1.0, bayes=sc, bayes_D=D)
        r0 = next(r for r in rows if r["c"] == 0)
        assert r0["nmse_bayes_c0"] == pytest.approx(1.0)
        assert all(float(r["gap_to_bayes_m2"]) == 0.0 for r in rows if r["gap_to_bayes_m2"] != "")
