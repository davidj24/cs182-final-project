"""Harness tests on SYNTHETIC items (physics.py closed form; not MuJoCo)."""

from dataclasses import replace

import numpy as np
import pytest

from cs182_icl.eval import experiments as ex
from cs182_icl.eval import plots
from cs182_icl.eval.cache import RefCache, next_results_version
from cs182_icl.eval.cases import (
    Case, assert_prefix_paired, fingerprint, load_npz_rows, prefix_cases, rows_as_cases,
)
from cs182_icl.eval.metrics import evaluate_all, read_csv
from cs182_icl.eval.predictors import (
    AnalyticTruePredictor, FunctionPredictor, OraclePredictor, Predictor,
    bayes_predictor_or_none, dedupe_by_measurement_id,
)
from cs182_icl.types import Prediction
from tests.eval.synth import (
    make_e1_rows, make_e2_cases, make_e3_cases, stub_physics_fit, write_dataset,
)

VAR_D = 0.05**2


def refs():
    return [OraclePredictor(), AnalyticTruePredictor(),
            FunctionPredictor("physics_fit", stub_physics_fit, cache_key="stub")]


# ---- same items, prefix pairing, clean target
def test_protocol_conformance():
    for p in refs():
        assert isinstance(p, Predictor)


def test_same_items_for_all_methods():
    cases = prefix_cases(make_e1_rows(5), ex.E1_CS)
    sc = evaluate_all(cases, refs())
    keys = {tuple(s.keys) for s in sc.values()}
    assert len(keys) == 1 and len(next(iter(keys))) == 5 * len(ex.E1_CS)
    from cs182_icl.eval.metrics import assert_same_items
    other = evaluate_all(cases[:-1], [OraclePredictor()])
    sc2 = dict(sc)
    sc2["oracle_short"] = other["oracle"]
    with pytest.raises(AssertionError, match="different items"):
        assert_same_items(sc2)


def test_prefix_pairing():
    cases = prefix_cases(make_e1_rows(3), ex.E1_CS)
    assert_prefix_paired(cases)
    by = {(c.item_id, c.c): c for c in cases}
    assert by[(0, 4)].item.context[:2] == by[(0, 2)].item.context  # first c of one ordered list
    # break the prefix: swap two context entries at c=4
    bad = by[(0, 4)]
    ctx = list(bad.item.context)
    ctx[0], ctx[1] = ctx[1], ctx[0]
    broken = [replace(c, item=replace(c.item, context=ctx)) if c is bad else c for c in cases]
    with pytest.raises(AssertionError, match="not a prefix"):
        assert_prefix_paired(broken)
    # same ordered list but a different query across c
    other_q = [replace(c, item=replace(c.item, d_query_clean=c.item.d_query_clean + 1))
               if c is by[(1, 8)] else c for c in cases]
    with pytest.raises(AssertionError, match="cube/query/target"):
        assert_prefix_paired(other_q)


def test_scoring_uses_clean_target_only(tmp_path):
    rows = make_e1_rows(4)
    # corrupt every context d_obs: targets and oracle error must be unaffected
    noisy = []
    for i, it in rows:
        ctx = [replace(o, d_obs=o.d_obs + 123.0) for o in it.context]
        noisy.append((i, replace(it, context=ctx)))
    cases = prefix_cases(noisy, (0, 10))
    sc = evaluate_all(cases, [OraclePredictor()])["oracle"]
    assert np.array_equal(sc.d_true, np.array([c.item.d_query_clean for c in cases]))
    assert np.all(sc.sq_err == 0.0)
    # npz round trip keeps d_query_clean as target and d_obs only inside the context
    write_dataset(tmp_path)
    loaded = load_npz_rows(tmp_path / "e1.npz")
    orig = dict(make_e1_rows(10, 0))
    for i, it in loaded:
        assert it.d_query_clean == orig[i].d_query_clean


def test_not_determined_reported_for_physics_fit_at_c0_and_1():
    cases = prefix_cases(make_e1_rows(6), (0, 1, 2))
    sc = evaluate_all(cases, refs())["physics_fit"]
    st = {c: [s for s, cc in zip(sc.status, sc.c) if cc == c] for c in (0, 1, 2)}
    assert set(st[0]) == {"not_determined"} and set(st[1]) == {"not_determined"}
    assert st[2].count("ok") >= 5  # two random distinct forces identify the line


def test_dedupe_by_measurement_id():
    cases = make_e3_cases(2)
    big = max(cases, key=lambda c: c.c)
    assert len(dedupe_by_measurement_id(big.item.context)) == 2


# ---- cache
def test_cache_reuse_and_no_overwrite(tmp_path):
    calls = {"n": 0}

    def fn(ctx, q):
        calls["n"] += 1
        return Prediction(0.5, "ok")

    p = FunctionPredictor("f", fn, cache_key="v1", dedupe=False)
    cases = prefix_cases(make_e1_rows(3), (0, 2))
    c1 = RefCache(tmp_path)
    a = c1.get_or_compute(p, cases)
    n_first = calls["n"]
    b = RefCache(tmp_path).get_or_compute(p, cases)
    assert calls["n"] == n_first and a == b
    files = list((tmp_path / "ref_cache").glob("*.json"))
    assert len(files) == 1
    assert fingerprint(cases)[:16] in files[0].name
    # uncacheable predictors never write
    RefCache(tmp_path).get_or_compute(OraclePredictor(), cases)
    assert len(list((tmp_path / "ref_cache").glob("*.json"))) == 1


def test_cache_reads_earlier_version(tmp_path):
    calls = {"n": 0}

    def fn(ctx, q):
        calls["n"] += 1
        return Prediction(0.5, "ok")

    p = FunctionPredictor("f", fn, cache_key="v1", dedupe=False)
    cases = prefix_cases(make_e1_rows(2), (0, 2))
    RefCache(tmp_path / "v1").get_or_compute(p, cases)
    n = calls["n"]
    new = RefCache(tmp_path / "v2", [tmp_path / "v1"])
    new.get_or_compute(p, cases)
    assert calls["n"] == n and new.hits == 1
    assert not list((tmp_path / "v2" / "ref_cache").glob("*.json"))


def test_next_results_version(tmp_path):
    a = next_results_version(tmp_path)
    b = next_results_version(tmp_path)
    assert (a.name, b.name) == ("v1", "v2")


# ---- bayes omission / availability
def test_bayes_blocked_means_unavailable(monkeypatch):
    import cs182_icl.reference.bayes as bmod
    monkeypatch.delenv("CS282A_BAYES_PREDICTOR", raising=False)
    monkeypatch.setattr(bmod, "bayes_predict", lambda c, q, prior="train": Prediction(None, "blocked", "R1"))
    assert bayes_predictor_or_none() is None
    monkeypatch.setattr(bmod, "bayes_predict", lambda c, q, prior="train": Prediction(0.1, "ok"))
    assert bayes_predictor_or_none() is not None


def test_e1_without_bayes_omits_bayes_metrics(tmp_path):
    cases = prefix_cases(make_e1_rows(6), ex.E1_CS)
    r = ex.run_e1(cases, refs(), tmp_path / "e1", VAR_D, bayes=None)
    comments, rows = read_csv(tmp_path / "e1" / "e1_metrics.csv")
    assert any("bayes: unavailable" in c for c in comments)
    assert all(x["nmse_bayes_c0"] == "" and x["gap_to_bayes_m2"] == "" for x in rows)
    assert not any(x["method"].startswith("bayes") for x in rows)
    assert r["bayes_D"] is None
    for f in ("e1_error_vs_context.png", "e1_error_vs_context.csv", "e1_paired.csv"):
        assert (tmp_path / "e1" / f).exists()
    ora = [x for x in rows if x["method"] == "oracle"]
    assert all(float(x["mse_m2"]) == 0.0 for x in ora)
    ana = [x for x in rows if x["method"] == "analytic_true"]
    assert all(float(x["mse_m2"]) < 1e-20 for x in ana)  # synthetic target IS the analytic form


def test_e1_with_stub_bayes_has_nmse(tmp_path):
    stub = FunctionPredictor("bayes_train", lambda ctx, q: Prediction(0.01, "ok"), cache_key="stub")
    cases = prefix_cases(make_e1_rows(6), ex.E1_CS)
    r = ex.run_e1(cases, refs(), tmp_path / "e1", VAR_D, bayes=stub)
    _, rows = read_csv(tmp_path / "e1" / "e1_metrics.csv")
    b0 = next(x for x in rows if x["method"] == "bayes_train" and x["c"] == "0")
    assert float(b0["nmse_bayes_c0"]) == pytest.approx(1.0)
    assert r["bayes_D"] > 0


# ---- E2, E3, E4
def test_e2_pairing_and_outputs(tmp_path):
    cases = make_e2_cases(6)
    r = ex.run_e2(cases, refs(), tmp_path / "e2", VAR_D)
    assert (tmp_path / "e2" / "e2_same_vs_distinct.png").exists()
    labels = {p["comparison"] for p in r["paired"]}
    assert any("familiar" in l for l in labels) and any("new" in l for l in labels)
    # cell B (same force, new query) is not determined for physics fit; D (distinct) is
    fit = {(x["group"], x["c"]): x for x in r["table"] if x["method"] == "physics_fit"}
    assert fit[("E2/B", 4)]["n_ok"] == 0 and fit[("E2/B", 4)]["n_not_determined"] == 6
    assert fit[("E2/D", 2)]["n_ok"] == 6
    assert fit[("E2/A", 1)]["n_ok"] == 6  # same force as query: one observation suffices
    # paired same-vs-distinct rows only cover c in {2,4}
    assert {p["c"] for p in r["paired"]} <= {2, 4}


def test_e3_references_unchanged_by_copies(tmp_path):
    cases = make_e3_cases(5)

    class Drifting:  # synthetic model-like predictor whose output depends on sequence length
        name, kind = "transformer", "model"
        def cache_key(self): return None
        def predict(self, case): return Prediction(0.01 + 0.001 * len(case.item.context), "ok")

    r = ex.run_e3(cases, [Drifting(), *refs()], tmp_path / "e3", VAR_D)
    ch = {(x["method"], x["k"]): x for x in r["changes"]}
    for m in ("oracle", "analytic_true", "physics_fit"):
        for k in (1, 2, 4):
            assert ch[(m, k)]["max_abs_change_m"] == 0.0, (m, k)
    assert ch[("transformer", 4)]["mean_abs_change_m"] == pytest.approx(0.004)
    assert (tmp_path / "e3" / "e3_redundancy.png").exists()
    comments, _ = read_csv(tmp_path / "e3" / "e3_prediction_change.csv")
    assert any(plots.E3_LABEL in c for c in comments)


def test_e3_figure_carries_label(tmp_path, monkeypatch):
    import matplotlib.figure
    texts = []
    orig = matplotlib.figure.Figure.text
    monkeypatch.setattr(matplotlib.figure.Figure, "text",
                        lambda self, *a, **k: (texts.append(a[2]), orig(self, *a, **k))[1])
    ex.run_e3(make_e3_cases(3), refs(), tmp_path / "e3", VAR_D)
    assert plots.E3_LABEL in texts
    assert plots.E3_LABEL.startswith("Preliminary no-noise redundancy test. This does not test")


def test_e4_in_distribution_vs_ood(tmp_path):
    idc = prefix_cases(make_e1_rows(8, 0), ex.E4_CS)
    ood = prefix_cases(make_e1_rows(6, 5, 1.5, 2.0, "E4/heavy"), ex.E4_CS)
    r = ex.run_e4(idc, ood, refs(), tmp_path / "e4", VAR_D)
    comments, rows = read_csv(tmp_path / "e4" / "e4_metrics.csv")
    assert any("preliminary heavy-OOD behavior" in c for c in comments)
    assert {x["group"] for x in rows} == {"E1 in-distribution", "E4 heavy OOD"}
    assert {int(x["c"]) for x in rows} == {0, 2, 10}
    assert (tmp_path / "e4" / "e4_heavy_ood.png").exists()


def test_examples_printed(capsys):
    cases = prefix_cases(make_e1_rows(12), (0, 2))
    sc = evaluate_all(cases, refs())
    text = ex.print_examples(cases, sc, 2, n=10)
    assert len(text.splitlines()) == 1 + 1 + 10 + 1  # title, header, 10 rows, bayes note
    assert "(bayes: unavailable)" in capsys.readouterr().out


def test_cli_end_to_end_on_synthetic_dataset(tmp_path):
    from cs182_icl.eval.cli import main
    write_dataset(tmp_path / "ds")
    out = main(["--dataset-dir", str(tmp_path / "ds"), "--out-dir", str(tmp_path / "res")])
    names = {p.name for p in out.iterdir()}
    for f in ("e1_error_vs_context.png", "e2_same_vs_distinct.png", "e3_redundancy.png", "e4_heavy_ood.png"):
        assert f in names and f.replace(".png", ".csv") in names
    comments, _ = read_csv(out / "e1_metrics.csv")
    assert any("Var(d) = 2.500000e-03" in c for c in comments)
