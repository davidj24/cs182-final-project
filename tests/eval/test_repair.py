"""T06 repair-cycle tests: E2 structure (M1/m6), partial Bayes (M3), cache keys (M4/m8),
dedupe copy semantics (m1), clean-value masking (m2), provenance (m4/m7)."""

import json
from dataclasses import replace

import numpy as np
import pytest

from cs182_icl.eval import cache as cache_mod
from cs182_icl.eval import experiments as ex
from cs182_icl.eval.cache import RefCache
from cs182_icl.eval.cases import assert_e2_cells, assert_e3_copies, prefix_cases
from cs182_icl.eval.metrics import read_csv
from cs182_icl.eval.predictors import FunctionPredictor, ModelPredictor, dedupe_by_measurement_id
from cs182_icl.types import Action, Prediction
from tests.eval.synth import make_e1_rows, make_e2_cases, make_e3_cases, write_dataset
from tests.eval.test_harness import VAR_D, refs


# ---------------------------------------------------------------- M1 / m6
def _mutate(cases, cell, c, fn):
    out, done = [], False
    for cs in cases:
        if cs.item_id == 0 and cs.item.condition == f"E2/{cell}" and cs.c == c:
            cs, done = fn(cs), True
        out.append(cs)
    assert done
    return out


def _with_forces(cs, forces):
    ctx = [replace(o, action=Action(f, o.action.k)) for o, f in zip(cs.item.context, forces)]
    return replace(cs, item=replace(cs.item, context=ctx))


def test_e2_valid_cells_pass_and_nesting_not_required(tmp_path):
    cases = make_e2_cases(4)
    assert_e2_cells(cases)

    def reorder(cs):  # D at c=2 uses a context that is NOT a prefix of the c=4 context
        return replace(cs, item=replace(cs.item, context=list(cs.item.context[::-1])))

    cases = [reorder(cs) if cs.item.condition == "E2/D" and cs.c == 2 else cs for cs in cases]
    ex.run_e2(cases, refs(), tmp_path / "e2", VAR_D)  # must not raise


@pytest.mark.parametrize("cell,c,forces,msg", [
    ("A", 2, "notFq", "!= F_q"),
    ("B", 2, "near", "within 3 N"),
    ("D", 2, "near", "within 3 N"),
    ("C", 2, "noFq", "lacks F_q"),
    ("C", 2, "dup", "not distinct"),
])
def test_e2_force_constraints_raise(cell, c, forces, msg):
    cases = make_e2_cases(3)
    Fq = next(cs.item.query.F for cs in cases if cs.item_id == 0)

    def fn(cs):
        n = cs.c
        table = {"notFq": [Fq + 1.0] * n, "near": [Fq + 1.0, Fq - 6.0, Fq + 5.0, Fq - 8.0][:n],
                 "noFq": [Fq - 5.0, Fq + 4.0, Fq - 6.0, Fq + 7.0][:n],
                 "dup": [Fq, Fq, Fq - 5.0, Fq + 4.0][:n]}
        return _with_forces(cs, table[forces])

    with pytest.raises(AssertionError, match=msg):
        assert_e2_cells(_mutate(cases, cell, c, fn))


def test_e2_cross_cell_pairing_raises():
    cases = make_e2_cases(3)
    bad_target = _mutate(cases, "C", 2, lambda cs: replace(
        cs, item=replace(cs.item, d_query_clean=cs.item.d_query_clean + 1e-3)))
    with pytest.raises(AssertionError, match="differ across cells"):
        assert_e2_cells(bad_target)
    bad_query = _mutate(cases, "B", 4, lambda cs: replace(
        cs, item=replace(cs.item, query=Action(cs.item.query.F, cs.item.query.k + 1))))
    with pytest.raises(AssertionError, match="differ across cells"):
        assert_e2_cells(bad_query)
    # D has a c with no B partner / C has c with no A partner / pair shares no c
    no_pair = [cs for cs in cases if not (cs.item_id == 0 and cs.item.condition == "E2/B")]
    with pytest.raises(AssertionError, match="share no context length"):
        assert_e2_cells(no_pair)
    only_c4 = [cs for cs in cases if not (cs.item_id == 0 and cs.item.condition == "E2/B" and cs.c == 2)]
    with pytest.raises(AssertionError, match="no B pair"):
        assert_e2_cells(only_c4)


def test_e2_run_raises_on_violation(tmp_path):
    cases = _mutate(make_e2_cases(3), "A", 1, lambda cs: _with_forces(cs, [cs.item.query.F + 2.0]))
    with pytest.raises(AssertionError):
        ex.run_e2(cases, refs(), tmp_path / "e2", VAR_D)


# ---------------------------------------------------------------- M3
def _partial_bayes():
    """Valid for most cases, not_determined for some c=0 items -> D must be None."""
    def fn(ctx, q):
        if len(ctx) == 0 and q.k % 2 == 0:
            return Prediction(None, "not_determined", "stub")
        return Prediction(0.01, "ok")
    return FunctionPredictor("bayes_train", fn, cache_key="partial")


def test_partial_bayes_dropped_everywhere_when_D_none(tmp_path):
    cases = prefix_cases(make_e1_rows(10), ex.E1_CS)
    assert any(cs.item.query.k % 2 == 0 for cs in cases)
    r = ex.run_e1(cases, refs(), tmp_path / "e1", VAR_D, bayes=_partial_bayes())
    assert r["bayes_D"] is None and "bayes_train" not in r["scores"]
    for f in (tmp_path / "e1").iterdir():
        if f.suffix == ".csv":
            text = f.read_text()
            rows = [ln for ln in text.splitlines() if not ln.startswith("#")]
            assert not any("bayes_train" in ln for ln in rows), f.name
    comments, rows = read_csv(tmp_path / "e1" / "e1_metrics.csv")
    assert any("bayes: unavailable" in c for c in comments)
    assert all(x["nmse_bayes_c0"] == "" and x["gap_to_bayes_m2"] == "" for x in rows)


def test_partial_bayes_dropped_in_e2_e3_e4(tmp_path):
    b = _partial_bayes()
    r2 = ex.run_e2(make_e2_cases(4), refs(), tmp_path / "e2", VAR_D, bayes=b)
    r3 = ex.run_e3(make_e3_cases(4), refs(), tmp_path / "e3", VAR_D, bayes=b)
    idc = prefix_cases(make_e1_rows(8, 0), ex.E4_CS)
    ood = prefix_cases(make_e1_rows(6, 5, 1.5, 2.0, "E4/heavy"), ex.E4_CS)
    r4 = ex.run_e4(idc, ood, refs(), tmp_path / "e4", VAR_D, bayes=b)
    assert "bayes_train" not in r2["scores"] and "bayes_train" not in r3["scores"]
    assert "bayes_train" not in r4["scores_id"] and "bayes_train" not in r4["scores_ood"]


# ---------------------------------------------------------------- M4 / m8
def test_cache_invalidated_when_code_hash_changes(tmp_path, monkeypatch):
    calls = {"n": 0}

    def fn(ctx, q):
        calls["n"] += 1
        return Prediction(0.5, "ok")

    p = FunctionPredictor("f", fn, cache_key="v1", dedupe=False)
    cases = prefix_cases(make_e1_rows(2), (0, 2))
    RefCache(tmp_path / "v1").get_or_compute(p, cases)
    n = calls["n"]
    monkeypatch.setattr(cache_mod, "code_hash", lambda: "f" * 64)
    new = RefCache(tmp_path / "v2", [tmp_path / "v1"])
    new.get_or_compute(p, cases)
    assert calls["n"] > n and new.hits == 0 and new.misses == 1


def test_code_hash_covers_sources_and_decisions():
    h = cache_mod.code_hash()
    assert len(h) == 64
    assert (cache_mod._PKG.parents[1] / "docs" / "DECISIONS.md").exists()


def test_header_records_cache_source(tmp_path):
    p = FunctionPredictor("f", lambda c, q: Prediction(0.5, "ok"), cache_key="v1", dedupe=False)
    cases = prefix_cases(make_e1_rows(3), ex.E1_CS)
    ex.run_e1(cases, [p], tmp_path / "v1", VAR_D, cache=RefCache(tmp_path / "v1"))
    ex.run_e1(cases, [p], tmp_path / "v2", VAR_D, cache=RefCache(tmp_path / "v2", [tmp_path / "v1"]))
    comments, _ = read_csv(tmp_path / "v2" / "e1_metrics.csv")
    assert any("supplied by" in c and str(tmp_path / "v1") in c for c in comments)
    assert any("code_hash" in c for c in comments)


def test_cli_refuses_existing_out_dir_and_checks_hash(tmp_path):
    from cs182_icl.eval.cli import main
    write_dataset(tmp_path / "ds")
    (tmp_path / "exists").mkdir()
    with pytest.raises(SystemExit):
        main(["--dataset-dir", str(tmp_path / "ds"), "--out-dir", str(tmp_path / "exists")])
    meta = json.loads((tmp_path / "ds" / "meta.json").read_text())
    meta["sha256"] = {"e1.npz": "0" * 64}
    (tmp_path / "ds" / "meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="sha256"):
        main(["--dataset-dir", str(tmp_path / "ds"), "--out-dir", str(tmp_path / "res"),
              "--experiments", "e1"])


def test_cli_header_reports_var_key(tmp_path):
    from cs182_icl.eval.cli import main
    write_dataset(tmp_path / "ds")
    out = main(["--dataset-dir", str(tmp_path / "ds"), "--out-dir", str(tmp_path / "res"),
                "--experiments", "e1"])
    comments, _ = read_csv(out / "e1_metrics.csv")
    assert any("normalization.d_obs.std" in c for c in comments)
    assert any("NOT VERIFIED" in c for c in comments)  # synthetic meta has no hashes


# ---------------------------------------------------------------- m1 / m2
def test_dedupe_rejects_non_exact_copies():
    big = max(make_e3_cases(2), key=lambda c: c.c)
    ctx = list(big.item.context)
    ctx.append(replace(ctx[0], d_obs=ctx[0].d_obs + 1e-3))  # same ID, different value
    with pytest.raises(ValueError, match="exact copies"):
        dedupe_by_measurement_id(ctx)
    bad = replace(big, c=len(ctx), item=replace(big.item, context=ctx))
    with pytest.raises(AssertionError, match="exact copy"):
        assert_e3_copies([bad])
    assert_e3_copies([big])


def test_model_predictor_never_sees_clean_values(monkeypatch):
    import cs182_icl.model.predict as mp
    seen = []

    def fake(model, items, norm):
        seen.extend(items)
        return np.full(len(items), 0.01)

    monkeypatch.setattr(mp, "model_predict", fake)
    cases = prefix_cases(make_e1_rows(3), (0, 4))
    out = ModelPredictor(None, {}).predict_batch(cases)
    assert len(out) == len(cases) and seen
    for it in seen:
        assert np.isnan(it.d_query_clean) and all(np.isnan(o.d_clean) for o in it.context)
    assert all(np.isfinite(cs.item.d_query_clean) for cs in cases)  # originals untouched
