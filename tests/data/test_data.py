"""T04 tests: construction rules on a tiny real-simulator dataset, round trips through the T05
training loader and the T06 evaluation path, train-only normalization."""

import copy
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from cs182_icl.data import checks
from cs182_icl.data.build import Sizes, build_arrays, build_dataset
from cs182_icl.data.store import compute_normalization, content_sha256, git_provenance

SIZES = Sizes(train=12, val=4, e1=6, e2=8, e3=8, e4=5, val_rows=16)
PROV = {"source_git_sha": "test-sha", "source_tree_clean": True}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("ds") / "tiny_v1"
    meta = build_dataset(out, sizes=SIZES, seed=0, workers=1, version="tiny_v1", provenance=PROV)
    D = {}
    for n in meta["npz_sha256"]:
        with np.load(out / n) as z:
            D[n[:-4]] = {k: z[k] for k in z.files}
    return out, meta, D


def mutated(D, name, field, fn):
    E = copy.deepcopy(D)
    fn(E[name][field])
    return E


def test_all_checks_pass_and_counts(built):
    out, meta, D = built
    checks.run_all(D)
    assert meta["counts"] == {"train": 12, "val": 4, "e1": 6, "e2": 72, "e3": 32, "e4": 5, "e4_cellD": 5}
    assert D["train"]["F"].shape == (12, 32) and D["val"]["row_idx"].shape == (16, 11)
    assert D["train"]["k"].dtype == np.int64 and D["e2"]["ctx_k"].dtype == np.int64


def test_flags_are_stored_and_counted(built):
    out, meta, D = built
    for n in ("train", "val"):
        assert meta["flags"][n]["timed_out"] == int(D[n]["timed_out"].sum())
        assert meta["flags"][n]["rollouts"] == D[n]["F"].size  # nothing dropped
    assert meta["flags"]["e2"]["rows"] == len(D["e2"]["item_id"])


def test_disjoint_cube_ids_and_ranges(built):
    out, meta, D = built
    E = copy.deepcopy(D)
    E["val"]["cube_id"][0] = E["train"]["cube_id"][0]
    with pytest.raises(AssertionError, match="cube ids"):
        checks.check_cube_ids(E)
    E = copy.deepcopy(D)
    E["e3"]["cube_id"][:] = E["e1"]["cube_id"][0]
    with pytest.raises(AssertionError):
        checks.check_cube_ids(E)


def test_actions_independent_of_cube_and_ranges(built):
    out, meta, D = built
    with pytest.raises(AssertionError, match="out of range"):
        checks.check_ranges(mutated(D, "train", "F", lambda x: x.__setitem__((0, 0), 30.0)))
    with pytest.raises(AssertionError, match="k out of range|integer"):
        checks.check_ranges(mutated(D, "train", "k", lambda x: x.__setitem__((0, 0), 99)))
    E = copy.deepcopy(D)
    E["train"]["k"] = E["train"]["k"].astype(float)
    with pytest.raises(AssertionError):
        checks.check_schema(E)
    # actions come from a separate stream: same action matrix for different cube params
    a1, _ = build_arrays(Sizes(train=3, val=1, e1=1, e2=1, e3=1, e4=1, val_rows=2), 0, 1)
    assert np.array_equal(a1["train"]["F"], D["train"]["F"][:3])


def test_measurement_ids_unique_and_consistent(built):
    out, meta, D = built
    ids = D["train"]["measurement_id"]
    assert len(set(ids.ravel().tolist())) == ids.size
    E = mutated(D, "train", "measurement_id", lambda x: x.__setitem__((0, 1), x[0, 0]))
    with pytest.raises(AssertionError, match="not unique"):
        checks.check_measurement_ids(E)
    E = copy.deepcopy(D)
    E["e1"]["ctx_measurement_id"][0, 0] = E["train"]["measurement_id"][0, 0]
    with pytest.raises(AssertionError):
        checks.check_measurement_ids(E)


def test_e1_structure(built):
    out, meta, D = built
    a = D["e1"]
    assert (a["ctx_len"] == 10).all() and not np.isnan(a["ctx_F"]).any()
    with pytest.raises(AssertionError):
        checks.check_e1(mutated(D, "e1", "ctx_len", lambda x: x.__setitem__(0, 9)))


def test_e2_cells_rules(built):
    out, meta, D = built
    a = D["e2"]
    conds = set(a["condition"].tolist())
    assert conds == {"E2/A", "E2/B", "E2/C", "E2/D"}
    rows = {(str(c), int(n)) for c, n in zip(a["condition"], a["ctx_len"])}
    assert rows == {("E2/A", 1), ("E2/A", 2), ("E2/A", 4), ("E2/B", 2), ("E2/B", 4), ("E2/C", 2), ("E2/C", 4),
                    ("E2/D", 2), ("E2/D", 4)}
    # cell B: ONE single context force, >= 3 N from F_q; A: all F_q
    for i in range(len(a["item_id"])):
        n = int(a["ctx_len"][i])
        F, Fq = a["ctx_F"][i, :n], a["query_F"][i]
        c = str(a["condition"][i])[-1]
        if c == "B":
            assert len(set(F)) == 1 and abs(F[0] - Fq) >= 3
        if c == "A":
            assert (F == Fq).all()
        if c == "D":
            assert (np.abs(F - Fq) >= 3).all() and len(set(F)) == n
        assert 17 <= Fq <= 22
    # randomized order: not every context sorted by duration
    assert any(list(a["ctx_k"][i, :4]) != sorted(a["ctx_k"][i, :4]) for i in range(len(a["item_id"])) if a["ctx_len"][i] == 4)
    # mutations fail loudly
    def idx(cell, c):
        return next(i for i in range(len(a["item_id"])) if str(a["condition"][i]) == f"E2/{cell}" and a["ctx_len"][i] == c)
    i = idx("B", 4)
    with pytest.raises(AssertionError, match="SINGLE"):
        checks.check_e2(mutated(D, "e2", "ctx_F", lambda x: x.__setitem__((i, 0), a["ctx_F"][i, 0] + 0.5)))
    i = idx("D", 2)
    with pytest.raises(AssertionError, match="within 3 N"):
        checks.check_e2(mutated(D, "e2", "ctx_F", lambda x: x.__setitem__((i, 0), a["query_F"][i] + 1.0)))
    i = idx("A", 2)
    with pytest.raises(AssertionError):
        checks.check_e2(mutated(D, "e2", "ctx_F", lambda x: x.__setitem__((i, 0), a["query_F"][i] + 1.0)))
    i = idx("C", 4)
    with pytest.raises(AssertionError, match="lacks F_q|multisets"):
        checks.check_e2(mutated(D, "e2", "ctx_F", lambda x: x.__setitem__((i, np.flatnonzero(a["ctx_F"][i] == a["query_F"][i])[0]), 14.0)))
    i = idx("D", 4)
    with pytest.raises(AssertionError, match="multisets"):
        checks.check_e2(mutated(D, "e2", "ctx_k", lambda x: x.__setitem__((i, 0), 100 if a["ctx_k"][i, 0] != 100 else 101)))
    i = idx("C", 2)
    with pytest.raises(AssertionError, match="differs across cells"):
        checks.check_e2(mutated(D, "e2", "d_query_clean", lambda x: x.__setitem__(i, 1.0)))
    # same duration multiset between paired cells, query duration never in a context
    by = {}
    for i in range(len(a["item_id"])):
        by[(int(a["item_id"][i]), str(a["condition"][i])[-1], int(a["ctx_len"][i]))] = sorted(a["ctx_k"][i, :a["ctx_len"][i]])
    for (it, cell, c), ks in by.items():
        if cell == "C":
            assert ks == by[(it, "A", c)]
        if cell == "D":
            assert ks == by[(it, "B", c)]


def test_e3_copy_identity(built):
    out, meta, D = built
    a = D["e3"]
    for i in range(len(a["item_id"])):
        n = int(str(a["condition"][i])[5:])
        assert a["ctx_len"][i] == 2 + n
        assert abs(a["ctx_F"][i, 0] - a["ctx_F"][i, 1]) >= 6
        for j in range(n):
            s, t = 2 + j, j % 2
            for f in ("ctx_F", "ctx_k", "ctx_d_clean", "ctx_d_obs", "ctx_measurement_id"):
                assert a[f][i, s] == a[f][i, t]
    i = next(i for i in range(len(a["item_id"])) if a["condition"][i] == "E3/k=2")
    with pytest.raises(AssertionError, match="identical copy"):
        checks.check_e3(mutated(D, "e3", "ctx_d_obs", lambda x: x.__setitem__((i, 2), x[i, 2] + 1e-3)))
    with pytest.raises(AssertionError, match="identical copy|measurement id"):
        checks.check_e3(mutated(D, "e3", "ctx_measurement_id", lambda x: x.__setitem__((i, 2), 10**9)))
    with pytest.raises(AssertionError, match="apart"):
        checks.check_e3(mutated(D, "e3", "ctx_F", lambda x: x.__setitem__((slice(None), 1), x[:, 0] + 1.0)))


def test_e4_heavy_and_cell_d(built):
    out, meta, D = built
    assert (D["e4"]["m"] >= 1.5).all() and (D["e4"]["m"] <= 2.0).all()
    assert set(D["e4_cellD"]["condition"].tolist()) == {"E4/D"} and (D["e4_cellD"]["ctx_len"] == 2).all()
    with pytest.raises(AssertionError):
        checks.check_e4(mutated(D, "e4", "m", lambda x: x.__setitem__(0, 1.0)))


def test_targets_are_real_simulator_outcomes(built):
    out, meta, D = built
    from cs182_icl.sim.rollout import simulate_push
    from cs182_icl.types import Action, Cube

    a = D["e2"]
    for i in (0, 5, 30):
        r = simulate_push(Cube(float(a["m"][i]), float(a["mu"][i]), 0), Action(float(a["query_F"][i]), int(a["query_k"][i])))
        assert r.d == a["d_query_clean"][i]


def test_deterministic_and_hashes(built, tmp_path):
    out, meta, D = built
    assert meta["content_sha256"] == {f"{n}.npz": content_sha256(D[n]) for n in D}
    again, _ = build_arrays(SIZES, 0, 1)
    assert content_sha256(again["e2"]) == meta["content_sha256"]["e2.npz"]
    other, _ = build_arrays(SIZES, 1, 1)
    assert content_sha256(other["e2"]) != meta["content_sha256"]["e2.npz"]
    assert meta["generation_seed"] == 0 and meta["dataset_hash"] and meta["version"] == "tiny_v1"
    with pytest.raises(FileExistsError):
        build_dataset(out, sizes=SIZES, seed=0, workers=1, version="x", provenance=PROV)


def test_git_clean_check(tmp_path):
    run = lambda *c: subprocess.run(["git", "-C", str(tmp_path), *c], check=True, capture_output=True)
    run("init", "-q")
    (tmp_path / "f").write_text("a")
    run("add", "f")
    run("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x")
    assert git_provenance(tmp_path)["source_tree_clean"]
    (tmp_path / "f").write_text("b")
    with pytest.raises(RuntimeError, match="not clean"):
        git_provenance(tmp_path)


def test_train_only_normalization(built):
    out, meta, D = built
    t = D["train"]
    base = compute_normalization(t["F"], t["k"], t["d_obs"])
    assert base == meta["normalization"]
    # same format/values as the T05 helper
    from cs182_icl.model.tokens import compute_norm

    assert base == compute_norm(t["F"], t["k"], t["d_obs"])
    # val/test changes cannot move the statistics (they are a function of train arrays only)
    E = copy.deepcopy(D)
    E["val"]["d_obs"] *= 1e6
    E["e1"]["ctx_d_obs"] *= 1e6
    assert compute_normalization(E["train"]["F"], E["train"]["k"], E["train"]["d_obs"]) == base
    # and the builder: a different val/test seed stream leaves train, hence normalization, unchanged
    a2, _ = build_arrays(Sizes(train=12, val=7, e1=2, e2=2, e3=2, e4=2, val_rows=4), 0, 1)
    assert compute_normalization(a2["train"]["F"], a2["train"]["k"], a2["train"]["d_obs"]) == base


def test_round_trip_t05_training_loader(built):
    out, meta, D = built
    from cs182_icl.train.data import RowSampler, load_dataset

    ds = load_dataset(out)
    assert ds["norm"] == meta["normalization"]
    assert ds["files"]["train"] == meta["npz_sha256"]["train.npz"] and ds["files"]["val"] == meta["npz_sha256"]["val.npz"]
    row_cube, row_idx = ds["val_rows"]
    assert row_idx.shape == (16, 11) and row_cube.shape == (16,)
    s = RowSampler(ds["train"], "cpu", seed=0)
    F, T, d_obs, d_clean = s.sample(5)
    assert F.shape == (5, 11) and bool((d_obs == d_clean).all())


def test_round_trip_t06_eval_path(built, tmp_path):
    out, meta, D = built
    from cs182_icl.eval import cli
    from cs182_icl.eval.cases import (assert_e2_cells, assert_e3_copies, assert_finite_targets,
                                      assert_prefix_paired, load_npz_rows, prefix_cases, rows_as_cases)
    from cs182_icl.eval.metrics import train_d_variance_with_key

    var_d, key = train_d_variance_with_key(meta)
    assert key == "normalization.d_obs.std" and var_d == meta["normalization"]["d_obs"]["std"] ** 2
    lines = cli.provenance(out, meta, key, None, ["e1.npz", "e2.npz", "e3.npz", "e4.npz"])
    assert "NOT VERIFIED" not in lines[-1] and "e1.npz, e2.npz, e3.npz, e4.npz" in lines[-1]
    assert "test-sha" in lines[0] and "tiny_v1" in lines[0]
    e1 = prefix_cases(load_npz_rows(out / "e1.npz"), (0, 1, 2, 4, 8, 10))
    assert_prefix_paired(e1)
    e2 = rows_as_cases(load_npz_rows(out / "e2.npz"))
    assert_e2_cells(e2)
    assert_finite_targets(e2)
    e3 = rows_as_cases(load_npz_rows(out / "e3.npz"))
    assert_e3_copies(e3)
    assert_prefix_paired(e3, group_key=lambda cs: (cs.item_id,))
    assert prefix_cases(load_npz_rows(out / "e4.npz"), (0, 2, 10))
    # corrupt file -> consumer refuses (hash is really checked)
    bad = tmp_path / "bad"
    bad.mkdir()
    for n in ("e1.npz",):
        (bad / n).write_bytes((out / n).read_bytes() + b"x")
    (bad / "meta.json").write_text((out / "meta.json").read_text())
    with pytest.raises(RuntimeError, match="sha256"):
        cli.provenance(bad, meta, key, None, ["e1.npz"])


def test_round_trip_t06_cli_references_only(built, tmp_path):
    out, meta, D = built
    from cs182_icl.eval import cli

    res = cli.main(["--dataset-dir", str(out), "--out-dir", str(tmp_path / "res"), "--experiments", "e1,e2,e3,e4"])
    text = "\n".join(p.read_text() for p in Path(res).glob("*.csv"))
    assert "NOT VERIFIED" not in text and "sha256 verified" in text
