"""Assertions for every construction rule. Each check raises AssertionError (fail loudly).

They run on the arrays exactly as written to disk (dict name -> dict of arrays), both when the
dataset is built and when the dataset card is made (scripts/make_dataset_card.py).
"""

from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from cs182_icl.data import sampling as S
from cs182_icl.physics import analytic_displacement  # noqa: F401  (kept for tests importing via checks)

SPLIT_FIELDS = {"cube_id": "i", "m": "f", "mu": "f", "F": "f", "k": "i", "measurement_id": "i", "d_clean": "f",
                "d_obs": "f", "t_stop": "f", "max_rotation_deg": "f", "timed_out": "b"}
TEST_FIELDS = {"item_id": "i", "condition": "U", "cube_id": "i", "m": "f", "mu": "f", "query_F": "f",
               "d_query_clean": "f", "query_t_stop": "f", "query_max_rotation_deg": "f", "query_k": "i",
               "query_timed_out": "b", "ctx_len": "i", "ctx_F": "f", "ctx_d_clean": "f", "ctx_d_obs": "f",
               "ctx_k": "i", "ctx_measurement_id": "i"}
TEST_SETS = ("e1", "e2", "e3", "e4", "e4_cellD")
EPS = 1e-9


def _kind(a):
    return a.dtype.kind


def check_schema(D):
    for name, a in D.items():
        fields = SPLIT_FIELDS if name in ("train", "val") else TEST_FIELDS
        for f, kind in fields.items():
            assert f in a, f"{name}.npz lacks field {f}"
            assert _kind(a[f]) == kind, f"{name}.{f} dtype {a[f].dtype}, want kind {kind}"
        if name in ("train", "val"):
            n = a["cube_id"].shape[0]
            for f in ("F", "k", "measurement_id", "d_clean", "d_obs", "t_stop", "max_rotation_deg", "timed_out"):
                assert a[f].shape[0] == n and a[f].ndim == 2 and a[f].shape[1] >= 11, f"{name}.{f} shape {a[f].shape}"
            assert all(np.isfinite(a[f]).all() for f in ("F", "d_clean", "d_obs", "t_stop")), f"{name}: non-finite values"
        else:
            I = a["item_id"].shape[0]
            for f in fields:
                assert a[f].shape[0] == I, f"{name}.{f} first dim"
    return "dtypes and shapes match docs/INTERFACES.md"


def check_val_rows(D):
    v = D["val"]
    rc, ri = v["row_cube"], v["row_idx"]
    assert ri.ndim == 2 and ri.shape[1] == 11 and rc.shape[0] == ri.shape[0]
    assert ri.min() >= 0 and ri.max() < v["F"].shape[1]
    assert all(len(set(r)) == 11 for r in ri.tolist()), "val row_idx has repeated pushes"
    assert rc.min() >= 0 and rc.max() < v["F"].shape[0]
    return f"{ri.shape[0]} fixed validation rows of 11 distinct pushes"


def _in(x, lo, hi):
    return bool(((x >= lo - EPS) & (x <= hi + EPS)).all())


def check_ranges(D):
    for name in ("train", "val"):
        a = D[name]
        assert _in(a["m"], *S.M_RANGE) and _in(a["mu"], *S.MU_RANGE), f"{name}: cube params out of range"
        assert _in(a["F"], *S.F_RANGE), f"{name}: F out of range"
        assert _in(a["k"], *S.K_RANGE), f"{name}: k out of range"
        assert a["k"].dtype.kind == "i", f"{name}: k must be integer steps"
    for name in TEST_SETS:
        a = D[name]
        mr = S.M_HEAVY_RANGE if name.startswith("e4") else S.M_RANGE
        assert _in(a["m"], *mr) and _in(a["mu"], *S.MU_RANGE), f"{name}: cube params out of range"
        for f, k in (("ctx_F", "ctx_k"),):
            msk = a[k] >= 0
            assert _in(a[f][msk], *S.F_RANGE) and _in(a[k][msk], *S.K_RANGE), f"{name}: context action out of range"
        assert _in(a["query_F"], *S.F_RANGE) and _in(a["query_k"], *S.K_RANGE), f"{name}: query out of range"
        assert a["query_k"].dtype.kind == "i" and a["ctx_k"].dtype.kind == "i"
    return "all cube params, forces and integer durations inside the frozen ranges (heavy m for E4)"


def check_cube_ids(D):
    sets = {}
    for name in ("train", "val", "e1", "e2", "e3", "e4"):
        ids = D[name]["cube_id"]
        if name in TEST_SETS:
            # one cube per item: E2/E3 rows repeat it across cells, E1/E4 do not
            pass
        sets[name] = set(ids.tolist())
        if name in ("train", "val", "e1", "e4"):
            assert len(sets[name]) == len(ids), f"{name}: duplicate cube ids"
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not (sets[a] & sets[b]), f"cube ids shared between {a} and {b}"
    assert set(D["e4_cellD"]["cube_id"].tolist()) <= sets["e4"], "e4_cellD cubes must be the e4 cubes"
    # one parameter set per cube id everywhere
    for name in TEST_SETS:
        a = D[name]
        seen = {}
        for c, m, mu in zip(a["cube_id"], a["m"], a["mu"]):
            assert seen.setdefault(int(c), (m, mu)) == (m, mu), f"{name}: cube {c} has two parameter sets"
    return {n: len(s) for n, s in sets.items()}


def _obs_rows(a):
    for i in range(a["item_id"].shape[0]):
        for j in range(int(a["ctx_len"][i])):
            yield i, j, (float(a["ctx_F"][i, j]), int(a["ctx_k"][i, j]), float(a["ctx_d_obs"][i, j]),
                         float(a["ctx_d_clean"][i, j]), int(a["cube_id"][i]))


def check_measurement_ids(D):
    owner = {}
    for name in ("train", "val"):
        ids = D[name]["measurement_id"].ravel().tolist()
        assert len(set(ids)) == len(ids), f"{name}: measurement ids not unique"
        assert min(ids) >= 0
        for i in ids:
            assert owner.setdefault(i, name) == name, f"measurement id {i} in {owner[i]} and {name}"
    for name in TEST_SETS:
        a = D[name]
        pad = np.arange(a["ctx_F"].shape[1])[None, :] >= a["ctx_len"][:, None]
        assert (a["ctx_measurement_id"][pad] == -1).all() and (a["ctx_measurement_id"][~pad] >= 0).all(), \
            f"{name}: padding must be exactly the positions beyond ctx_len"
        assert np.isnan(a["ctx_F"][pad]).all() and not np.isnan(a["ctx_F"][~pad]).any()
        obs = {}
        for i, j, o in _obs_rows(a):
            mid = int(a["ctx_measurement_id"][i, j])
            assert obs.setdefault(mid, o) == o, f"{name}: measurement id {mid} used for two different observations"
            assert owner.setdefault(mid, name) == name, f"measurement id {mid} shared by {owner[mid]} and {name}"
    return "ids unique per fresh measurement; each id names exactly one (action, outcome); files disjoint"


def _ctx(a, i):
    n = int(a["ctx_len"][i])
    return [(float(a["ctx_F"][i, j]), int(a["ctx_k"][i, j]), int(a["ctx_measurement_id"][i, j]),
             float(a["ctx_d_obs"][i, j]), float(a["ctx_d_clean"][i, j])) for j in range(n)]


def check_noise0(D):
    for name in TEST_SETS:
        a = D[name]
        m = a["ctx_len"][:, None] > np.arange(a["ctx_F"].shape[1])[None, :]
        assert (a["ctx_d_obs"][m] == a["ctx_d_clean"][m]).all(), f"{name}: d_obs != d_clean at noise 0"
    for name in ("train", "val"):
        assert (D[name]["d_obs"] == D[name]["d_clean"]).all()
    return "d_obs == d_clean everywhere (pilot noise 0)"


def check_e1(D, name="e1", cond="E1"):
    a = D[name]
    assert (a["ctx_len"] == 10).all(), f"{name}: every item needs a 10-push context"
    assert set(a["condition"].tolist()) == {cond}
    assert len(set(a["item_id"].tolist())) == a["item_id"].shape[0]
    for i in range(a["item_id"].shape[0]):
        ids = [o[2] for o in _ctx(a, i)]
        assert len(set(ids)) == 10, f"{name} item {i}: repeated measurement id"
    return f"{a['item_id'].shape[0]} items, 10 independent pushes + 1 query each"


def _cell_forces_ok(cell, Fq, forces, where):
    near = [abs(f - Fq) < S.NEAR for f in forces]
    if cell == "A":
        assert all(f == Fq for f in forces), f"{where}: A context force != F_q"
    elif cell == "B":
        assert len(set(forces)) == 1, f"{where}: B must use a SINGLE context force"
        assert not any(near), f"{where}: B force within 3 N of F_q"
    elif cell == "C":
        assert len(set(forces)) == len(forces), f"{where}: C forces not distinct"
        assert Fq in forces, f"{where}: C lacks F_q"
    elif cell == "D":
        assert len(set(forces)) == len(forces), f"{where}: D forces not distinct"
        assert not any(near), f"{where}: D force within 3 N of F_q"


def _check_cells(a, prefix, cells, require_pairs):
    by = defaultdict(dict)  # item_id -> (cell, c) -> row
    for i in range(a["item_id"].shape[0]):
        cond = str(a["condition"][i])
        pre, cell = cond.split("/")
        assert pre == prefix and cell in cells, f"unknown condition {cond}"
        c = int(a["ctx_len"][i])
        assert c in cells[cell], f"{cond}: context length {c} not in {cells[cell]}"
        key = (cell, c)
        iid = int(a["item_id"][i])
        assert key not in by[iid], f"item {iid}: duplicate row {key}"
        by[iid][key] = i
        Fq, kq = float(a["query_F"][i]), int(a["query_k"][i])
        ctx = _ctx(a, i)
        _cell_forces_ok(cell, Fq, [o[0] for o in ctx], f"item {iid} {cond} c={c}")
        assert all(o[1] != kq for o in ctx), f"item {iid}: a context duration equals k_q"
        if cell in ("A", "B"):
            assert len({o[1] for o in ctx}) == len(ctx), f"item {iid} {cond}: durations must vary"
    for iid, rows in by.items():
        idx = list(rows.values())
        for f in ("cube_id", "m", "mu", "query_F", "query_k", "d_query_clean"):
            assert len({a[f][i].item() for i in idx}) == 1, f"item {iid}: {f} differs across cells"
        assert S.FQ_RANGE[0] <= a["query_F"][idx[0]] <= S.FQ_RANGE[1]
        assert len({int(a["ctx_measurement_id"][i, j]) for i in idx for j in range(int(a["ctx_len"][i]))}) == \
            sum(int(a["ctx_len"][i]) for i in idx), f"item {iid}: every E2 push must be a fresh measurement"
        for same, dist in (("A", "C"), ("B", "D")):
            if not require_pairs:
                continue
            for c in cells[dist]:
                assert (same, c) in rows and (dist, c) in rows, f"item {iid}: missing {same}/{dist} at c={c}"
                ds = sorted(o[1] for o in _ctx(a, rows[(same, c)]))
                dd = sorted(o[1] for o in _ctx(a, rows[(dist, c)]))
                assert ds == dd, f"item {iid}: duration multisets differ between {same} and {dist} at c={c}"
        expected = sum(len(v) for v in cells.values())
        assert len(rows) == expected, f"item {iid}: {len(rows)} rows, expected {expected}"
    return len(by)


def check_e2(D):
    cells = {"A": (1, 2, 4), "B": (2, 4), "C": (2, 4), "D": (2, 4)}
    n = _check_cells(D["e2"], "E2", cells, True)
    return f"{n} items x 9 rows (A c=1,2,4; B,C,D c=2,4): force rules, B single force, same cube/query/target, duration multisets A=C, B=D, no k_q in contexts, fresh ids"


def check_e3(D):
    a = D["e3"]
    by = defaultdict(dict)
    for i in range(a["item_id"].shape[0]):
        cond = str(a["condition"][i])
        assert cond.startswith("E3/k="), cond
        n = int(cond[5:])
        assert n in (0, 1, 2, 4)
        iid = int(a["item_id"][i])
        by[iid][n] = i
    for iid, rows in by.items():
        assert sorted(rows) == [0, 1, 2, 4], f"item {iid}: copy counts {sorted(rows)}"
        base = _ctx(a, rows[0])
        assert len(base) == 2 and len({o[2] for o in base}) == 2
        (F1, k1, *_), (F2, k2, *_) = base
        assert abs(F1 - F2) >= S.E3_SEP, f"item {iid}: base forces only {abs(F1 - F2)} N apart"
        Fq, kq = float(a["query_F"][rows[0]]), int(a["query_k"][rows[0]])
        assert all(abs(F - Fq) >= S.NEAR for F in (F1, F2)), f"item {iid}: not a new-force query"
        assert kq not in (k1, k2)
        for n, i in rows.items():
            ctx = _ctx(a, i)
            assert len(ctx) == 2 + n, f"item {iid} k={n}: context length {len(ctx)}"
            assert ctx[:2] == base, f"item {iid} k={n}: base context changed"
            for j in range(n):
                assert ctx[2 + j] == base[j % 2], f"item {iid} k={n}: copy {j} is not an identical copy of base {j % 2}"
            assert len({o[2] for o in ctx}) == 2, "copies must reuse measurement ids"
            for f in ("cube_id", "m", "mu", "query_F", "query_k", "d_query_clean"):
                assert a[f][i] == a[f][rows[0]], f"item {iid}: {f} differs across copy counts"
    return f"{len(by)} items x k=0,1,2,4: base forces >= 6 N apart, new-force query, copies identical in action, outcome and measurement_id"


def check_e4(D):
    n = check_e1(D, "e4", "E4/heavy")
    a = D["e4"]
    assert (a["m"] >= S.M_HEAVY_RANGE[0] - EPS).all() and (a["m"] <= S.M_HEAVY_RANGE[1] + EPS).all()
    d = D["e4_cellD"]
    assert set(d["condition"].tolist()) == {"E4/D"} and (d["ctx_len"] == 2).all()
    cells = {"D": (2,)}
    _check_cells(d, "E4", cells, False)
    by_cube = {int(c): (m, mu) for c, m, mu in zip(a["cube_id"], a["m"], a["mu"])}
    assert all(by_cube[int(c)] == (m, mu) for c, m, mu in zip(d["cube_id"], d["m"], d["mu"]))
    return f"{n} heavy cubes (m in [1.5, 2.0]); e4_cellD has {d['item_id'].shape[0]} cell-D rows at c=2 on the same cubes"


def check_flags(D):
    out = {}
    for name in ("train", "val"):
        a = D[name]
        out[name] = {"rollouts": int(a["F"].size), "timed_out": int(a["timed_out"].sum()),
                     "rotation_gt_1deg": int((a["max_rotation_deg"] > 1.0).sum())}
    for name in TEST_SETS:
        a = D[name]
        out[name] = {"rows": int(a["item_id"].shape[0]), "query_timed_out": int(a["query_timed_out"].sum()),
                     "query_rotation_gt_1deg": int((a["query_max_rotation_deg"] > 1.0).sum())}
    return out  # flagged rollouts are kept and counted, never dropped


CHECKS = (check_schema, check_val_rows, check_ranges, check_cube_ids, check_measurement_ids, check_noise0,
          check_e1, check_e2, check_e3, check_e4, check_flags)


def run_all(D) -> list[tuple[str, object]]:
    return [(f.__name__, f(D)) for f in CHECKS]
