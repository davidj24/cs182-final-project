"""SYNTHETIC items for harness tests. Targets come from physics.analytic_displacement, not
MuJoCo; nothing here says anything about the real simulator or any trained model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cs182_icl.eval.cases import Case, prefix_cases
from cs182_icl.physics import analytic_displacement, same_force_displacement, line_value, params_from_line
from cs182_icl.types import Action, Cube, Observation, Prediction, TestItem

_mid = iter(range(10**6, 10**9))


def _obs(m, mu, F, k):
    d = float(analytic_displacement(m, mu, F, k * 0.001))
    return Observation(Action(float(F), int(k)), next(_mid), d, d)


def _item(cube, F, k, ctx, cond):
    d = float(analytic_displacement(cube.m, cube.mu, F, k * 0.001))
    return TestItem(cube, Action(float(F), int(k)), d, list(ctx), cond)


def _cube(rng, cid, mlo=0.3, mhi=1.0):
    m = float(np.exp(rng.uniform(np.log(mlo), np.log(mhi))))
    return Cube(m, float(rng.uniform(0.3, 0.8)), cid)


def make_e1_rows(n=12, seed=0, mlo=0.3, mhi=1.0, cond="E1"):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        cube = _cube(rng, i, mlo, mhi)
        ctx = [_obs(cube.m, cube.mu, rng.uniform(13, 26), rng.integers(100, 400)) for _ in range(10)]
        rows.append((i, _item(cube, rng.uniform(13, 26), rng.integers(100, 400), ctx, cond)))
    return rows


def make_e2_cases(n=8, seed=1):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        cube = _cube(rng, i)
        Fq, kq = float(rng.uniform(17, 22)), int(rng.integers(100, 400))
        ks = [int(k) for k in rng.integers(100, 400, size=4)]
        forces = {
            "A": [Fq] * 4,
            "B": [Fq - 4.0] * 4,
            "C": [Fq, Fq - 5.0, Fq + 4.0, Fq - 3.5],
            "D": [Fq - 4.0, Fq + 4.0, Fq - 6.0, Fq + 5.5],
        }
        for cell, fs in forces.items():
            ctx = [_obs(cube.m, cube.mu, f, k) for f, k in zip(fs, ks)]
            cs = (1, 2, 4) if cell == "A" else (2, 4)
            cases += prefix_cases([(i, _item(cube, Fq, kq, ctx, f"E2/{cell}"))], cs)
    return cases


def make_e3_cases(n=8, seed=2, kinds=(0, 1, 2, 4)):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        cube = _cube(rng, i)
        Fq, kq = float(rng.uniform(17, 22)), int(rng.integers(100, 400))
        base = [_obs(cube.m, cube.mu, Fq - 4.0, rng.integers(100, 400)),
                _obs(cube.m, cube.mu, Fq + 3.5, rng.integers(100, 400))]
        for k in kinds:
            copies = [base[j % 2] for j in range(k)]  # same object: same ID and value
            item = _item(cube, Fq, kq, base + copies, f"E3/k={k}")
            cases.append(Case(i, len(item.context), item))
    return cases


def stub_physics_fit(context, query):
    """Synthetic physics-fit-like estimator for tests: not_determined unless the context
    identifies the query (same-force observation, or two distinct forces -> line fit)."""
    same = [o for o in context if o.action.F == query.F]
    if same:
        o = same[0]
        return Prediction(float(same_force_displacement(o.d_obs, o.action.T, query.T)), "ok")
    forces = sorted({o.action.F for o in context})
    if len(forces) < 2:
        return Prediction(None, "not_determined", "fewer than two distinct forces")
    o1 = next(o for o in context if o.action.F == forces[0])
    o2 = next(o for o in context if o.action.F == forces[-1])
    y1, y2 = line_value(o1.d_obs, o1.action.F, o1.action.T), line_value(o2.d_obs, o2.action.F, o2.action.T)
    A = (y2 - y1) / (o2.action.F - o1.action.F)
    B = A * o1.action.F - y1
    if A <= 0 or B <= 0:
        return Prediction(None, "not_determined", "degenerate line")
    m, mu = params_from_line(A, B)
    return Prediction(float(analytic_displacement(m, mu, query.F, query.T)), "ok")


def write_npz(path: Path, rows, cmax=10):
    n = len(rows)
    z = dict(
        item_id=np.array([i for i, _ in rows]), condition=np.array([it.condition for _, it in rows]),
        cube_id=np.array([it.cube.cube_id for _, it in rows]), m=np.array([it.cube.m for _, it in rows]),
        mu=np.array([it.cube.mu for _, it in rows]), query_F=np.array([it.query.F for _, it in rows]),
        query_k=np.array([it.query.k for _, it in rows]),
        d_query_clean=np.array([it.d_query_clean for _, it in rows]),
        ctx_len=np.array([len(it.context) for _, it in rows]),
        ctx_F=np.full((n, cmax), np.nan), ctx_d_clean=np.full((n, cmax), np.nan),
        ctx_d_obs=np.full((n, cmax), np.nan), ctx_k=np.full((n, cmax), -1),
        ctx_measurement_id=np.full((n, cmax), -1),
    )
    for r, (_, it) in enumerate(rows):
        for j, o in enumerate(it.context):
            z["ctx_F"][r, j], z["ctx_k"][r, j] = o.action.F, o.action.k
            z["ctx_d_clean"][r, j], z["ctx_d_obs"][r, j] = o.d_clean, o.d_obs
            z["ctx_measurement_id"][r, j] = o.measurement_id
    np.savez(path, **z)


def write_dataset(dirpath: Path):
    """Synthetic dataset dir with e1-e4.npz + meta.json (normalization d_obs std = 0.05)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    write_npz(dirpath / "e1.npz", make_e1_rows(10, 0))
    write_npz(dirpath / "e4.npz", make_e1_rows(6, 5, 1.5, 2.0, "E4/heavy"))
    for name, cases in (("e2", make_e2_cases(4)), ("e3", make_e3_cases(4))):
        write_npz(dirpath / f"{name}.npz", [(c.item_id, c.item) for c in cases], cmax=10)
    (dirpath / "meta.json").write_text(json.dumps({"normalization": {"d_obs": {"mean": 0.0, "std": 0.05}}}))
