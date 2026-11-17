"""Pilot dataset builders (PILOT_SPEC H, K, L, M, N; PROTOCOL 5, 6, 9). Noise is 0 (d_obs == d_clean).

Flow: sample cubes/actions with explicit seeds -> plan rows (actions + measurement ids, no
outcomes) -> one batched real-simulator call per set (simulate_batch; flagged rollouts kept)
-> pack npz arrays (docs/INTERFACES.md) -> checks.py assertions -> write files + meta.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cs182_icl.data import sampling as S
from cs182_icl.data.store import compute_normalization, content_sha256, file_sha256
from cs182_icl.types import DT, Action, Cube

P = 32  # pushes per cube
ROW_PUSHES = 11
STREAMS = {"train_cubes": 1, "train_actions": 2, "val_cubes": 3, "val_actions": 4, "val_rows": 5,
           "e1": 6, "e2": 7, "e3": 8, "e4": 9, "e4_cellD": 10}
E2_CELLS = {"A": (1, 2, 4), "B": (2, 4), "C": (2, 4), "D": (2, 4)}
E3_COPIES = (0, 1, 2, 4)
FILES = ("train", "val", "e1", "e2", "e3", "e4", "e4_cellD")


@dataclass
class Sizes:
    train: int = 1000
    val: int = 100
    e1: int = 100
    e2: int = 50
    e3: int = 50
    e4: int = 50
    val_rows: int = 1000


class Ids:
    """Allocator of measurement ids, unique within a dataset version."""

    def __init__(self):
        self.n = 0

    def new(self) -> int:
        self.n += 1
        return self.n - 1


class SimCache:
    """Real-simulator results, one rollout per unique (cube_id, F, k); flagged rollouts kept."""

    def __init__(self, workers: int, seed: int):
        self.workers, self.seed, self.cache = workers, seed, {}

    def run(self, pairs: list[tuple[Cube, Action]]):
        from cs182_icl.sim.rollout import simulate_batch

        todo = {}
        for c, a in pairs:
            todo.setdefault((c.cube_id, a.F, a.k), (c, a))
        todo = {k: v for k, v in todo.items() if k not in self.cache}
        if todo:
            res = simulate_batch(list(todo.values()), workers=self.workers, seed=self.seed)
            for key, r in zip(todo, res):
                if not np.isfinite(r.d):
                    raise RuntimeError(f"non-finite displacement for {key}")
                self.cache[key] = r
        return [self.cache[(c.cube_id, a.F, a.k)] for c, a in pairs]


def rng_for(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng([seed, STREAMS[name]])


# ---------------------------------------------------------------- train / val
def build_split(n: int, cube_id0: int, ids: Ids, sim: SimCache, rng_cube, rng_act) -> dict:
    m, mu = S.sample_cubes(rng_cube, n)
    F = S.sample_forces(rng_act, (n, P))  # drawn from a separate stream: independent of m, mu
    k = S.sample_ks(rng_act, (n, P))
    cube_id = np.arange(cube_id0, cube_id0 + n, dtype=np.int64)
    mid = np.array([[ids.new() for _ in range(P)] for _ in range(n)], dtype=np.int64)
    pairs = [(Cube(float(m[i]), float(mu[i]), int(cube_id[i])), Action(float(F[i, j]), int(k[i, j])))
             for i in range(n) for j in range(P)]
    res = sim.run(pairs)
    f = lambda attr, dt: np.array([getattr(r, attr) for r in res], dtype=dt).reshape(n, P)
    d = f("d", np.float64)
    return {"cube_id": cube_id, "m": m, "mu": mu, "F": F, "k": k, "measurement_id": mid,
            "d_clean": d, "d_obs": d.copy(), "t_stop": f("t_stop", np.float64),
            "max_rotation_deg": f("max_rotation_deg", np.float64), "timed_out": f("timed_out", bool)}


def val_rows(n_cubes: int, n_rows: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """Fixed validation rows: row_cube (R,), row_idx (R, 11) distinct pushes in random order."""
    row_cube = (np.arange(n_rows) % n_cubes).astype(np.int64)  # equal rows per cube
    row_idx = np.stack([rng.permutation(P)[:ROW_PUSHES] for _ in range(n_rows)]).astype(np.int64)
    return row_cube, row_idx


# ---------------------------------------------------------------- test-set plans
@dataclass
class RowPlan:
    item_id: int
    condition: str
    cube: Cube
    query: Action
    ctx: list = field(default_factory=list)  # [(F, k, measurement_id)]


def _cubes(rng, n, cube_id0, m_range=S.M_RANGE) -> list[Cube]:
    m, mu = S.sample_cubes(rng, n, m_range)
    return [Cube(float(m[i]), float(mu[i]), cube_id0 + i) for i in range(n)]


def _push(rng) -> tuple[float, int]:
    return float(S.sample_forces(rng, ())), int(S.sample_ks(rng, ()))


def plan_e1(cubes, rng, ids, cond="E1") -> list[RowPlan]:
    rows = []
    for i, cube in enumerate(cubes):
        ctx = [(*_push(rng), ids.new()) for _ in range(10)]
        rows.append(RowPlan(i, cond, cube, Action(*_push(rng)), ctx))
    return rows


def _item_cells(rng, ids, cube, item_id, Fq, kq, cells: dict[str, tuple], prefix="E2") -> list[RowPlan]:
    """E2 rows for one item (cells maps cell name -> context lengths). One duration multiset per
    (item, c) shared by the four cells (same-force vs distinct-force pairing, PILOT_SPEC L2)."""
    q = Action(Fq, kq)
    rows = []
    for c in sorted({c for cs in cells.values() for c in cs}):
        durs = S.distinct_ks(rng, c, exclude=kq)
        Fb = S.stratified_forces(rng, 1, S.far_intervals(Fq))[0]
        near_excl = [(S.F_RANGE[0], Fq - 1.0), (Fq + 1.0, S.F_RANGE[1])]
        pushes = {"A": [(Fq, k) for k in durs], "B": [(Fb, k) for k in durs]}
        if c >= 2:
            fc = [Fq] + S.stratified_forces(rng, c - 1, near_excl)
            fd = S.stratified_forces(rng, c, S.far_intervals(Fq))
            pushes["C"] = list(zip(fc, [durs[j] for j in rng.permutation(c)]))
            pushes["D"] = list(zip(fd, [durs[j] for j in rng.permutation(c)]))
        for cell, cs in cells.items():
            if c in cs:
                order = rng.permutation(c)  # randomized context order
                ctx = [(float(pushes[cell][j][0]), int(pushes[cell][j][1]), ids.new()) for j in order]
                rows.append(RowPlan(item_id, f"{prefix}/{cell}", cube, q, ctx))
    return rows


def plan_e2(cubes, rng, ids, cells=E2_CELLS, item_id0=0, prefix="E2") -> list[RowPlan]:
    rows = []
    for i, cube in enumerate(cubes):
        Fq = float(rng.uniform(*S.FQ_RANGE))
        kq = int(S.sample_ks(rng, ()))
        rows += _item_cells(rng, ids, cube, item_id0 + i, Fq, kq, cells, prefix)
    return rows


def plan_e3(cubes, rng, ids) -> list[RowPlan]:
    rows = []
    for i, cube in enumerate(cubes):
        Fq = float(rng.uniform(*S.FQ_RANGE))
        kq = int(S.sample_ks(rng, ()))
        lo = float(rng.uniform(S.F_RANGE[0], Fq - S.NEAR))
        hi = float(rng.uniform(Fq + S.NEAR, S.F_RANGE[1]))
        ks = S.distinct_ks(rng, 2, exclude=kq)
        base = [(lo, ks[0], ids.new()), (hi, ks[1], ids.new())]
        base = [base[j] for j in rng.permutation(2)]
        for n in E3_COPIES:
            copies = [base[j % 2] for j in range(n)]  # identical action, ... and measurement id
            rows.append(RowPlan(i, f"E3/k={n}", cube, Action(Fq, kq), base + copies))
    return rows


# ---------------------------------------------------------------- pack
def pack_rows(rows: list[RowPlan], sim: SimCache) -> tuple[dict, dict]:
    """Resolve outcomes with the simulator and pack into the e*.npz layout. Returns (arrays, flags)."""
    pairs = []
    for r in rows:
        pairs.append((r.cube, r.query))
        pairs += [(r.cube, Action(F, k)) for F, k, _ in r.ctx]
    res = iter(sim.run(pairs))
    I, cmax = len(rows), max(len(r.ctx) for r in rows)
    z = lambda dt, fill: np.full((I, cmax), fill, dtype=dt)
    A = {"item_id": np.array([r.item_id for r in rows], np.int64),
         "condition": np.array([r.condition for r in rows], dtype="<U32"),
         "cube_id": np.array([r.cube.cube_id for r in rows], np.int64),
         "m": np.array([r.cube.m for r in rows]), "mu": np.array([r.cube.mu for r in rows]),
         "query_F": np.array([r.query.F for r in rows]), "query_k": np.array([r.query.k for r in rows], np.int64),
         "ctx_len": np.array([len(r.ctx) for r in rows], np.int64),
         "ctx_F": z(np.float64, np.nan), "ctx_d_clean": z(np.float64, np.nan), "ctx_d_obs": z(np.float64, np.nan),
         "ctx_k": z(np.int64, -1), "ctx_measurement_id": z(np.int64, -1)}
    q = {"d_query_clean": [], "query_t_stop": [], "query_max_rotation_deg": [], "query_timed_out": []}
    seen_obs, n_to, n_rot = {}, 0, 0
    for i, r in enumerate(rows):
        rq = next(res)
        q["d_query_clean"].append(rq.d); q["query_t_stop"].append(rq.t_stop)
        q["query_max_rotation_deg"].append(rq.max_rotation_deg); q["query_timed_out"].append(rq.timed_out)
        n_to += rq.timed_out; n_rot += rq.max_rotation_deg > 1.0
        for j, (F, k, mid) in enumerate(r.ctx):
            ro = next(res)
            A["ctx_F"][i, j], A["ctx_k"][i, j], A["ctx_measurement_id"][i, j] = F, k, mid
            A["ctx_d_clean"][i, j] = ro.d
            A["ctx_d_obs"][i, j] = ro.d  # noise 0
            if mid not in seen_obs:
                seen_obs[mid] = 1
                n_to += ro.timed_out; n_rot += ro.max_rotation_deg > 1.0
    A["d_query_clean"] = np.array(q["d_query_clean"])
    A["query_t_stop"] = np.array(q["query_t_stop"])
    A["query_max_rotation_deg"] = np.array(q["query_max_rotation_deg"])
    A["query_timed_out"] = np.array(q["query_timed_out"], bool)
    flags = {"rows": I, "unique_context_measurements": len(seen_obs), "timed_out_counted": int(n_to),
             "rotation_gt_1deg_counted": int(n_rot)}
    return A, flags


# ---------------------------------------------------------------- orchestration
def build_arrays(sizes: Sizes, seed: int, workers: int) -> tuple[dict, dict]:
    """All files as {name: arrays} plus extra info (flag counts, cube id ranges)."""
    ids, sim = Ids(), SimCache(workers, seed)
    out, info = {}, {"flags": {}, "cube_id_ranges": {}}
    nxt = 0

    def take(n):
        nonlocal nxt
        a = nxt
        nxt += n
        return a

    c0 = take(sizes.train)
    out["train"] = build_split(sizes.train, c0, ids, sim, rng_for(seed, "train_cubes"), rng_for(seed, "train_actions"))
    c0 = take(sizes.val)
    out["val"] = build_split(sizes.val, c0, ids, sim, rng_for(seed, "val_cubes"), rng_for(seed, "val_actions"))
    out["val"]["row_cube"], out["val"]["row_idx"] = val_rows(sizes.val, sizes.val_rows, rng_for(seed, "val_rows"))
    for name in ("train", "val"):
        a = out[name]
        info["flags"][name] = {"rollouts": int(a["F"].size), "timed_out": int(a["timed_out"].sum()),
                               "rotation_gt_1deg": int((a["max_rotation_deg"] > 1.0).sum())}
    rng = rng_for(seed, "e1")  # one stream: cubes first, then actions (independent draws)
    e1_cubes = _cubes(rng, sizes.e1, take(sizes.e1))
    out["e1"] = pack_rows(plan_e1(e1_cubes, rng, ids), sim)
    rng = rng_for(seed, "e2")  # one stream: cubes first, then actions (independent draws)
    e2_cubes = _cubes(rng, sizes.e2, take(sizes.e2))
    out["e2"] = pack_rows(plan_e2(e2_cubes, rng, ids), sim)
    rng = rng_for(seed, "e3")  # one stream: cubes first, then actions (independent draws)
    e3_cubes = _cubes(rng, sizes.e3, take(sizes.e3))
    out["e3"] = pack_rows(plan_e3(e3_cubes, rng, ids), sim)
    rng = rng_for(seed, "e4")
    e4_cubes = _cubes(rng, sizes.e4, take(sizes.e4), S.M_HEAVY_RANGE)
    out["e4"] = pack_rows(plan_e1(e4_cubes, rng, ids, cond="E4/heavy"), sim)
    # E4 cell D at c=2 on the same heavy cubes; separate file because eval prefix_cases needs 10-push
    # contexts in every e4.npz row. item_id offset marks that the query differs from e4.npz's query.
    out["e4_cellD"] = pack_rows(plan_e2(e4_cubes, rng_for(seed, "e4_cellD"), ids, cells={"D": (2,)},
                                        item_id0=5000, prefix="E4"), sim)
    for name in FILES[2:]:
        out[name], fl = out[name]
        info["flags"][name] = fl
    ranges = {"train": (0, sizes.train)}
    ranges["val"] = (sizes.train, sizes.train + sizes.val)
    for name, cubes in (("e1", e1_cubes), ("e2", e2_cubes), ("e3", e3_cubes), ("e4", e4_cubes)):
        ranges[name] = (cubes[0].cube_id, cubes[-1].cube_id + 1)
    info["cube_id_ranges"] = {k: list(v) for k, v in ranges.items()}
    info["n_unique_rollouts"] = len(sim.cache)
    return out, info


def write_dataset(out_dir: Path, arrays: dict, info: dict, *, seed: int, workers: int, version: str,
                  provenance: dict, config_path: Path | None, sizes: Sizes) -> dict:
    """Run all checks on the arrays, write the npz files and meta.json into the NEW dir out_dir."""
    from cs182_icl.data import checks

    results = checks.run_all(arrays)  # raises AssertionError on any violation
    out_dir.mkdir(parents=True, exist_ok=False)
    for name, a in arrays.items():
        np.savez(out_dir / f"{name}.npz", **a)
    t = arrays["train"]
    meta = {
        "version": version,
        "dataset_version": version,
        "pilot_config_path": str(config_path) if config_path else None,
        "pilot_config_sha256": file_sha256(config_path) if config_path else None,
        "generation_seed": seed, "sim_workers": workers, **provenance,
        "dt": DT, "g": 9.81, "noise_sigma_d": 0.0,
        "ranges": {"m": list(S.M_RANGE), "m_heavy": list(S.M_HEAVY_RANGE), "mu": list(S.MU_RANGE),
                   "F": list(S.F_RANGE), "k": list(S.K_RANGE), "E2_E3_query_F": list(S.FQ_RANGE),
                   "near_N": S.NEAR, "e3_base_separation_N": S.E3_SEP},
        "sizes": sizes.__dict__,
        "counts": {n: int(a["cube_id"].shape[0]) if n in ("train", "val") else int(a["item_id"].shape[0])
                   for n, a in arrays.items()},
        "cube_id_ranges": info["cube_id_ranges"],
        "npz_sha256": {f"{n}.npz": file_sha256(out_dir / f"{n}.npz") for n in arrays},
        "content_sha256": {f"{n}.npz": content_sha256(a) for n, a in arrays.items()},
        "flags": info["flags"], "n_unique_rollouts": info["n_unique_rollouts"],
        "normalization": compute_normalization(t["F"], t["k"], t["d_obs"]),
        "normalization_source": "train.npz only",
        "checks_passed": [r[0] for r in results],
    }
    import hashlib

    meta["dataset_hash"] = hashlib.sha256(
        "".join(f"{k}:{v}\n" for k, v in sorted(meta["content_sha256"].items())).encode()).hexdigest()
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def next_version_dir(root: Path, prefix: str = "pilot_v") -> tuple[str, Path]:
    n = 1
    while any((root / f"{prefix}{n}{s}").exists() for s in ("", ".tmp")):
        n += 1
    return f"{prefix}{n}", root / f"{prefix}{n}"


def build_dataset(out_dir: Path, *, sizes: Sizes, seed: int, workers: int, version: str, provenance: dict,
                  config_path: Path | None = None) -> dict:
    arrays, info = build_arrays(sizes, seed, workers)
    return write_dataset(out_dir, arrays, info, seed=seed, workers=workers, version=version,
                         provenance=provenance, config_path=config_path, sizes=sizes)
