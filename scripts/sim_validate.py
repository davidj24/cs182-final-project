"""Stratified implementation check of the idealized 1D simulator against its declared law.

PILOT_SPEC A4: the simulator is an implementation of the declared IDEAL law (N = m g, Coulomb
friction), so agreement with the closed form is BY CONSTRUCTION. This is NOT independent validation
of MuJoCo contact physics. Nothing is tuned; every number is measured and written to a CSV.

Strata (never mixed in statistics): train (stratified, training domain), dt_half (subset of train
settings re-run at dt=0.0005 with the same physical push), t2 (subset of train settings re-run with
k doubled), ood (m log-uniform [1.5, 2.0], other parameters as in training).

Run:  python scripts/sim_validate.py [--n 2000] [--workers 16] [--seed 20261116]
Writes $CS282A_DATA/validation/sim_validate/vN/{validation.csv,meta.json} (next free N, never overwrites).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from cs182_icl.physics import G, analytic_displacement
from cs182_icl.sim.rollout import Simulator
from cs182_icl.types import DT, Action

# training domain (PILOT_SPEC C)
M_RANGE, MU_RANGE, F_RANGE, K_RANGE = (0.5, 1.5), (0.3, 0.6), (13.0, 26.0), (100, 200)
OOD_M_RANGE = (1.5, 2.0)
FINE_DT = 0.0005
ROT_FLAG_DEG = 1.0  # PROTOCOL: flag > 1 degree
COLUMNS = ["stratum", "setting_id", "m", "mu", "F", "k", "dt", "T", "ratio", "d_analytic", "d_sim",
           "abs_err", "rel_err", "timed_out", "max_rotation_deg", "rotation_flag", "t_stop",
           "seconds", "d_sim_ref", "ref_change_rel", "t2_ratio"]


def stratified_sample(n: int, seed: int, m_range=M_RANGE) -> dict[str, np.ndarray]:
    """Latin hypercube over (m log-scale, mu, F, k): each of the 4 axes has exactly one point in each
    of n equal-probability bins (the same marginals as the independent training draw). F/(mu m g) is
    derived; the packet reports its bin occupancy."""
    rng = np.random.default_rng(seed)
    u = np.stack([(rng.permutation(n) + rng.uniform(size=n)) / n for _ in range(4)], axis=1)
    m = np.exp(np.log(m_range[0]) + u[:, 0] * (np.log(m_range[1]) - np.log(m_range[0])))
    mu = MU_RANGE[0] + u[:, 1] * (MU_RANGE[1] - MU_RANGE[0])
    F = F_RANGE[0] + u[:, 2] * (F_RANGE[1] - F_RANGE[0])
    k = np.minimum(K_RANGE[0] + np.floor(u[:, 3] * (K_RANGE[1] - K_RANGE[0] + 1)).astype(int), K_RANGE[1])
    return dict(m=m, mu=mu, F=F, k=k)


def build_jobs(n: int, n_dt: int, n_t2: int, n_ood: int, seed: int) -> list[dict]:
    """All jobs (one dict per rollout). dt_half/t2 rows reference a train setting_id."""
    s = stratified_sample(n, seed)
    rng = np.random.default_rng(seed + 1)
    jobs = [dict(stratum="train", setting_id=i, m=float(s["m"][i]), mu=float(s["mu"][i]), F=float(s["F"][i]),
                 k=int(s["k"][i]), dt=DT) for i in range(n)]
    pick = rng.permutation(n)
    dt_ids, t2_ids = pick[:n_dt], pick[n_dt:n_dt + n_t2]
    for i in dt_ids:  # same physical push (F, k*0.001 s) with steps of 0.0005
        j = dict(jobs[i]); j.update(stratum="dt_half", dt=FINE_DT, k=int(round(jobs[i]["k"] * DT / FINE_DT)))
        jobs.append(j)
    for i in t2_ids:  # same cube and F, duration doubled
        j = dict(jobs[i]); j.update(stratum="t2", k=2 * jobs[i]["k"])
        jobs.append(j)
    o = stratified_sample(n_ood, seed + 2, OOD_M_RANGE)
    jobs += [dict(stratum="ood", setting_id=i, m=float(o["m"][i]), mu=float(o["mu"][i]), F=float(o["F"][i]),
                  k=int(o["k"][i]), dt=DT) for i in range(n_ood)]
    return jobs


_SIMS: dict[float, Simulator] = {}


def run_job(job: dict) -> dict:
    sim = _SIMS.setdefault(job["dt"], Simulator(job["dt"]))
    t0 = time.perf_counter()
    sim.set_cube(job["m"], job["mu"])
    res, _ = sim.rollout(Action(job["F"], job["k"]))
    secs = time.perf_counter() - t0
    T = job["k"] * job["dt"]  # analytic duration is k*dt, i.e. k_orig*0.001 for the dt-halving reruns
    an = float(analytic_displacement(job["m"], job["mu"], job["F"], T))
    row = dict(job, T=T, ratio=job["F"] / (job["mu"] * job["m"] * G), d_analytic=an, d_sim=res.d,
               abs_err=abs(res.d - an), rel_err=abs(res.d - an) / an if an > 0 else np.nan,
               timed_out=bool(res.timed_out), max_rotation_deg=res.max_rotation_deg,
               rotation_flag=bool(res.max_rotation_deg > ROT_FLAG_DEG), t_stop=res.t_stop, seconds=secs)
    return row


def _work(chunk):
    return [run_job(j) for j in chunk]


def run_validation(jobs: list[dict], workers: int) -> pd.DataFrame:
    workers = max(1, min(int(workers), 16))
    chunks = [jobs[i::workers * 8] for i in range(workers * 8)]  # interleaved chunks for balance
    chunks = [c for c in chunks if c]
    if workers == 1:
        parts = [_work(c) for c in chunks]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            parts = pool.map(_work, chunks, chunksize=1)
    df = pd.DataFrame([r for p in parts for r in p])
    df = df.sort_values(["stratum", "setting_id", "dt", "k"], kind="stable").reset_index(drop=True)
    return add_reference_columns(df)


def add_reference_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Fill d_sim_ref / ref_change_rel (dt_half) / t2_ratio (t2) from the matching train row."""
    base = df[df.stratum == "train"].set_index("setting_id")["d_sim"]
    df = df.copy()
    df["d_sim_ref"] = np.nan
    df["ref_change_rel"] = np.nan
    df["t2_ratio"] = np.nan
    for st in ("dt_half", "t2"):
        mk = df.stratum == st
        df.loc[mk, "d_sim_ref"] = df.loc[mk, "setting_id"].map(base)
    mk = df.stratum == "dt_half"
    df.loc[mk, "ref_change_rel"] = (df.loc[mk, "d_sim"] - df.loc[mk, "d_sim_ref"]).abs() / df.loc[mk, "d_sim_ref"]
    mk = df.stratum == "t2"
    df.loc[mk, "t2_ratio"] = df.loc[mk, "d_sim"] / df.loc[mk, "d_sim_ref"]
    return df[COLUMNS]


def next_version_dir(root: Path, prefix: str) -> Path:
    n = 1
    while (root / f"{prefix}{n}").exists():
        n += 1
    return root / f"{prefix}{n}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n-dt", type=int, default=50)
    ap.add_argument("--n-t2", type=int, default=20)
    ap.add_argument("--n-ood", type=int, default=100)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20261116)
    ap.add_argument("--out-root", default=None)
    a = ap.parse_args()
    data = Path(os.environ["CS282A_DATA"])
    out = next_version_dir(Path(a.out_root) if a.out_root else data / "validation" / "sim_validate", "v")
    jobs = build_jobs(a.n, a.n_dt, a.n_t2, a.n_ood, a.seed)
    # throughput benchmark first (serial, one worker, includes cube setup)
    t0 = time.perf_counter(); [run_job(j) for j in jobs[:100]]
    serial = (time.perf_counter() - t0) / 100
    t0 = time.perf_counter()
    df = run_validation(jobs, a.workers)
    wall = time.perf_counter() - t0
    out.mkdir(parents=True)
    df.to_csv(out / "validation.csv", index=False)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).parent).stdout.strip()
    meta = dict(args=vars(a), source_commit=sha, serial_seconds_per_rollout=serial, wall_seconds=wall,
                n_rows=len(df), workers=min(a.workers, 16))
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out}\n  rows={len(df)} serial {serial * 1e3:.2f} ms/rollout, wall {wall:.1f} s with {meta['workers']} workers")
    print(df.groupby("stratum").agg(n=("d_sim", "size"), rel_med=("rel_err", "median"), rel_max=("rel_err", "max"),
                                    timeouts=("timed_out", "sum")).to_string())


if __name__ == "__main__":
    main()
