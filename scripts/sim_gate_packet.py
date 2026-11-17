"""Build the H2 gate packet (figures + README.md) from a sim_validate.py CSV.

Numbers only: no ranges / target space / sigma_d decisions here. R1 inputs and R2 candidates are
printed as CANDIDATES; control applies PREAPPROVAL R1/R2.
Run: python scripts/sim_gate_packet.py --csv <validation.csv> [--meta <meta.json>]
Writes $CS282A_DATA/gates/H2/attempt-NN (next free NN; never overwrites).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PILOT_ROLLOUTS_FIXED = 1000 * 32 + 100 * 32  # training + validation cubes x 32 pushes
TEST_ITEMS = 100 + 50 + 50 + 50  # E1..E4 items from the pilot config
ROW_PUSHES = 11  # pushes per test item (config row_pushes); upper-bound assumption, T04 owns the real count
F_SPLIT = 0.1  # m (PREAPPROVAL R2)
TAU_FLOOR = 1e-4


def round_up_sig(x: float, sig: int = 2) -> float:
    """Round up to `sig` significant figures (0 stays 0)."""
    if x <= 0 or not np.isfinite(x):
        return float(x)
    e = math.floor(math.log10(x)) - (sig - 1)
    return float(math.ceil(round(x / 10.0**e, 9)) * 10.0**e)


def q(x, p):
    return float(np.percentile(np.asarray(x, float), p))


def r1_stats(df: pd.DataFrame) -> dict:
    """PREAPPROVAL R1 inputs: train-domain MOVING pushes (analytic d > 0) only; dt_half uses its own rows."""
    tr = df[(df.stratum == "train") & (df.d_analytic > 0)]
    dh = df[df.stratum == "dt_half"]
    out = dict(n_train_moving=len(tr), rel_median=float(tr.rel_err.median()), rel_p95=q(tr.rel_err, 95),
               rel_max=float(tr.rel_err.max()), n_dt=len(dh))
    out["dt_change_median"] = float(dh.ref_change_rel.median()) if len(dh) else float("nan")
    out["dt_change_p95"] = q(dh.ref_change_rel, 95) if len(dh) else float("nan")
    out["meets_R1_thresholds"] = bool(out["rel_median"] <= 0.01 and out["rel_p95"] <= 0.03
                                      and out["dt_change_median"] <= 0.005 and out["dt_change_p95"] <= 0.02)
    return out


def r2_candidates(df: pd.DataFrame) -> dict:
    """delta = sim - f over train-domain moving pushes. r_model = RMS(delta/f) over f >= 0.1 m;
    tau_abs = RMS(delta) over f < 0.1 m floored at 1e-4 m; both rounded up to 2 sig figs."""
    tr = df[(df.stratum == "train") & (df.d_analytic > 0)]
    f, delta = tr.d_analytic.to_numpy(), (tr.d_sim - tr.d_analytic).to_numpy()
    hi, lo = f >= F_SPLIT, f < F_SPLIT
    r_raw = float(np.sqrt(np.mean((delta[hi] / f[hi]) ** 2))) if hi.any() else float("nan")
    t_raw = float(np.sqrt(np.mean(delta[lo] ** 2))) if lo.any() else 0.0
    return dict(n_f_ge_0p1=int(hi.sum()), n_f_lt_0p1=int(lo.sum()), r_model_raw=r_raw,
                r_model_candidate=round_up_sig(r_raw), tau_abs_raw=t_raw,
                tau_abs_candidate=round_up_sig(max(t_raw, TAU_FLOOR)))


def quantile_row(x) -> dict:
    return dict(n=len(x), median=q(x, 50), p95=q(x, 95), p99=q(x, 99), max=float(np.max(x)))


def ratio_bins(df: pd.DataFrame, nb: int = 8) -> pd.DataFrame:
    tr = df[df.stratum == "train"]
    edges = np.exp(np.linspace(np.log(tr.ratio.min()), np.log(tr.ratio.max()) + 1e-12, nb + 1))
    b = pd.cut(tr.ratio, edges, include_lowest=True)
    g = tr.groupby(b, observed=False).agg(n=("rel_err", "size"), rel_median=("rel_err", "median"),
                                          rel_max=("rel_err", "max"))
    g.index = [f"{i.left:.2f}-{i.right:.2f}" for i in g.index]
    return g


def figures(df: pd.DataFrame, out: Path) -> list[str]:
    tr, ood = df[df.stratum == "train"], df[df.stratum == "ood"]
    names = []

    def save(name):
        plt.tight_layout(); plt.savefig(out / name, dpi=130); plt.close(); names.append(name)

    plt.figure(figsize=(5, 5))
    plt.loglog(tr.d_analytic, tr.d_sim, ".", ms=3, label="training domain")
    plt.loglog(ood.d_analytic, ood.d_sim, ".", ms=3, label="heavy OOD (separate stratum)")
    lim = [min(tr.d_analytic.min(), ood.d_analytic.min()), max(tr.d_analytic.max(), ood.d_analytic.max())]
    plt.loglog(lim, lim, "k-", lw=0.6)
    plt.xlabel("analytic d (m)"); plt.ylabel("simulator d (m)"); plt.legend()
    plt.title("sim vs analytic (agreement by construction)"); save("scatter_loglog.png")

    for col, name, lab in (("abs_err", "hist_abs.png", "|sim - analytic| (m)"),
                           ("rel_err", "hist_rel.png", "|sim - analytic| / analytic")):
        plt.figure(figsize=(5, 3.5))
        for d, l in ((tr, "training"), (ood, "heavy OOD")):
            plt.hist(np.log10(np.clip(d[col], 1e-16, None)), bins=40, alpha=0.6, label=l, density=True)
        plt.xlabel(f"log10 {lab}"); plt.ylabel("density"); plt.legend(); save(name)

    for x, name, lab in (("ratio", "err_vs_ratio.png", "F / (mu m g)"), ("d_analytic", "err_vs_d.png", "analytic d (m)")):
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
        for a, col, yl in zip(ax, ("rel_err", "abs_err"), ("relative error", "absolute error (m)")):
            a.semilogy(tr[x], np.clip(tr[col], 1e-16, None), ".", ms=3, label="training")
            a.semilogy(ood[x], np.clip(ood[col], 1e-16, None), ".", ms=3, label="heavy OOD")
            a.set_xlabel(lab); a.set_ylabel(yl)
        ax[0].legend(); save(name)
    plt.figure(figsize=(5, 3.5))
    plt.hist(tr.t_stop, bins=40); plt.xlabel("t_stop (s), training"); save("hist_tstop.png")
    plt.figure(figsize=(5, 3.5))
    plt.hist(tr.d_sim, bins=40); plt.xlabel("simulator d (m), training"); save("hist_d.png")
    return names


G1_TABLE = """| G1 check | Shown by (tests/sim/test_mechanics.py unless noted) |
|---|---|
| side 0.05 m, half-size 0.025 m | `test_geometry_and_model` |
| mass updated; inertia m s^2/6 on every axis | `test_mass_inertia_and_friction_force` |
| `mj_setConst` called after mass/inertia change | `test_setconst_called_and_derived_change` |
| floor friction does not override cube friction / readout = mu | `test_no_contact_influence` (contacts are disabled, so this is true by construction); `test_mass_inertia_and_friction_force` (Coulomb force = mu m g) |
| settles before every push | `test_initial_state_exactly_at_rest` (trivial: the state is exactly at rest, no settling dynamics) |
| force applied exactly k steps, then zeroed | `test_force_exactly_k_steps_then_zero`, `test_force_steps_at_half_dt` |
| zero-force creep negligible | `test_zero_force_creep`, `test_below_threshold_does_not_move`, `test_stays_at_rest_after_stopping` |
| stopping conditions (linear only; no rotation DoF) | `test_stop_conditions`, `test_linear_condition_blocks_stop` |
| max rotation recorded | `test_max_rotation_is_vacuous_zero` (always 0.0: vacuous) |
| timeouts recorded; no flagged rollout dropped | `test_timeout_flag_returned`, `test_flagged_rollouts_in_batch_not_dropped` |
| push accel (F-mu m g)/m and coast decel -mu g | `test_push_acceleration_and_coast_deceleration` |
| displacement vs closed form, T^2, dt rerun | `test_matches_analytic_displacement`, `test_t_squared_scaling`, `test_dt_rerun_consistent`; and this packet's CSV |
| frames/replay for light, middle, heavy cases | `scripts/sim_view.py` (T01), not re-run by T02 |
"""


def md_table(df: pd.DataFrame, fmt="{:.3e}") -> str:
    d = df.reset_index()
    if "n" in d.columns:
        d["n"] = d["n"].astype(int)
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    rows = ["| " + " | ".join(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |" for r in d.to_numpy()]
    return head + "\n".join(rows) + "\n"


def build_readme(df: pd.DataFrame, meta: dict | None, figs: list[str], csv: str) -> str:
    tr, ood = df[df.stratum == "train"], df[df.stratum == "ood"]
    r1, r2 = r1_stats(df), r2_candidates(df)
    qt = pd.DataFrame({"train rel": quantile_row(tr.rel_err.dropna()), "train abs (m)": quantile_row(tr.abs_err),
                       "OOD rel": quantile_row(ood.rel_err.dropna()), "OOD abs (m)": quantile_row(ood.abs_err)}).T
    dh, t2 = df[df.stratum == "dt_half"], df[df.stratum == "t2"]
    dh_t = pd.DataFrame({"dt_half rel change in d vs dt=0.001": quantile_row(dh.ref_change_rel),
                         "dt_half rel err vs analytic": quantile_row(dh.rel_err)}).T
    t2_dev = (t2.t2_ratio / 4 - 1).abs()
    t2_t = pd.DataFrame({"ratio d(2k)/d(k)": quantile_row(t2.t2_ratio), "rel deviation of ratio from 4": quantile_row(t2_dev)}).T
    flags = df.groupby("stratum").agg(n=("d_sim", "size"), timeouts=("timed_out", "sum"),
                                      rotation_flagged=("rotation_flag", "sum"),
                                      max_rotation_deg=("max_rotation_deg", "max"))
    sec = df.seconds.mean()
    ser = meta["serial_seconds_per_rollout"] if meta else sec
    wall_per = meta["wall_seconds"] / meta["n_rows"] if meta else float("nan")
    workers = meta["workers"] if meta else 16
    n_roll = PILOT_ROLLOUTS_FIXED + TEST_ITEMS * ROW_PUSHES
    ts = df[(df.stratum == "train")].t_stop
    rb = ratio_bins(df)
    return f"""# H2 gate packet: simulator validation (pilot initial_e2e_2026_11_16)

Source CSV: `{csv}`  (settings: {len(tr)} training-domain stratified, {len(dh)} dt=0.0005, {len(t2)} T^2, {len(ood)} heavy-OOD m in [1.5, 2.0]).
Generated by `scripts/sim_gate_packet.py`. No ranges, target space, sigma_d or Bayes decisions are made here.

## Read this first
**Agreement between the simulator and the analytic model here is by construction.** The simulator
(PILOT_SPEC A4) is an implementation of the declared IDEAL law: 1D slide, Coulomb friction mu m g (N = m g) and
the push force applied explicitly each step. It is NOT an independent validation of MuJoCo contact physics, and the
numbers below cannot show that real pushes obey N = m g or Coulomb friction. What they can show: the code
implements the ideal law with small explicit-Euler discretisation error, and that dt, stiction, T^2 and the batch
machinery behave. Realistic contact is a later relaxation.

## Judgment 1: physical-simulator validity (PILOT_SPEC G3)
Scope: does the code implement the intended mechanics (force timing, friction, rest state, stopping, flags)?
G1 checklist and what shows each (the test run is `pytest tests/sim/test_mechanics.py`; this packet does not rerun it):

{G1_TABLE}
Shown here: the stratified run has {int(flags.timeouts.sum())} timeouts and {int(flags.rotation_flagged.sum())} rotation flags; max rotation is
{df.max_rotation_deg.max():.1f} deg for every row because the scene has no rotational DoF (vacuous, not evidence of stability).
Not shown: contact friction readout, floor/cube friction combination, settling under gravity, rotation behaviour. Those
are replaced by the ideal-law checks (A4). Judgment for control: do the mechanics tests above and the zero timeout count meet G1
for the idealized model? The packet does not claim more.

## Judgment 2: analytic-model validity (PILOT_SPEC G3)
Scope: does the closed form describe the simulator data? Measured discrepancy, training domain, moving pushes
(sim vs analytic, n={r1['n_train_moving']}): relative median {r1['rel_median']:.3e}, p95 {r1['rel_p95']:.3e}, max {r1['rel_max']:.3e}.
This tiny discrepancy is by construction (same law) plus O(dt) discretisation, so it says nothing about real-contact
data. Per G3 the result packet should still distinguish: MuJoCo clean target / simulator oracle; analytic
true-parameter prediction; analytic discrepancy (the CSV columns `d_sim`, `d_analytic`, `abs_err`/`rel_err`).

## Quantiles (training domain and heavy OOD are separate strata, never pooled)
{md_table(qt)}
By F/(mu m g) (training, 8 log bins; F>mu m g always in the training domain, so every push moves):
{md_table(rb)}
## Numbers PREAPPROVAL R1 needs (training-domain MOVING pushes only)
- n = {r1['n_train_moving']}; median rel = {r1['rel_median']:.3e}; p95 rel = {r1['rel_p95']:.3e} (R1: <= 1e-2 and <= 3e-2)
- dt-halving (n={r1['n_dt']}): median |change in d| = {r1['dt_change_median']:.3e}; p95 = {r1['dt_change_p95']:.3e} (R1: <= 5e-3 and <= 2e-2)
- thresholds met as numbers: {r1['meets_R1_thresholds']} (control applies R1; informational only)

## R2 CANDIDATES (control decides; not adopted here)
delta = sim - f; f = analytic d. Moving training pushes with f >= 0.1 m: {r2['n_f_ge_0p1']}; with f < 0.1 m: {r2['n_f_lt_0p1']}.
- r_model = RMS(delta/f) over f >= 0.1 m: raw {r2['r_model_raw']:.3e}, rounded up to 2 sig figs: **{r2['r_model_candidate']:.2g}**
- tau_abs = RMS(delta) over f < 0.1 m (floor 1e-4 m): raw {r2['tau_abs_raw']:.3e}, with floor, rounded up to 2 sig figs: **{r2['tau_abs_candidate']:.2g}**
  ({'no pushes with f < 0.1 m, so tau_abs is the floor' if r2['n_f_lt_0p1'] == 0 else 'from the f < 0.1 m pushes'})

## dt-halving (dt = 0.0005, same physical push, analytic T = k_orig * 0.001)
{md_table(dh_t)}
## T^2 scaling (same cube and F, k doubled; closed-form ratio is exactly 4)
{md_table(t2_t)}
## Flags per stratum (rotation is vacuous: no rotational DoF)
{md_table(flags, '{:.3g}')}
## Timing and projection
- seconds per rollout (mean, in-worker, incl. cube setup): {sec:.4f}; serial benchmark (100 rollouts): {ser:.4f}; wall per rollout with {workers} workers: {wall_per:.5f}
- training t_stop: median {ts.median():.2f} s, p95 {q(ts, 95):.2f} s, max {ts.max():.2f} s
- pilot rollouts: {PILOT_ROLLOUTS_FIXED} (1,000 train + 100 val cubes x 32 pushes) + test items {TEST_ITEMS} x {ROW_PUSHES} pushes (E1..E4 items x row_pushes: an assumption, T04 owns the real count) = {n_roll}
- projected simulation time: serial {n_roll * ser / 60:.1f} min; with {workers} workers about {n_roll * ser / workers / 60:.2f} min (ideal scaling; measured wall/rollout above gives {n_roll * wall_per / 60:.2f} min)
- median training displacement d: {tr.d_sim.median():.4f} m (p5 {q(tr.d_sim, 5):.4f}, p95 {q(tr.d_sim, 95):.4f})

## Figures
{chr(10).join(f'- `{f}`' for f in figs)}
"""


def next_attempt(root: Path) -> Path:
    n = 1
    while (root / f"attempt-{n:02d}").exists():
        n += 1
    return root / f"attempt-{n:02d}"


def build_packet(csv: Path, meta_path: Path | None, out: Path) -> Path:
    df = pd.read_csv(csv)
    meta = json.loads(meta_path.read_text()) if meta_path and meta_path.exists() else None
    out.mkdir(parents=True)
    figs = figures(df, out)
    (out / "README.md").write_text(build_readme(df, meta, figs, str(csv)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--root", default=None, help="default $CS282A_DATA/gates/H2")
    a = ap.parse_args()
    root = Path(a.root) if a.root else Path(os.environ["CS282A_DATA"]) / "gates" / "H2"
    csv = Path(a.csv)
    meta = Path(a.meta) if a.meta else csv.with_name("meta.json")
    out = build_packet(csv, meta, next_attempt(root))
    df = pd.read_csv(csv)
    print(f"wrote {out}")
    print("R1:", json.dumps(r1_stats(df), indent=1))
    print("R2 CANDIDATES (control decides):", json.dumps(r2_candidates(df), indent=1))


if __name__ == "__main__":
    main()
