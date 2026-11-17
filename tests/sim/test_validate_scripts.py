"""Fast tests for scripts/sim_validate.py and scripts/sim_gate_packet.py (stratification, schema,
R1/R2 statistics on a hand-made table, packet builder on a tiny sample)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # needed so multiprocessing can pickle the worker functions
    spec.loader.exec_module(mod)
    return mod


V = _load("sim_validate")
P = _load("sim_gate_packet")


def test_stratified_sample_covers_every_bin_and_is_seeded():
    n = 200
    s = V.stratified_sample(n, seed=3)
    for key, (lo, hi), log in (("m", V.M_RANGE, True), ("mu", V.MU_RANGE, False), ("F", V.F_RANGE, False)):
        x = np.log(s[key]) if log else s[key]
        lo, hi = (np.log(lo), np.log(hi)) if log else (lo, hi)
        counts = np.bincount(np.minimum(((x - lo) / (hi - lo) * n).astype(int), n - 1), minlength=n)
        assert (counts == 1).all(), key  # Latin hypercube: exactly one point per bin
    assert s["k"].min() >= 100 and s["k"].max() <= 200
    assert np.bincount(s["k"] - 100).min() >= 1  # 101 values, 200 samples: every k value is hit
    again = V.stratified_sample(n, seed=3)
    assert all((s[k] == again[k]).all() for k in s)
    ood = V.stratified_sample(50, seed=1, m_range=V.OOD_M_RANGE)
    assert ood["m"].min() >= 1.5 and ood["m"].max() <= 2.0


def test_ratio_covers_wide_range():
    s = V.stratified_sample(500, seed=0)
    r = s["F"] / (s["mu"] * s["m"] * V.G)
    assert r.min() > 1.0 and r.min() < 2.5 and r.max() > 10


@pytest.fixture(scope="module")
def tiny_df():
    jobs = V.build_jobs(n=12, n_dt=3, n_t2=2, n_ood=4, seed=5)
    return V.run_validation(jobs, workers=2)


def test_csv_schema_and_strata(tiny_df):
    df = tiny_df
    assert list(df.columns) == V.COLUMNS
    assert df.groupby("stratum").size().to_dict() == {"dt_half": 3, "ood": 4, "t2": 2, "train": 12}
    assert (df[df.stratum == "ood"].m >= 1.5).all() and (df[df.stratum == "train"].m <= 1.5).all()
    assert df.seconds.gt(0).all() and not df.timed_out.any()
    assert (df[df.stratum == "dt_half"].dt == V.FINE_DT).all()
    t2 = df[df.stratum == "t2"]
    assert np.allclose(t2.t2_ratio, 4.0, rtol=1e-4)
    assert df[df.stratum == "dt_half"].ref_change_rel.lt(1e-3).all()
    assert df.rel_err.max() < 1e-3  # implementation check of the ideal law
    # dt_half analytic T is k_orig*0.001 (== k_run*dt)
    dh = df[df.stratum == "dt_half"]
    assert np.allclose(dh["T"], dh.k * V.FINE_DT)


def _hand_table():
    rows = []
    def add(stratum, f, sim, ref=np.nan, sid=0):
        rows.append(dict(stratum=stratum, setting_id=sid, d_analytic=f, d_sim=sim, rel_err=abs(sim - f) / f,
                         abs_err=abs(sim - f), ref_change_rel=ref))
    # f >= 0.1: rel errors 0.01 and 0.03 -> RMS = sqrt((1e-4 + 9e-4)/2)
    add("train", 1.0, 1.01); add("train", 2.0, 2.0 * 1.03)
    # f < 0.1: deltas 2e-4 and 2e-4 -> RMS 2e-4 (above the floor)
    add("train", 0.05, 0.0502); add("train", 0.02, 0.0202)
    add("train", 3.0, 3.0)
    add("ood", 1.0, 5.0)  # must never enter training statistics
    add("dt_half", 1.0, 1.0, ref=0.001); add("dt_half", 1.0, 1.0, ref=0.003)
    return pd.DataFrame(rows)


def test_r1_stats_hand_table():
    r = P.r1_stats(_hand_table())
    assert r["n_train_moving"] == 5
    assert r["rel_median"] == pytest.approx(np.median([0.01, 0.03, 0.004, 0.01, 0.0]))
    assert r["rel_max"] == pytest.approx(0.03)  # the OOD row (rel 4.0) is excluded
    assert r["dt_change_median"] == pytest.approx(0.002)
    assert r["meets_R1_thresholds"] is True


def test_r2_candidates_hand_table():
    c = P.r2_candidates(_hand_table())
    assert c["n_f_ge_0p1"] == 3 and c["n_f_lt_0p1"] == 2
    raw = np.sqrt((0.01**2 + 0.03**2 + 0.0**2) / 3)
    assert c["r_model_raw"] == pytest.approx(raw)
    assert c["r_model_candidate"] == pytest.approx(0.019)  # raw 0.01826 rounded UP to 2 sig figs
    assert c["tau_abs_raw"] == pytest.approx(2e-4)
    assert c["tau_abs_candidate"] == pytest.approx(2e-4)


def test_tau_floor_and_round_up():
    df = _hand_table()
    df.loc[(df.stratum == "train") & (df.d_analytic < 0.1), "d_sim"] = df.d_analytic + 1e-6
    assert P.r2_candidates(df)["tau_abs_candidate"] == pytest.approx(1e-4)  # floored
    assert P.round_up_sig(0.0123) == pytest.approx(0.013)
    assert P.round_up_sig(0.012) == pytest.approx(0.012)  # exact value is not bumped
    assert P.round_up_sig(9.51e-7) == pytest.approx(9.6e-7)


def test_packet_builder_tiny(tiny_df, tmp_path):
    csv = tmp_path / "validation.csv"
    tiny_df.to_csv(csv, index=False)
    out = P.build_packet(csv, None, P.next_attempt(tmp_path / "gates"))
    assert out.name == "attempt-01"
    assert P.next_attempt(tmp_path / "gates").name == "attempt-02"  # never overwrites
    text = (out / "README.md").read_text()
    for needle in ("by construction", "Judgment 1: physical-simulator validity", "Judgment 2: analytic-model validity",
                   "R2 CANDIDATES", "NOT an independent validation", "projected simulation time"):
        assert needle in text, needle
    for f in ("scatter_loglog.png", "hist_abs.png", "hist_rel.png", "err_vs_ratio.png", "err_vs_d.png"):
        assert (out / f).stat().st_size > 0
