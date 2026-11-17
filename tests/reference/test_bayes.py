"""Fast tests for the MuJoCo-mode Bayes reference (reference/bayes.py, priors.py).

Observations here are generated from the analytic closed form (the pilot simulator implements
the same ideal law, PILOT_SPEC A4). Brute-force convergence on the hard-case subset is in
test_bayes_slow.py (@slow).
"""

import math

import numpy as np
import pytest

import cs182_icl.reference.bayes as B
from cs182_icl.physics import G, analytic_displacement
from cs182_icl.reference.priors import BROAD_PRIOR, TRAIN_PRIOR, base_grid, get_prior
from cs182_icl.types import Action, Observation
from tests.reference._bayes_cases import HARD_CASES


def _obs(m, mu, acts, start_id=1):
    out = []
    for i, (F, k) in enumerate(acts):
        d = float(analytic_displacement(m, mu, F, k * 1e-3))
        out.append(Observation(Action(float(F), int(k)), start_id + i, d, d))
    return out


ACTS10 = [(14.2, 120), (25.1, 180), (18.7, 100), (21.9, 200), (16.4, 150),
          (23.3, 130), (13.6, 170), (19.8, 110), (24.4, 160), (15.5, 190)]


# ------------------------------------------------------------------------- priors / forward


def test_priors_match_frozen_ranges_and_constant_cell_weights():
    assert (TRAIN_PRIOR.m_lo, TRAIN_PRIOR.m_hi, TRAIN_PRIOR.mu_lo, TRAIN_PRIOR.mu_hi) == (0.5, 1.5, 0.3, 0.6)
    assert (BROAD_PRIOR.m_lo, BROAD_PRIOR.m_hi, BROAD_PRIOR.mu_lo, BROAD_PRIOR.mu_hi) == (0.35, 2.0, 0.2, 0.7)
    for pr in (TRAIN_PRIOR, BROAD_PRIOR):
        u, mu, logw = base_grid(pr, 400)
        assert u.size == mu.size == 400
        # constant cell weight in (log m, mu); total prior mass 1 (no extra 1/m)
        assert math.isclose(400 * 400 * math.exp(logw), 1.0, rel_tol=1e-12)
        # density w.r.t. dm dmu integrates to 1 on a fine grid in m
        m = np.linspace(pr.m_lo, pr.m_hi, 20001)
        mid = 0.5 * (m[1:] + m[:-1])
        pm = np.exp(pr.log_density(mid, 0.5 * (pr.mu_lo + pr.mu_hi))) * (pr.mu_hi - pr.mu_lo)
        assert math.isclose(float(np.sum(pm * np.diff(m))), 1.0, rel_tol=1e-8)
    with pytest.raises(ValueError):
        get_prior("nope")


def test_line_form_forward_equals_analytic_displacement():
    rng = np.random.default_rng(0)
    m = np.exp(rng.uniform(math.log(0.35), math.log(2.0), 5000))
    mu = rng.uniform(0.2, 0.7, 5000)
    F = rng.uniform(5, 26, 5000)
    T = rng.integers(100, 201, 5000) * 1e-3
    A, Bc = 1 / (2 * m * m * mu * G), 1 / (2 * m)
    a = analytic_displacement(m, mu, F, T)
    b = B._forward(A, Bc, F, T)
    assert np.any(a == 0) and np.any(a > 0)  # both clipped and moving cases covered
    assert np.all(np.abs(a - b) <= 1e-13 * np.maximum(np.abs(a), 1e-3))


# ------------------------------------------------------------------------- dedupe / copies


def test_copies_leave_posterior_and_prediction_bitwise_unchanged():
    ctx = _obs(0.8, 0.42, ACTS10[:4])
    q = Action(20.5, 140)
    base = B.bayes_predict(ctx, q)
    B._cached.cache_clear()
    with_copies = [ctx[0], ctx[1], ctx[0], ctx[2], ctx[1], ctx[3], ctx[3], ctx[0]]
    again = B.bayes_predict(with_copies, q)
    assert base.status == again.status == "ok"
    assert base.d_pred == again.d_pred  # bitwise
    assert B.make_groups(ctx) == B.make_groups(with_copies)
    # a fresh measurement of the same action (new ID) DOES change the group count n
    fresh = Observation(ctx[0].action, 999, ctx[0].d_clean, ctx[0].d_obs)
    g = {(x.F, x.k): x.n for x in B.make_groups(ctx + [fresh])}
    assert g[(ctx[0].action.F, ctx[0].action.k)] == 2
    # padding ignored; corrupt copy (same ID, different value) -> blocked, never raises
    pad = Observation(Action(13.0, 100), -1, 0.0, 0.0)
    assert B.make_groups(ctx + [pad]) == B.make_groups(ctx)
    bad = Observation(ctx[0].action, ctx[0].measurement_id, ctx[0].d_clean, ctx[0].d_obs * 1.01)
    p = B.bayes_predict(ctx + [bad], q)
    assert p.status == "blocked" and "measurement_id" in p.reason


def test_fresh_measurements_of_same_action_leave_noise_free_posterior_unchanged():
    # sigma_d = 0: var = sigma_d^2/n + tau^2 = tau^2 regardless of n (delta is shared per action)
    ctx = _obs(0.8, 0.42, ACTS10[:3])
    fresh = [Observation(o.action, 500 + i, o.d_clean, o.d_obs) for i, o in enumerate(ctx)]
    q = Action(17.0, 150)
    assert B.bayes_predict(ctx, q).d_pred == B.bayes_predict(ctx + fresh, q).d_pred


# ------------------------------------------------------------------------- c = 0


def _closed_form_prior_mean_train(F, T):
    # no clipping under the training prior: F >= 13 N > max mu m g = 8.83 N
    a, b, lo, hi = TRAIN_PRIOR.m_lo, TRAIN_PRIOR.m_hi, TRAIN_PRIOR.mu_lo, TRAIN_PRIOR.mu_hi
    L = math.log(b / a)
    E_m2 = (a**-2 - b**-2) / (2 * L)
    E_m1 = (1 / a - 1 / b) / L
    E_inv_mu = math.log(hi / lo) / (hi - lo)
    return F * F * T * T / (2 * G) * E_m2 * E_inv_mu - F * T * T / 2 * E_m1


@pytest.mark.parametrize("F,k", [(13.0, 100), (19.3, 150), (26.0, 200)])
def test_c0_equals_closed_form_training_prior_mean(F, k):
    q = Action(F, k)
    p = B.bayes_predict([], q)
    exact = _closed_form_prior_mean_train(F, q.T)
    assert p.status == "ok"
    assert abs(p.d_pred - exact) / exact < 2e-6, (p.d_pred, exact)


def test_c0_broad_prior_with_clipped_region_matches_dense_grid():
    # F = 13 N: part of the broad prior (mu m g up to 13.7 N) does not move -> clipped f = 0
    q = Action(13.0, 150)
    p = B.bayes_predict([], q, prior="broad")
    n = 4000
    pr = BROAD_PRIOR
    u = pr.u_lo + (pr.u_hi - pr.u_lo) * (np.arange(n) + 0.5) / n
    mu = pr.mu_lo + (pr.mu_hi - pr.mu_lo) * (np.arange(n) + 0.5) / n
    f = analytic_displacement(np.exp(u)[:, None], mu[None, :], q.F, q.T)
    assert np.any(f == 0)
    dense = float(f.mean())
    assert abs(p.d_pred - dense) / dense < 1e-5, (p.d_pred, dense)


# ------------------------------------------------------------------------- defined / stable


def test_sigma_d_zero_posterior_defined_on_inconsistent_and_ood_data():
    cases = [
        # T06 harness probe (not consistent with any cube in the prior)
        ([Observation(Action(15.0, 200), 1, 0.01, 0.01), Observation(Action(20.0, 200), 2, 0.05, 0.05)],
         Action(18.0, 200)),
        (_obs(2.0, 0.7, ACTS10[:6]), Action(13.0, 120)),  # far outside the training prior
        (_obs(0.3, 0.8, ACTS10[:3], 50), Action(26.0, 200)),
    ]
    for prior in ("train", "broad"):
        pr = get_prior(prior)
        for ctx, q in cases:
            p = B.bayes_predict(ctx, q, prior=prior)
            assert p.status == "ok" and np.isfinite(p.d_pred), p
            lo = analytic_displacement(pr.m_hi, pr.mu_hi, q.F, q.T)
            hi = analytic_displacement(pr.m_lo, pr.mu_lo, q.F, q.T)
            assert lo - 1e-12 <= p.d_pred <= hi + 1e-12


@pytest.mark.parametrize("y", [1e3, 1e-12, 0.0, 1e6])
def test_log_sum_exp_stable_on_extreme_likelihoods(y):
    # every grid point has log-likelihood far below -1e6 (y = 1e3 m: about -5e13 nats)
    ctx = [Observation(Action(20.0, 150), 1, y, y), Observation(Action(14.0, 110), 2, y, y)]
    q = Action(17.0, 130)
    for prior in ("train", "broad"):
        pr = get_prior(prior)
        mean, sd, info = B.posterior_mean(B.make_groups(ctx), q, prior=prior)
        assert info["resolved"] and np.isfinite(mean) and np.isfinite(sd)
        assert np.isfinite(info["log_evidence"])
        # the mass sits in the prior corner whose displacement is closest to y (largest d at
        # (m_lo, mu_lo) for huge y, smallest at (m_hi, mu_hi) for y ~ 0), within a few
        # posterior sd of it (the corner truncates a steep but finite likelihood)
        corner = (pr.m_lo, pr.mu_lo) if y > 1 else (pr.m_hi, pr.mu_hi)
        expect = analytic_displacement(*corner, q.F, q.T)
        assert abs(mean - expect) <= 3 * sd + 1e-6 * expect, (prior, y, mean, sd, expect)


# ------------------------------------------------------------------------- exact consistency


@pytest.mark.parametrize("m,mu", [(0.6, 0.35), (1.0, 0.45), (1.4, 0.55)])
def test_noise_free_observations_recover_analytic_truth(m, mu):
    q = Action(22.2, 175)
    truth = analytic_displacement(m, mu, q.F, q.T)
    for acts in (ACTS10[:2], ACTS10):  # 2 well-separated forces, and 10 pushes
        mean, sd, info = B.posterior_mean(B.make_groups(_obs(m, mu, acts)), q)
        err = abs(mean - truth)
        # posterior sd is ~1e-4 m (tau_abs); the mean sits far inside it
        assert err < 0.05 * sd and err / truth < 1e-6, (m, mu, len(acts), err, sd)
        assert not info["hit_min_width"]


# ------------------------------------------------------------------------- prefixes / cache


def test_prefixes_incremental_equal_recomputed_bitwise():
    ctx = _obs(1.1, 0.5, ACTS10)
    ctx = ctx[:5] + [ctx[1]] + ctx[5:]  # include a copy mid-way
    q = Action(16.6, 145)
    cs = [0, 1, 2, 4, 8, 10]
    B._cached.cache_clear()
    inc = B.predict_prefixes(ctx, q, cs)
    B._cached.cache_clear()
    rec = [B.bayes_predict(ctx[:c], q) for c in cs]
    assert [p.d_pred for p in inc] == [p.d_pred for p in rec]
    assert all(p.status == "ok" for p in inc)
    assert inc[0].d_pred == B.bayes_predict([], q).d_pred  # c = 0 is the prior mean
    assert B.predict_prefixes(ctx, q, [12])[0].status == "blocked"


# ------------------------------------------------------------------------- grid convergence


@pytest.mark.parametrize("name", ["light_slippery_c2", "middle_c2_sameforce", "ood_heavy_broad_c2_nearthr"])
def test_400_vs_800_base_grid_fast(name):
    hc = next(h for h in HARD_CASES if h.name == name)
    g, q = B.make_groups(hc.context()), hc.qaction()
    a, _, ia = B.posterior_mean(g, q, prior=hc.prior, n=400)
    b, _, ib = B.posterior_mean(g, q, prior=hc.prior, n=800)
    assert ia["resolved"] and ib["resolved"]
    assert abs(a - b) / abs(b) < 1e-5, (a, b)


# ------------------------------------------------------------------------- harness hook


def test_harness_probe_sees_a_valid_bayes_predictor():
    from cs182_icl.eval.predictors import bayes_status

    pred, reason = bayes_status()
    assert pred is not None and reason == "", reason
    assert B.bayes_predict_broad([], Action(15.0, 120)).status == "ok"
    assert B.bayes_predict([], Action(15.0, 120), prior="nope").status == "blocked"
