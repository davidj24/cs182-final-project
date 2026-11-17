"""Physics-fit reference (PROTOCOL §8, PILOT_SPEC §I3). Analytic-data checks only. Fast."""

import itertools

import pytest

from cs182_icl import physics as ph
from cs182_icl.reference.physics_fit import fit_line, physics_fit_predict
from cs182_icl.types import Action, Observation

M, MU = 0.9, 0.42
_ids = itertools.count(1)


def obs(F, k, m=M, mu=MU, mid=None):
    """Observation with the analytic (ideal-model) outcome."""
    d = ph.analytic_displacement(m, mu, F, k * ph.DT)
    return Observation(Action(F, k), next(_ids) if mid is None else mid, d, d)


def truth(F, k, m=M, mu=MU):
    return ph.analytic_displacement(m, mu, F, k * ph.DT)


# ---------- not determined ----------


def test_empty_context_not_determined():
    p = physics_fit_predict([], Action(20.0, 150))
    assert p.status == "not_determined" and p.d_pred is None
    assert "empty" in p.reason


def test_one_force_level_new_query_not_determined():
    ctx = [obs(15.0, 100), obs(15.0, 150), obs(15.0, 200)]
    p = physics_fit_predict(ctx, Action(21.0, 120))
    assert p.status == "not_determined" and p.d_pred is None


def test_forces_closer_than_sep_not_determined():
    ctx = [obs(15.0, 100), obs(17.5, 150)]
    p = physics_fit_predict(ctx, Action(22.0, 120))
    assert p.status == "not_determined" and "span" in p.reason


def test_single_moving_push_not_determined():
    p = physics_fit_predict([obs(15.0, 100)], Action(22.0, 120))
    assert p.status == "not_determined"


def test_only_clipped_pushes_not_determined():
    m, mu = 1.5, 0.6  # threshold 8.83 N
    ctx = [obs(5.0, 100, m, mu), obs(8.0, 150, m, mu)]
    assert all(o.d_obs == 0.0 for o in ctx)
    p = physics_fit_predict(ctx, Action(20.0, 120))
    assert p.status == "not_determined" and "clipped" in p.reason


def test_clipped_push_gives_no_line_information():
    # one moving push + one clipped push at a far force: still only one moving force level
    m, mu = 1.5, 0.6
    ctx = [obs(5.0, 100, m, mu), obs(20.0, 150, m, mu)]
    p = physics_fit_predict(ctx, Action(25.0, 120))
    assert p.status == "not_determined"


# ---------- same-force rule ----------


def test_same_force_rule_exact_single_push():
    o = obs(18.0, 120)
    q = Action(18.0, 190)
    p = physics_fit_predict([o], q)
    assert p.status == "ok" and "same-force" in p.reason
    assert p.d_pred == ph.same_force_displacement(o.d_obs, o.action.T, q.T)
    assert p.d_pred == pytest.approx(truth(18.0, 190), rel=1e-12)


def test_same_force_rule_pooled():
    ctx = [obs(18.0, 100), obs(18.0, 160), obs(18.0, 200)]
    p = physics_fit_predict(ctx, Action(18.0, 130))
    assert p.status == "ok" and "3 push" in p.reason
    assert p.d_pred == pytest.approx(truth(18.0, 130), rel=1e-12)


def test_query_at_observed_force_uses_same_force_rule_even_if_line_determined():
    ctx = [obs(14.0, 100), obs(24.0, 150), obs(19.0, 170)]
    q = Action(19.0, 110)
    p = physics_fit_predict(ctx, q)
    assert "same-force" in p.reason
    assert p.d_pred == ph.same_force_displacement(ctx[2].d_obs, ctx[2].action.T, q.T)


def test_same_force_clipped_predicts_zero():
    m, mu = 1.5, 0.6
    p = physics_fit_predict([obs(8.0, 100, m, mu)], Action(8.0, 180))
    assert p.status == "ok" and p.d_pred == 0.0


# ---------- line fit ----------


@pytest.mark.parametrize("space", ["d", "line"])
def test_two_force_line_recovery(space):
    ctx = [obs(14.0, 130), obs(23.0, 170)]
    A, B = fit_line(ctx, fit_space=space)
    A0, B0 = ph.line_coeffs(M, MU)
    assert A == pytest.approx(A0, rel=1e-10)
    assert B == pytest.approx(B0, rel=1e-10)
    m, mu = ph.params_from_line(A, B)
    assert m == pytest.approx(M, rel=1e-10)
    assert mu == pytest.approx(MU, rel=1e-10)
    p = physics_fit_predict(ctx, Action(19.5, 140), fit_space=space)
    assert p.status == "ok" and "line fit" in p.reason
    assert p.d_pred == pytest.approx(truth(19.5, 140), rel=1e-10)


def test_line_fit_many_pushes_and_clipped_ignored():
    m, mu = 1.4, 0.55  # threshold 7.55 N
    ctx = [obs(5.0, 200, m, mu)] + [obs(F, k, m, mu) for F, k in [(13, 100), (17, 180), (25, 140)]]
    p = physics_fit_predict(ctx, Action(21.0, 160))
    assert p.status == "ok"
    assert p.d_pred == pytest.approx(truth(21.0, 160, m, mu), rel=1e-10)


def test_line_fit_clipped_query_predicts_zero():
    m, mu = 1.5, 0.6
    ctx = [obs(14.0, 130, m, mu), obs(23.0, 170, m, mu)]
    p = physics_fit_predict(ctx, Action(7.0, 150))
    assert p.status == "ok" and p.d_pred == 0.0


# ---------- copies / measurement IDs ----------


@pytest.mark.parametrize("query", [Action(19.5, 140), Action(14.0, 160)])
def test_copies_leave_prediction_bitwise_unchanged(query):
    base = [obs(14.0, 130), obs(23.0, 170), obs(18.0, 110)]
    p0 = physics_fit_predict(base, query)
    for n_copies in (1, 2, 4):
        ctx = base + [base[i % len(base)] for i in range(n_copies)]
        p = physics_fit_predict(ctx, query)
        assert p.d_pred == p0.d_pred and p.status == p0.status


def test_conflicting_duplicate_id_raises():
    a = obs(14.0, 130, mid=99)
    b = Observation(a.action, 99, a.d_clean, a.d_obs * 1.01)
    with pytest.raises(ValueError):
        physics_fit_predict([a, obs(23.0, 170), b], Action(19.0, 140))


def test_fresh_measurement_counts_but_is_consistent():
    # fresh measurement of the same action: new ID, same clean outcome; noise-free -> same answer
    a = obs(14.0, 130)
    fresh = Observation(a.action, 12345, a.d_clean, a.d_obs)
    ctx = [a, obs(23.0, 170)]
    q = Action(19.5, 140)
    assert physics_fit_predict(ctx + [fresh], q).d_pred == pytest.approx(
        physics_fit_predict(ctx, q).d_pred, rel=1e-12
    )
