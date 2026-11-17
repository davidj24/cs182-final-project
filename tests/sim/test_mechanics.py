"""Simulator checks for the idealized 1D translational pusher (PILOT_SPEC A4, fast).

These test that the code implements the declared ideal law (and the simulator API contract);
they are not independent validation of contact physics.
"""

import mujoco
import numpy as np
import pytest

from cs182_icl.physics import G, analytic_displacement
from cs182_icl.sim import rollout as R
from cs182_icl.sim.rollout import Simulator, action_for_dt, simulate_batch, simulate_push
from cs182_icl.types import Action, Cube

CASES = [(0.5, 0.3), (1.0, 0.45), (1.5, 0.6)]
FOUR = [(0.5, 0.3, 26.0, 200), (1.0, 0.45, 20.0, 150), (1.5, 0.6, 26.0, 200), (0.5, 0.6, 26.0, 200)]


@pytest.fixture(scope="module")
def sim():
    return Simulator()


def test_geometry_and_model(sim):
    mdl = sim.model
    assert np.allclose(mdl.geom_size[sim.geom], 0.025)  # half-size, side 0.05 m
    assert R.SIDE == 0.05 and R.HALF == 0.025
    assert mdl.opt.timestep == 0.001
    assert mdl.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert mdl.nv == 1 and mdl.nq == 1  # only the x slide dof
    assert mdl.jnt_type[0] == mujoco.mjtJoint.mjJNT_SLIDE and np.allclose(mdl.jnt_axis[0], [1, 0, 0])
    assert mdl.dof_damping[0] == 0 and mdl.dof_armature[0] == 0


@pytest.mark.parametrize("m,mu", CASES)
def test_mass_inertia_and_friction_force(sim, m, mu):
    sim.set_cube(m, mu)
    assert sim.model.body_mass[sim.body] == m
    assert np.allclose(sim.model.body_inertia[sim.body], m * 0.05**2 / 6, rtol=1e-12)
    assert np.isclose(sim.model.body_subtreemass[sim.body], m)
    assert sim.coulomb == pytest.approx(mu * m * G, rel=1e-15)  # explicit Coulomb force readback
    assert sim.model.dof_frictionloss[0] == 0  # frictionloss not used (soft-constraint creep)


def test_setconst_called_and_derived_change(monkeypatch):
    s = Simulator()
    s.set_cube(1.0, 0.4)
    iw_before = s.model.body_invweight0[s.body].copy()
    calls = []
    real = mujoco.mj_setConst
    monkeypatch.setattr(mujoco, "mj_setConst", lambda *a: (calls.append(1), real(*a))[1])
    s.set_cube(2.0, 0.4)
    assert len(calls) == 1
    assert np.isclose(s.model.body_invweight0[s.body][0], 0.5 * iw_before[0], rtol=1e-6)


@pytest.mark.parametrize("m,mu", CASES)
def test_initial_state_exactly_at_rest(m, mu):
    s = Simulator()
    s.set_cube(m, mu)
    assert s.settle_stats == dict(height=0.025, excursion=0.0, max_rot_deg=0.0, speed=0.0)
    s.restore()
    assert s.data.qpos[0] == 0.0 and s.data.qvel[0] == 0.0


def test_no_contact_influence(sim):
    sim.set_cube(1.0, 0.45)
    assert np.all(sim.model.geom_contype == 0) and np.all(sim.model.geom_conaffinity == 0)
    sim.restore()
    for _ in range(300):
        sim.data.qfrc_applied[0] = 20.0
        mujoco.mj_step(sim.model, sim.data)
        assert sim.data.ncon == 0
    sim.data.qfrc_applied[0] = 0.0
    mujoco.mj_kinematics(sim.model, sim.data)  # mj_step leaves xpos one step stale
    # visual box follows the x joint; height stays at the half-size
    assert sim.data.xpos[sim.body][0] == pytest.approx(sim.data.qpos[0])
    assert sim.data.xpos[sim.body][2] == pytest.approx(0.025)


@pytest.mark.parametrize("m,mu,F,k", FOUR)
def test_push_acceleration_and_coast_deceleration(sim, m, mu, F, k):
    sim.set_cube(m, mu)
    _, tr = sim.rollout(Action(F, k), record=True)
    v = tr["vel"]
    a_push = (F - mu * m * G) / m
    assert v[k - 1] / (k * 1e-3) == pytest.approx(a_push, rel=1e-6)
    assert v[k - 1] == pytest.approx(a_push * k * 1e-3, rel=1e-6)  # PUSH accel to 1e-6 (< 1%)
    n = 10
    assert (v[k + n - 1] - v[k - 1]) / (n * 1e-3) == pytest.approx(-mu * G, rel=1e-6)


def test_force_exactly_k_steps_then_zero(sim):
    sim.set_cube(1.0, 0.45)
    k = 150
    res, tr = sim.rollout(Action(20.0, k), record=True)
    assert int(tr["force_on"].sum()) == k
    assert tr["force_on"][:k].all() and not tr["force_on"][k:].any()
    assert np.all(sim.data.qfrc_applied == 0) and np.all(sim.data.xfrc_applied == 0)
    # momentum check: net force (F - mu m g) acts for exactly k ms, then only friction acts
    v = tr["vel"]
    assert v[k - 1] == pytest.approx((20.0 - 0.45 * G) * k * 1e-3, rel=1e-9)
    assert v[k + 5] < v[k - 1]  # friction decelerates immediately after release
    # applied force really is zero from step k: count steps where it was applied via a spy
    s = Simulator()
    s.set_cube(1.0, 0.45)
    seen = []
    orig = s._friction_step
    s._friction_step = lambda f: (seen.append(f), orig(f))[1]
    s.rollout(Action(20.0, k))
    assert seen[:k] == [20.0] * k and all(f == 0.0 for f in seen[k:]) and len(seen) > k


def test_force_steps_at_half_dt():
    s = Simulator(dt=0.0005)
    s.set_cube(1.0, 0.45)
    a = action_for_dt(Action(20.0, 150), 0.0005)
    assert a.k == 300
    _, tr = s.rollout(a, record=True)
    assert int(tr["force_on"].sum()) == 300
    assert s.dwell == 100


@pytest.mark.parametrize("m,mu", CASES)
def test_below_threshold_does_not_move(m, mu):
    thr = mu * m * G
    for frac in (0.0, 0.5, 0.9, 0.999, 1.0):
        res = simulate_push(Cube(m, mu, 0), Action(frac * thr, 150))
        assert res.d == 0.0 and not res.timed_out


def test_just_above_threshold_moves_like_closed_form():
    m, mu = 1.0, 0.45
    F = 1.01 * mu * m * G
    d = simulate_push(Cube(m, mu, 0), Action(F, 150)).d
    assert d > 0
    assert d == pytest.approx(analytic_displacement(m, mu, F, 0.15), rel=0.02)


def test_zero_force_creep(sim):
    for m, mu in CASES:
        sim.set_cube(m, mu)
        res, tr = sim.rollout(Action(0.0, 1000), record=True)  # 1 s of zero force
        assert np.max(np.abs(tr["pos"][:, 0])) < 1e-4  # < 0.1 mm (actually exactly 0)
        assert abs(res.d) < 1e-4


def test_stays_at_rest_after_stopping(sim):
    sim.set_cube(1.0, 0.45)
    sim.rollout(Action(20.0, 150))
    x = sim.data.qpos[0]
    for _ in range(2000):  # no force, no chatter, no drift
        sim._friction_step(0.0)
        mujoco.mj_step(sim.model, sim.data)
    assert sim.data.qpos[0] == x and sim.data.qvel[0] == 0.0


def test_stop_conditions(sim):
    sim.set_cube(1.0, 0.45)
    res, tr = sim.rollout(Action(20.0, 150), record=True)
    assert not res.timed_out
    n = int(round(res.t_stop / 1e-3))
    assert abs(sim.data.qvel[0]) < 1e-4
    x = tr["pos"][:, 0]
    assert abs(x[n - 1] - x[n - 51]) < 50 * 1e-3 * 1e-4 + 1e-9
    assert tr["t"][-1] == pytest.approx(res.t_stop)
    ideal = 20 * 0.15 / (1.0 * 0.45 * G) + R.DWELL_S  # stop time + dwell
    assert ideal - 0.01 < res.t_stop < ideal + 0.01


def test_linear_condition_blocks_stop(monkeypatch):
    monkeypatch.setattr(R, "CAP_S", 3.0)
    monkeypatch.setattr(R, "LIN_TOL", 0.0)  # impossible tolerance: rule must not fire
    s = Simulator()
    s.set_cube(1.0, 0.45)
    assert s.rollout(Action(20.0, 150))[0].timed_out


def test_timeout_flag_returned():
    s = Simulator()
    s.set_cube(0.5, 0.3)
    s.cap_steps = 300  # push lasts 200 steps, cube still sliding at the cap
    res, _ = s.rollout(Action(26.0, 200))
    assert res.timed_out and res.t_stop == pytest.approx(R.CAP_S)
    assert res.d > 0  # flagged rollout is returned, not dropped


def test_flagged_rollouts_in_batch_not_dropped(monkeypatch):
    monkeypatch.setattr(R, "CAP_S", 0.3)
    items = [(Cube(0.5, 0.3, 0), Action(26.0, 200)), (Cube(1.0, 0.45, 1), Action(0.0, 100))]
    out = simulate_batch(items, workers=1, dt=0.00123, verbose=False)  # fresh dt: new Simulator
    assert len(out) == 2 and out[0].timed_out and not out[1].timed_out


def test_max_rotation_is_vacuous_zero(sim):
    for m, mu, F, k in FOUR:
        sim.set_cube(m, mu)
        res, tr = sim.rollout(Action(F, k), record=True)
        assert res.max_rotation_deg == 0.0 and np.all(tr["rot_deg"] == 0.0)


@pytest.mark.parametrize("m,mu,F,k", FOUR)
def test_matches_analytic_displacement(m, mu, F, k):
    r = simulate_push(Cube(m, mu, 0), Action(F, k))
    an = analytic_displacement(m, mu, F, k * 1e-3)
    assert r.d == pytest.approx(an, rel=1e-3)  # measured: < 1.4e-5 over 350 random pushes


def test_t_squared_scaling():
    for m, mu in CASES:
        d1 = simulate_push(Cube(m, mu, 0), Action(20.0, 100)).d
        d2 = simulate_push(Cube(m, mu, 0), Action(20.0, 200)).d
        assert d2 / d1 == pytest.approx(4.0, rel=1e-4)


def test_dt_rerun_consistent():
    a = Action(20.0, 150)
    r1 = simulate_push(Cube(1.0, 0.45, 0), a)
    r2 = simulate_push(Cube(1.0, 0.45, 0), action_for_dt(a, 0.0005), dt=0.0005)
    assert r2.d == pytest.approx(r1.d, rel=1e-3)


def test_batch_deterministic_and_complete(capsys):
    items = [(Cube(m, mu, i), Action(20.0, 120 + 10 * j)) for i, (m, mu) in enumerate(CASES) for j in range(2)]
    a = simulate_batch(items, workers=2, seed=1)
    b = simulate_batch(items, workers=1, seed=2, verbose=False)
    assert a == b and len(a) == len(items)
    assert "s/rollout" in capsys.readouterr().out
    assert [simulate_push(c, x) for c, x in items] == a  # input order preserved
    assert len(simulate_batch(items[:2], workers=1000, verbose=False)) == 2  # worker cap, no error


def test_worker_cap(monkeypatch):
    import multiprocessing as mp
    seen = []

    class FakePool:
        def __init__(self, n): seen.append(n)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def map(self, f, chunks, chunksize=1): return [f(c) for c in chunks]

    class Ctx:
        Pool = FakePool

    monkeypatch.setattr(mp, "cpu_count", lambda: 64)
    monkeypatch.setattr(mp, "get_context", lambda *a: Ctx)
    items = [(Cube(0.5 + 0.1 * i, 0.4, i), Action(15.0, 100)) for i in range(4)]
    simulate_batch(items, workers=1000, verbose=False)
    assert seen == [16]
