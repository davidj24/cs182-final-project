"""PILOT_SPEC G1 simulator mechanics checks (fast)."""

import mujoco
import numpy as np
import pytest

from cs182_icl.physics import G, analytic_displacement
from cs182_icl.sim import rollout as R
from cs182_icl.sim.rollout import Simulator, action_for_dt, simulate_batch, simulate_push
from cs182_icl.types import Action, Cube

CASES = [(0.5, 0.3), (1.0, 0.45), (1.5, 0.6)]


@pytest.fixture(scope="module")
def sim():
    return Simulator()


def test_geometry(sim):
    assert np.allclose(sim.model.geom_size[sim.geom], 0.025)  # half-size, side 0.05 m
    assert R.SIDE == 0.05 and R.HALF == 0.025
    assert sim.model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert sim.model.opt.impratio == 10 and sim.model.opt.timestep == 0.001
    assert sim.model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST


@pytest.mark.parametrize("m,mu", CASES)
def test_mass_inertia(sim, m, mu):
    sim.set_cube(m, mu)
    assert sim.model.body_mass[sim.body] == m
    expect = m * 0.05**2 / 6
    assert np.allclose(sim.model.body_inertia[sim.body], expect, rtol=1e-12)  # all three axes
    assert np.allclose(sim.model.body_iquat[sim.body], [1, 0, 0, 0])
    assert np.isclose(sim.model.body_subtreemass[sim.body], m)


def test_setconst_called_and_derived_change(monkeypatch):
    s = Simulator()
    s.set_cube(1.0, 0.4)
    iw_before = s.model.body_invweight0[s.body].copy()
    calls = []
    real = mujoco.mj_setConst
    monkeypatch.setattr(mujoco, "mj_setConst", lambda *a: (calls.append(1), real(*a))[1])
    s.set_cube(2.0, 0.4)
    assert len(calls) == 1
    # translational inverse weight is 1/m: derived quantity really was refreshed
    assert np.isclose(s.model.body_invweight0[s.body][0], 0.5 * iw_before[0], rtol=1e-6)
    assert s.model.body_invweight0[s.body][0] > 0


@pytest.mark.parametrize("m,mu", CASES)
def test_friction_readout(m, mu):
    s = Simulator()
    s.set_cube(m, mu)
    floor = mujoco.mj_name2id(s.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert np.all(s.model.geom_friction[floor] == 0)  # floor does not override
    s.restore()
    d = s.data
    assert d.ncon >= 1
    for i in range(d.ncon):
        assert d.contact.dim[i] == 3
        assert d.contact[i].friction[0] == pytest.approx(mu, abs=1e-12)  # contact readout == cube mu
        assert d.contact[i].friction[1] == pytest.approx(mu, abs=1e-12)


@pytest.mark.parametrize("m,mu", CASES)
def test_settle(m, mu):
    s = Simulator()
    s.set_cube(m, mu)
    st = s.settle_stats
    assert abs(st["height"] - 0.025) < 1e-4
    assert st["excursion"] < 1e-3
    assert st["max_rot_deg"] < 1.0
    assert st["speed"] < 1e-3
    s.restore()
    assert abs(s.data.qpos[2] - 0.025) < 1e-4


def test_settle_failure_raises(monkeypatch):
    s = Simulator()
    # the guard fires when the cube cannot settle: start 5 mm above rest and allow only 2 ms
    monkeypatch.setattr(R, "SETTLE_S", 0.002)
    monkeypatch.setattr(R, "HALF", 0.03)
    with pytest.raises(RuntimeError):
        s.set_cube(1.0, 0.4)


def test_force_exactly_k_steps_then_zero(sim):
    sim.set_cube(1.0, 0.45)
    k = 150
    res, tr = sim.rollout(Action(20.0, k), record=True)
    assert int(tr["force_on"].sum()) == k
    assert tr["force_on"][:k].all() and not tr["force_on"][k:].any()
    assert np.all(sim.data.xfrc_applied == 0)
    # momentum check: with F=20 > mu m g the cube accelerates for exactly k ms, then only friction acts
    a = (20.0 - 0.45 * 1.0 * G) / 1.0
    v_end = a * k * 1e-3
    x = tr["pos"][:, 0]
    v = np.gradient(x, 1e-3)
    assert v[k - 3] == pytest.approx(v_end, rel=0.05)  # finite-difference speed, cube also rocks slightly
    assert v[k + 100] < v[k - 3] - 0.2  # friction decelerates (mu g = 4.4 m/s^2) once force is off


def test_force_steps_at_half_dt():
    s = Simulator(dt=0.0005)
    s.set_cube(1.0, 0.45)
    a = action_for_dt(Action(20.0, 150), 0.0005)
    assert a.k == 300
    _, tr = s.rollout(a, record=True)
    assert int(tr["force_on"].sum()) == 300
    assert s.dwell == 100


def test_zero_force_creep(sim):
    for m, mu in CASES:
        sim.set_cube(m, mu)
        res, tr = sim.rollout(Action(0.0, 1000), record=True)  # 1 s of zero force
        assert np.max(np.abs(tr["pos"][:, :2] - tr["pos"][0, :2])) < 1e-4
        assert abs(res.d) < 1e-4


def test_stop_conditions(sim):
    sim.set_cube(1.0, 0.45)
    res, tr = sim.rollout(Action(20.0, 150), record=True)
    assert not res.timed_out
    n = int(round(res.t_stop / 1e-3))
    d = sim.data
    # at the firing step both speeds are below tolerance; dwell of 50 steps before that
    assert np.linalg.norm(d.qvel[:3]) < 1e-4 and np.linalg.norm(d.qvel[3:6]) < 1e-3
    x = tr["pos"][:, 0]
    assert abs(x[n - 1] - x[n - 51]) < 50 * 1e-3 * 1e-4 + 1e-9
    assert tr["t"][-1] == pytest.approx(res.t_stop)
    # the rule cannot fire before the ideal stopping time + dwell; rocking ring-down delays it
    ideal = 20 * 0.15 / (1.0 * 0.45 * G) + R.DWELL_S
    assert ideal - 0.01 < res.t_stop < ideal + 0.3


def test_angular_condition_blocks_stop(monkeypatch, sim):
    # Make the angular tolerance impossible: the rule must then not fire (timeout), even though
    # the linear speed is tiny. Likewise for the linear tolerance.
    sim.set_cube(1.0, 0.45)
    monkeypatch.setattr(R, "CAP_S", 3.0)
    monkeypatch.setattr(R, "ANG_TOL", 0.0)
    s2 = Simulator()
    s2.set_cube(1.0, 0.45)
    assert s2.rollout(Action(20.0, 150))[0].timed_out
    monkeypatch.setattr(R, "ANG_TOL", 1e-3)
    monkeypatch.setattr(R, "LIN_TOL", 0.0)
    assert s2.rollout(Action(20.0, 150))[0].timed_out


def test_timeout_flag_returned():
    s = Simulator()
    s.set_cube(0.5, 0.3)
    s.cap_steps = 300  # push lasts 200 steps, cube still sliding at the cap
    res, _ = s.rollout(Action(26.0, 200))
    assert res.timed_out and res.t_stop == pytest.approx(R.CAP_S)
    assert res.d > 0  # flagged rollout is returned, not dropped


def test_max_rotation_is_recorded_over_trajectory(sim):
    sim.set_cube(1.0, 0.45)
    res, tr = sim.rollout(Action(20.0, 150), record=True)
    assert res.max_rotation_deg == pytest.approx(tr["rot_deg"].max())
    assert res.max_rotation_deg >= tr["rot_deg"][-1] - 1e-12


def _push_flatness(m, mu, F, k):
    """(max rotation deg, centre-height excursion during the push mm, min contacts during the push)."""
    s = Simulator()
    s.set_cube(m, mu)
    res, _ = s.rollout(Action(F, k))
    s.restore()
    fx = s.data.xfrc_applied[s.body]
    zs, minc = [], 99
    for _ in range(k):
        fx[0] = F
        mujoco.mj_step(s.model, s.data)
        zs.append(s.data.qpos[2])
        minc = min(minc, s.data.ncon)
    return res.max_rotation_deg, (max(zs) - min(zs)) * 1e3, minc


FLAT_CORNERS = [(1.0, 0.45, 20.0, 150), (0.5, 0.3, 26.0, 200), (1.5, 0.6, 26.0, 200),
                (0.5, 0.6, 13.0, 100), (0.5, 0.6, 26.0, 200)]  # last: light + grippy


# KNOWN FAILURE (docs/questions/T01.md): with MuJoCo soft contacts the sliding cube hops and tumbles.
# A rigid cube pushed at its centre of mass must stay flat (tipping needs mu > s/(2h) = 1), so these
# assertions are the physical requirement; they are xfail(strict) until a contact setting meets them.
@pytest.mark.xfail(strict=True, reason="contact hopping/tumbling, see docs/questions/T01.md")
@pytest.mark.parametrize("m,mu,F,k", FLAT_CORNERS)
def test_push_stays_flat_and_in_contact(m, mu, F, k):
    rot, exc_mm, minc = _push_flatness(m, mu, F, k)
    assert rot < 1.0
    assert exc_mm < 1.0
    assert minc >= 1  # contact kept in every step of the push


def test_matches_analytic_roughly():
    r = simulate_push(Cube(1.0, 0.45, 0), Action(20.0, 150))
    an = analytic_displacement(1.0, 0.45, 20.0, 0.15)
    assert r.d == pytest.approx(an, rel=0.05)


def test_dt_rerun_consistent():
    a = Action(20.0, 150)
    r1 = simulate_push(Cube(1.0, 0.45, 0), a)
    r2 = simulate_push(Cube(1.0, 0.45, 0), action_for_dt(a, 0.0005), dt=0.0005)
    assert r2.d == pytest.approx(r1.d, rel=0.02)


def test_batch_deterministic_and_complete(capsys):
    items = [(Cube(m, mu, i), Action(20.0, 120 + 10 * j)) for i, (m, mu) in enumerate(CASES) for j in range(2)]
    a = simulate_batch(items, workers=2, seed=1)
    b = simulate_batch(items, workers=1, seed=2, verbose=False)
    assert a == b and len(a) == len(items)
    assert "s/rollout" in capsys.readouterr().out
    assert [simulate_push(c, x) for c, x in items] == a
    # workers capped at 16 (no error on huge request)
    assert len(simulate_batch(items[:2], workers=1000, verbose=False)) == 2
