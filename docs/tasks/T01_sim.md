# T01: MuJoCo scene and rollout

> PILOT OVERRIDE (docs/PILOT_SPEC.md wins over this brief on scope and counts): simulator mechanics checks in PILOT_SPEC §G1 are mandatory; save frames/replay for light-slippery, middle, heavy-grippy; --workers <= 16.
> Bayes is optional; never make your deliverable depend on it. Read docs/PILOT_SPEC.md first.
> A plan (docs/plans/Txx.md, no code) is owned by this task if control asks for one.

Worktree: sim | Writer: Claude Sonnet (writer-sonnet) | Reviewer: fresh Claude Opus (reviewer-opus)
Reads: PROTOCOL §3-4, INTERFACES | Gate: H1
Owns: src/cs182_icl/sim/, assets/, tests/sim/, scripts/sim_view.py,
docs/handoffs/T01.md, docs/questions/T01.md, docs/runbook/T01.md

Deliverables
- assets/cube.xml per PROTOCOL §4 (half-size 0.025, floor friction 0, condim 3, elliptic
  cone, impratio 10, dt 0.001, implicitfast)
- Simulator class: set_cube(m, mu) sets mass, inertia m*s^2/6 (s = 0.05), friction;
  mj_setConst; reset; settle 0.5 s; cache settled state
- simulate_push(action): restore state, xfrc_applied for exactly k steps, stopping rule
  (lin < 1e-4 m/s, ang < 1e-3 rad/s, 50 steps; cap 15 s), record d, t_stop, max rotation
  over the trajectory (all axes), timed_out
- dt parameter so the gate can rerun at 0.0005 (2k force steps, 100-step dwell)
- fast tests: exact-k force steps, friction readout equals mu, zero-force creep, settled height
- scripts/sim_view.py: viewer demo of 3 pushes for gate H1
- print seconds per rollout

Out of scope: validation statistics, datasets.
