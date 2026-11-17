# T02: Validation gate and pilot

> PILOT OVERRIDE (docs/PILOT_SPEC.md wins over this brief on scope and counts): 500-2,000 stratified settings (500 minimum), ~50 at dt=0.0005, ~20 T^2 checks, throughput benchmark first; NO 10k pilot, no 4 OOD domains, no decisions on ranges/target; report R1 numbers and the G3 two-judgment summary (physical validity vs analytic validity) so control can apply PREAPPROVAL R1.
> Bayes is optional; never make your deliverable depend on it. Read docs/PILOT_SPEC.md first.
> A plan (docs/plans/Txx.md, no code) is owned by this task if control asks for one.

Worktree: sim (after T01 is merged and the worktree synced) | Writer: Claude Sonnet (writer-sonnet; scripts and gate packet write-up) | Reviewer: fresh Claude Opus | Gate: H2 (human decides)
Reads: PROTOCOL §3-4, INTERFACES
Owns: scripts/sim_validate.py, scripts/sim_pilot.py, scripts/sim_gate_packet.py, tests/sim/test_validate_*.py,
docs/handoffs/T02.md, docs/questions/T02.md, docs/runbook/T02.md

Deliverables
- scripts/sim_validate.py: 2,000 settings over training + OOD domains; CSV with analytic d,
  sim d, flags, timing; dt-halving subset; T^2-scaling subset; multiprocessing; seed
- scripts/sim_pilot.py: 10k random training pushes; histograms of d and t_stop; median d; flags
- scripts/sim_gate_packet.py: builds $CS282A_DATA/gates/H2/attempt-NN/ figures + README.md with
  every number GATES.md H2 asks for: discrepancy quantiles (relative and absolute), vs F/(mu m g)
  and vs d; no comparison against sigma_d
- docs/runbook/T02.md entries for both runs (human launches)

Do not decide ranges, sigma_d, r_model, tau_abs, forward model, or target space. Present numbers only.
