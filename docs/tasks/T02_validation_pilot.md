# T02: Validation gate and pilot
Worktree: sim (after T01 is merged and the worktree synced) | Writer: Codex fast (scripts),
Claude Sonnet (gate packet write-up) | Gate: H2 (human decides)
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
