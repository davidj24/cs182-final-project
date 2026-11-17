# Frozen decisions (human edits only)

Under autonomous execution, control fills this table by the rules in PREAPPROVAL §B and records
the rule id in the Evidence column.

Each entry: value | gate | date | evidence path under $CS282A_DATA.
"Set at H2" means agents must not estimate or guess it; wait for the value.

| Item | Value | Gate | Date | Evidence |
|---|---|---|---|---|
| Training m range | log-uniform [0.5, 1.5] kg (frozen for pilot, PILOT_SPEC §C) | pilot | 2026-11-16 | PILOT_SPEC.md | |
| Training mu range | uniform [0.3, 0.6] (frozen for pilot, PILOT_SPEC §C) | pilot | 2026-11-16 | PILOT_SPEC.md | |
| Force range | uniform [13, 26] N (frozen for pilot, PILOT_SPEC §C) | pilot | 2026-11-16 | PILOT_SPEC.md | |
| Duration | k in {100..200}, dt = 0.001 s (frozen for pilot, PILOT_SPEC §C) | pilot | 2026-11-16 | PILOT_SPEC.md | |
| Bayes forward model f | analytic (clipped) with discrepancy term; R1 numerically PASSED (median rel 3.3e-7, p95 2.4e-6; dt-halving median 1.8e-7, p95 1.5e-6) but BY CONSTRUCTION (simulator implements the ideal law, A4): Bayes valid for the declared ideal dynamics only | H2 | 2026-11-16 | gates/H2/attempt-01; reviews/T02/3c74ca1 |
| sigma_d (noisy condition) | not used in pilot (noise 0) | pilot | 2026-11-16 | PREAPPROVAL R4 |
| r_model (relative discrepancy) | 9.5e-7 (R2: RMS(delta/f), f>=0.1 m, rounded up) | H2 | 2026-11-16 | gates/H2/attempt-01; reviews/T02/3c74ca1 |
| tau_abs (absolute discrepancy, m) | 1e-4 (R2 floor; raw RMS 5.1e-7 m) | H2 | 2026-11-16 | gates/H2/attempt-01; reviews/T02/3c74ca1 |
| Target space | raw standardized, clean MuJoCo target (frozen) | pilot | 2026-11-16 | PREAPPROVAL R5 |
| Bayes grid | 400 x 400 (800 x 800 check) | H3 | | |
| Training step budget | 150000 steps (~35 min at 71 steps/s measured on GPU 1, real data) | H5 | 2026-11-16 | runs/pilot_v1_smoke2k, logs/pilot_smoke2k.log |
| Seeds | 0 only, noise 0 only (pilot) | pilot | 2026-11-16 | PREAPPROVAL R7 |
| Accepted source commit for main runs | set before first long run | H7 | | |

tau_model(a) = r_model * f(theta, a) + tau_abs (PROTOCOL §8).
| Physics-fit fit space | regress d on (F^2T^2, -FT^2) (PROTOCOL 8); line-form only as sensitivity | pilot | 2026-11-16 | T03 review M3 |
| Physics-fit distinct-force tolerance | >= 3 N force span among >= 2 moving pushes; moving = d_obs > 1e-6 m | pilot | 2026-11-16 | T03 review M2 |
| Simulator scene (PROTOCOL 4 deviation) | idealized 1D translational dynamics in MuJoCo (slide joint, frictionloss = mu*m*g, or explicit Coulomb force); contact scene rejected; user decision 2026-11-16 (Option 2, third T01 cycle); fallback = direct analytic data if cycle fails | pilot | 2026-11-16 | PILOT_SPEC A4; reviews/T01/9876867; sim docs/questions/T01.md |
