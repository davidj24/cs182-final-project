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
| Bayes forward model f | analytic (clipped) with discrepancy term IF R1 passes, else Bayes blocked; no table tonight | H2 | | gates/H2/ |
| sigma_d (noisy condition) | not used in pilot (noise 0) | pilot | 2026-11-16 | PREAPPROVAL R4 |
| r_model (relative discrepancy) | set at H2 | H2 | | |
| tau_abs (absolute discrepancy, m) | set at H2 | H2 | | |
| Target space | raw standardized, clean MuJoCo target (frozen) | pilot | 2026-11-16 | PREAPPROVAL R5 |
| Bayes grid | 400 x 400 (800 x 800 check) | H3 | | |
| Training step budget | set after benchmark for ~30-45 min (R6) | H5 | | |
| Seeds | 0 only, noise 0 only (pilot) | pilot | 2026-11-16 | PREAPPROVAL R7 |
| Accepted source commit for main runs | set before first long run | H7 | | |

tau_model(a) = r_model * f(theta, a) + tau_abs (PROTOCOL §8).
