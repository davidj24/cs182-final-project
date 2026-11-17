# Frozen decisions (human edits only)

Each entry: value | gate | date | evidence path under $CS282A_DATA.
"Set at H2" means agents must not estimate or guess it; wait for the value.

| Item | Value | Gate | Date | Evidence |
|---|---|---|---|---|
| Training m range | log-uniform [0.5, 1.5] kg (provisional) | H2 | | |
| Training mu range | uniform [0.3, 0.6] (provisional) | H2 | | |
| Force range | uniform [13, 26] N (provisional) | H2 | | |
| Duration | k in {100..200}, dt = 0.001 s (provisional) | H2 | | |
| Bayes forward model f | analytic (clipped) or simulated table: set at H2 | H2 | | gates/H2/ |
| sigma_d (noisy condition) | 0.05 x median training d = set at H2 | H2 | | |
| r_model (relative discrepancy) | set at H2 | H2 | | |
| tau_abs (absolute discrepancy, m) | set at H2 | H2 | | |
| Target space | raw standardized / log: set at H2 | H2 | | |
| Bayes grid | 400 x 400 (800 x 800 check) | H3 | | |
| Training step budget | set at H5 (ceiling 100k) | H5 | | |
| Seeds | 0, 1, 2 | H0 | | |
| Accepted source commit for main runs | set before first long run | H7 | | |

tau_model(a) = r_model * f(theta, a) + tau_abs (PROTOCOL §8).
