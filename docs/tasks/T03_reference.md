# T03: Reference estimators
Worktree: ref | Writer: Claude Opus, visible session, plan first | Reviewer: fresh Claude Opus + human hand check at H3
Reads: PROTOCOL §2, §5, §8 | Gate: H3
Owns: src/cs182_icl/reference/, tests/reference/, docs/handoffs/T03.md,
docs/questions/T03.md, docs/runbook/T03.md
Phase A (after H0): infrastructure and analytic-data checks, using physics.py only.
Phase B (after H2 sync): MuJoCo-mode Bayes with frozen sigma_d, r_model, tau_abs, forward
model, target space from DECISIONS.md. Do not guess these; if "set at H2" is still blank, stop.

Deliverables
- priors.py: training and broad priors with explicit densities; grid in (log m, mu) with
  constant cell weights for log-uniform m (no extra 1/m)
- bayes.py: MuJoCo-mode estimator exactly per PROTOCOL §8: dedupe by measurement_id; group by
  identical action; per group N(ybar; f, sigma_d^2/n + tau_model(a)^2) with tau_model = r_model*f + tau_abs
  and the theta-dependent log-variance; clipped forward model; log-sum-exp; posterior mean of the
  query outcome in the target space; incremental prefixes; cache per unique action; vectorized;
  forward model pluggable (analytic or table)
- analytic_exact.py: noise-free closed-form solve (>= 2 distinct forces) and single-force 1-D
  integral with Jacobian p(m, mu(m)) * 2 m^2 g mu(m)^2 / F^2 (extra m on log grid). These are
  for analytically generated observations ONLY: separate module, never imported by bayes.py,
  and raising an error if given simulator-sourced data.
- physics_fit.py: OLS on (F^2 T^2, -F T^2); same-force estimator h = sum d T^2 / sum T^4;
  "not determined" cases; noisy reporting rules
- oracle.py: simulator-clean and analytic-true predictions
- tests per GATES.md H3, with analytic-data checks and MuJoCo-mode checks kept separate
  (fast ones unmarked; grid convergence @slow)

Out of scope: dataset building, plotting.
