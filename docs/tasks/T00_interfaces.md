# T00: Interfaces and skeleton
Worktree: spec | Writer: Claude Opus, visible session (plan mode) | Reviewer: fresh Claude Opus (reviewer-opus)
Reads: PROTOCOL §2-9 | Gate: H0
Owns: docs/INTERFACES.md, pyproject.toml (dependency additions only), src/cs182_icl/**
(skeleton + physics.py + types.py), tests/ (shared skeleton, tests/test_physics.py),
scripts/ (empty skeleton), configs/ (empty skeleton), .gitignore additions,
docs/handoffs/T00.md, docs/questions/T00.md, docs/runbook/T00.md

Deliverables
- docs/INTERFACES.md with:
  - dataclasses: Cube(m, mu, cube_id), Action(F, k), Observation(action, measurement_id,
    d_clean, d_obs), TestItem(cube, query, d_query_clean, context: list[Observation], tags)
  - signatures: analytic_displacement(m, mu, F, T), simulate_push(cube, action) ->
    PushResult(d, t_stop, max_rotation_deg, timed_out), bayes_predict(...) in MuJoCo mode
    (discrepancy-aware, PROTOCOL §8), analytic_exact_* (analytic-data only), physics_fit_predict(...),
    model_predict(...)
  - storage: .npz arrays + meta.json per split under $CS282A_DATA/datasets/<version>/;
    field names, dtypes, shapes
  - ownership table: for each of T01..T07, source dir, tests dir, scripts, task files
- Skeleton in the existing package: src/cs182_icl/{types.py, physics.py, sim/, reference/,
  data/, model/, train/, eval/}, tests/<module>/, scripts/, configs/
- src/cs182_icl/physics.py: closed form with clipping, t_stop, same-force rule, line-form
  helpers + tests (values in GATES.md H0)
- pytest marker "slow" (pyproject); data/outputs already ignored by .gitignore
- Add dependencies mujoco and pytest to pyproject.toml. Do NOT run uv lock/sync; the human does.
- docs/PROTOCOL.md already exists; do not edit it.

Out of scope: simulator, Bayes, model code.
