# Agent rules: push-dynamics ICL project (CS 182/282A)

Train a small GPT-2 transformer to predict cube push displacement in context
(MuJoCo data) and compare it with physics-aware reference estimators.
Package: `src/cs182_icl/`. The team-wide rules in CLAUDE.md also apply
(never invent numbers or citations; fixed seeds; every plot reproducible from a script).

## Sources of truth (read only what your task brief points to)
- docs/PILOT_SPEC.md  READ FIRST, every task. The meeting-pilot scope, counts, budgets and
                      deliverables. Where it conflicts with your task brief or PROTOCOL on scope,
                      PILOT_SPEC wins; correctness requirements are never reduced. Bayes is optional.
- docs/PROTOCOL.md   experiment plan; cite as "PROTOCOL §N"
- docs/INTERFACES.md types, signatures, file formats (written by T00)
- docs/DECISIONS.md  frozen values (ranges, sigma_d, r_model, tau_abs, target space, budgets)
- docs/tasks/Txx_*.md your task brief: scope, ownership, deliverables, acceptance tests

Never edit PROTOCOL.md or DECISIONS.md. If something there looks wrong or
ambiguous, stop and write it in your questions file.

## Scope and git
- Work only inside your worktree, on its exp/* branch. Never commit to
  soomi-integration, soomi or main; never merge, push, or rewrite history. (These limits bind workers.
  Under docs/AUTONOMY.md only the control session merges reviewed SHAs into soomi-integration.)
- Edit only what your brief's "Owns" line lists (source dir, tests dir, scripts,
  and your task files below, plus docs/plans/Txx.md if a plan is requested). Need a change elsewhere? Put it in your questions file.
- Task-specific report files, never shared ones:
  - questions: docs/questions/Txx.md
  - handoff:   docs/handoffs/Txx.md
  - commands for control/the human to launch: docs/runbook/Txx.md
  - plan (only when asked, no code): docs/plans/Txx.md
  Do not edit docs/RUNBOOK.md or docs/ai_usage.md (the merger does, at merge).
- Small commits, message prefix = task id, e.g. "T03: single-force Jacobian".

## Environment
- One shared uv environment (UV_PROJECT_ENVIRONMENT). Only the control session or the human changes the environment, and never with a plain `uv sync`
  against the shared env (see workflow/STEPS.md appendix).
  You run `python ...`, `pytest ...`, or `uv run --no-sync ...`. Never `uv sync`, `uv add`,
  `uv lock`, `pip install`, or `pip install -e .`. Code runs via PYTHONPATH=$PWD/src.
- Need a new dependency? State it in your questions file; T00 owns pyproject.toml.
- All data, runs, figures go under $CS282A_DATA (outside git). Never commit data or checkpoints.
- Physics and Bayes code use float64. Every random draw takes an explicit seed.

## Runtime limits
- `pytest -m "not slow"` must finish in under 60 s. Mark anything longer @slow.
- Do not start any job expected to take more than 5 minutes. Write the script, then put the
  exact command in docs/runbook/Txx.md with: source commit, working directory, config hash,
  dataset version, seed, expected runtime and CPU/GPU, outputs, how to check it worked.
- Never poll or watch a running job.
- Simulation scripts take `--workers` (default and maximum 16). Never exceed the pilot caps in PILOT_SPEC §D.

## Data preservation
- Never delete, move, truncate or overwrite anything under $CS282A_DATA or hw9_hardware_check/.
  Write only to new versioned paths (datasets/vN, runs/<run_id>, gates/Hn/attempt-NN, results/vN);
  if a path exists, use the next version.

## Honesty rules
- Never delete, skip, or loosen a test to make it pass. Never change a tolerance or constant
  from PROTOCOL/DECISIONS.
- Do not claim something works without showing the command and its output.
- If a result surprises you, report it; do not "fix" the experiment to match expectations.

## When you finish
Commit everything, then write docs/handoffs/Txx.md (under 40 lines) with sections:
1. what changed (files), 2. how to verify (exact commands), 3. deviations from
PROTOCOL/INTERFACES, 4. open questions, 5. known weaknesses,
6. **AI usage**: one row for docs/ai_usage.md (date, tool and model, project part,
what it was used for, how it was verified). The human copies it at merge.
Commit the handoff, state the final commit SHA, then stop. Do not keep working after review
starts; fixes happen only when the controller (or the human) sends back specific findings, as a
new worker session on the same branch.
