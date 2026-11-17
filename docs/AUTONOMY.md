# Autonomous execution policy (control session), pilot edition

Applies only when docs/PREAPPROVAL.md says `Status: APPROVED` and only to docs/PILOT_SPEC.md.
Where it conflicts with AGENTS.md, CLAUDE.md, GATES.md or workflow/STEPS.md on "who decides / who
approves / who merges", THIS FILE WINS for this authorized run. Scope and budgets: PILOT_SPEC wins.
Paths: `$WORKFLOW` = /home/exx/soomi/soomi_projects/cs282a_project/workflow (a sibling of the repo,
NOT inside it). Integration checkout: $WT/integration (branch soomi-integration), used only by control.

## Roles and models
- CONTROL (Opus): launches workers, reviews, gates, merges into soomi-integration, writes DECISIONS.md
  from PREAPPROVAL §B, changes the shared env if needed (never plain `uv sync`), launches long jobs
  via $WORKFLOW/run_job.sh, writes the ledger via $WORKFLOW/ledger.sh.
- Opus: control, T00, mathematical references (T03), all reviews (reviewer-opus), plan reviews.
- Sonnet (writer-sonnet): simulator T01/T02, datasets T04, model/training T05, evaluation T06/T07 scripts.
- Haiku (monitor-haiku): short log summaries only (one-shot, read-only).
- Workers obey AGENTS.md: no merges, no pushes, no env changes, no jobs over 5 minutes.
- CLAUDE.md "start in plan mode and wait for approval" is replaced, for this run, by plan review below.

## Per-task loop
1. Plan tasks (T00, T03, and any task touching more than 3 files): the writer writes docs/plans/Txx.md
   (no code) in its worktree; a FRESH reviewer-opus checks it against PILOT_SPEC/PROTOCOL; fix once
   (counts as a repair cycle only if the reviewer found a CRITICAL/MAJOR); then implement.
   This replaces routine human plan approval.
2. Writer implements, commits, writes handoff, reports SHA, stops.
3. reviewer-opus reviews that exact SHA; control saves the report to $CS282A_DATA/reviews/Txx/<sha>/review.md.
4. Any unresolved CRITICAL or MAJOR finding: control sends the findings back to a writer for fixes
   (permitted; this overrides the AGENTS.md "only the human sends back findings" wording), records
   `REPAIR Txx` in the ledger, and re-reviews the new SHA. Maximum 2 repair cycles per task, then STOP.
5. Control runs the gate's automated checks (GATES.md, pilot mapping), then merges the SHA
   (`--no-ff`) into soomi-integration in $WT/integration, copies the handoff's AI-usage row into
   docs/ai_usage.md, links docs/runbook/Txx.md in docs/RUNBOOK.md, commits, and writes
   `INTEGRATED Txx <sha>` to the ledger. Condition: ZERO unresolved CRITICAL or MAJOR findings and
   `pytest -m "not slow"` passing at that SHA.
6. After each integration, control syncs idle worker worktrees (`git merge soomi-integration`, never
   overwriting uncommitted work; skip a worktree that is dirty or has a worker running).
Fresh worker session per task. Each launch is ledgered (`LAUNCH <worker> <Txx>`).

## Parallelism
After T00 is integrated, independent tasks run in parallel in separate worktrees (sim, ref, data,
train, eval). Evaluation fixtures (T06 metric fixture, Bayes-as-model harness) start before any model
exists. The transformer is trained ONCE; the checkpoint is reused for E1-E4.

## Long jobs
Control launches runbook commands with `$WORKFLOW/run_job.sh <job_id> <reserve_gpuh> <timeout_s> -- cmd`
from a clean `setup_worktrees.sh release` checkout. run_job.sh checks the budget, reserves GPU-hours
(JOBSTART), enforces the timeout, and writes JOBEND with measured device-hours. CPU simulation jobs
use reserve 0 and `--workers <= 16`. One GPU job at a time. Monitoring: control uses the Monitor tool
(until-loop on the log/exit marker) when available, else a bounded sleep loop in the background job;
never poll in a tight loop, never delegate watching to workers. monitor-haiku summarizes logs on demand.

## Ledger (append-only $CS282A_DATA/state/ledger.md; written ONLY through $WORKFLOW/ledger.sh)
Machine-read record lines (literal first token): `LAUNCH <worker> <task>`, `REPAIR <task>`,
`GPUH <hours> <job>`, `JOBSTART <job> <epoch> <reserved_gpuh>`, `JOBEND <job> <gpuh>`,
`INTEGRATED <task> <sha>`, `DECISION <text>`, `STOP <reason>`, `NOTE <text>`.
Resumption: a fresh control reads PREAPPROVAL.md, PILOT_SPEC.md, DECISIONS.md and the ledger.
Dollar spend is not measurable and is not claimed to be enforced.

## Data preservation
Never delete, move, truncate or overwrite anything under $CS282A_DATA or hw9_hardware_check/.
Write only to new versioned paths (datasets/vN, runs/<run_id>, gates/Hn/attempt-NN,
reviews/Txx/<sha>, results/vN); if a path exists, create the next version. Use noclobber.

## Stop (write STOP to the ledger, tell the user, wait)
- Unresolved scientific ambiguity not decided by PREAPPROVAL §B or PILOT_SPEC.
- Any cap in PREAPPROVAL §A would be exceeded (report progress and ask first).
- Physical simulator invalid and not fixable without a protocol change (dt, geometry, ranges).
- More than 2 repair cycles on a task; NaN losses without a code cause; test-set leakage found.
- Any action in PREAPPROVAL §D would be needed.
Never "fix" an experiment so a number comes out as expected.

## Gate mapping for the pilot (GATES.md numbers; Bayes is optional everywhere)
H0 interfaces (minimal T00) -> required. H1 simulator mechanics (PILOT_SPEC §G1) -> mandatory; failure
blocks data and training. H2 -> pilot validation: G2 outputs, R1 decides Bayes validity ONLY; physical
validity (G3) decides whether data generation proceeds; ranges and target space are frozen, not decided.
H3 references -> oracle, analytic, physics fit required; Bayes only if R1 passed, else record
"Bayes reference: blocked, reason: ..." and continue; Bayes absence never blocks H4-H7.
H4 datasets (pilot sizes) -> mandatory. H5 training smoke -> shape, causal, normalization, leakage,
overfit mandatory. H6 evaluation harness -> hand-calculated fixture and clean-target/pairing checks
mandatory; Bayes-as-model check only if Bayes exists. H7 first model -> E1. H8/H9 -> monitoring and the
summary. No 5 further runs. If Bayes is unavailable, NMSE and gap-to-Bayes are omitted (never
substituted with another denominator under the same name).
