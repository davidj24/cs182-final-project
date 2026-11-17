# T07: Run experiments and results report

> PILOT OVERRIDE (docs/PILOT_SPEC.md wins over this brief on scope and counts): run E1 first and write results/SUMMARY.md (sections in PILOT_SPEC §Q.8) and the meeting packet; E2, E3, E4 afterwards if resources remain and update the summary. Descriptive single-seed wording only (PREAPPROVAL R10).
> Bayes is optional; never make your deliverable depend on it. Read docs/PILOT_SPEC.md first.
> A plan (docs/plans/Txx.md, no code) is owned by this task if control asks for one.

Worktree: eval (after T06 merged and synced) | Scripts: Claude Sonnet (writer-sonnet) | Monitoring: Claude Haiku
(no session; one-shot `claude -p`) | Interpretation draft: Claude Opus | Reviewer: fresh Claude Opus
Reads: PROTOCOL §9-11 | Gates: H7 (first model), H8 (monitor), H9 (results)
Owns: scripts/run_all_eval.sh, results/ (small CSVs/figures only), docs/handoffs/T07.md,
docs/questions/T07.md, docs/runbook/T07.md

Deliverables
- scripts/run_all_eval.sh (command in docs/runbook/T07.md; human launches from a `release` checkout)
- $CS282A_DATA/results/<version>/: all metric CSVs + figures; small final ones copied to results/
- results/SUMMARY.md: each PROTOCOL §10 prediction -> observed outcome with figure path;
  surprises listed separately with a re-check plan
- Opus drafts interpretation ONLY after H9 checks pass; every number must trace to a result file
