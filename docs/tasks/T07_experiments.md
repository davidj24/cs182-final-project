# T07: Run experiments and results report
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
