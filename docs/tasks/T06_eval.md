# T06: Evaluation harness, metrics, plots
Worktree: eval | Writer: Claude Sonnet (harness), Claude Haiku (plot styling) | Reviewer: Codex fast
Reads: PROTOCOL §9-11 | Gate: H6 (harness correctness, before any model exists)
Owns: src/cs182_icl/eval/, tests/eval/, scripts/eval_*, scripts/plot_*, docs/handoffs/T06.md,
docs/questions/T06.md, docs/runbook/T06.md

Deliverables
- run any predictor (model checkpoint, Bayes, physics fit, oracle) over a test file; cache
  reference predictions once, reuse across seeds
- metrics: MSE vs clean simulated target; NMSE with shared denominator D = training-prior
  Bayes c=0 on the same test set; absolute gap to Bayes; median relative error; tertiles;
  heavy-near-threshold subset; physics-fit noisy reporting rules (mean only c>=4)
- seeds: mean +- SE; paired differences on shared items
- tests: (a) tiny hand-calculated metric fixture with numbers worked out by hand;
  (b) Bayes-as-model reproduces Bayes metrics exactly
- plots for E1-E4 + force-separation sweep; one figure per file, CSV beside it
